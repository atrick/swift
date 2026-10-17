//===--- AccessEnforcementSelection.cpp - Select access enforcement -------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2017 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
///
/// This pass eliminates 'unknown' access enforcement by selecting either
/// static or dynamic enforcement.
///
/// TODO: This is currently a module transform so that it can process closures
/// after analyzing their parent scope. This isn't a big problem now because
/// AccessMarkerElimination is also a module pass that follows this pass, so all
/// markers will still be present when this pass runs. However, we would like to
/// mostly eliminate module transforms. This could be done by changing the
/// PassManager to follow ClosureScopeAnalysis. A new ClosureTransform type
/// would be pipelined just like FunctionTransform, but would have an entry
/// point that handled a parent closure scope and all its children in one
/// invocation. For function pipelining to be upheld, we would need to verify
/// that BasicCalleeAnalysis never conflicts with ClosureScopeAnalysis. i.e. we
/// could never create a caller->callee edge when the callee is passed as a
/// function argument. Normal FunctionTransforms would then be called on each
/// closure function and its parent scope before calling the ClosureTransform.
///
//===----------------------------------------------------------------------===//

#include "swift/SIL/SILInstruction.h"
#define DEBUG_TYPE "access-enforcement-selection"
#include "swift/Basic/Defer.h"
#include "swift/SIL/ApplySite.h"
#include "swift/SIL/BasicBlockBits.h"
#include "swift/SIL/BasicBlockDatastructures.h"
#include "swift/SIL/InstructionUtils.h"
#include "swift/SIL/MemAccessUtils.h"
#include "swift/SIL/SILArgument.h"
#include "swift/SIL/SILFunction.h"
#include "swift/SIL/SILUndef.h"
#include "swift/SILOptimizer/Analysis/AccessSummaryAnalysis.h"
#include "swift/SILOptimizer/Analysis/ClosureScope.h"
#include "swift/SILOptimizer/Analysis/PostOrderAnalysis.h"
#include "swift/SILOptimizer/PassManager/Transforms.h"
#include "swift/SILOptimizer/Utils/BoxAccessSummary.h"

using namespace swift;

static void setStaticEnforcement(BeginAccessInst *access) {
  // TODO: delete if we're not using static enforcement?
  access->setEnforcement(SILAccessEnforcement::Static);

  LLVM_DEBUG(llvm::dbgs() << "Static Access: " << *access);
}

static void setDynamicEnforcement(BeginAccessInst *access) {
  // TODO: delete if we're not using dynamic enforcement?
  access->setEnforcement(SILAccessEnforcement::Dynamic);

  LLVM_DEBUG(llvm::dbgs() << "Dynamic Access: " << *access);
}

namespace {
// Information about an address-type closure capture.
// This is only valid for inout_aliasable parameters.
//
// TODO: Verify somewhere that we properly handle any non-inout_aliasable
// partial apply captures or that they never happen. Eventually @inout_aliasable
// should be simply replaced by @in or @out, once we don't have special aliasing
// rules.
struct AddressCapture {
  ApplySite site;
  unsigned calleeArgIdx;

  AddressCapture(Operand &oper)
      : site(oper.getUser()), calleeArgIdx(site.getSubstCalleeArgIndex(oper)) {
    if (site.getOrigCalleeConv().getSILArgumentConvention(calleeArgIdx)
        != SILArgumentConvention::Indirect_InoutAliasable) {
      site = ApplySite();
      calleeArgIdx = ~0U;
      return;
    }
    assert(oper.get()->getType().isAddress());
  }

  bool isValid() const { return bool(site); }
};

LLVM_ATTRIBUTE_UNUSED
raw_ostream &operator<<(raw_ostream &os, const AddressCapture &capture) {
  os << *capture.site.getInstruction() << " captures Arg #"
     << capture.calleeArgIdx;
  auto *F = capture.site.getCalleeFunction();
  if (F)
    os << " of " << F->getName();
  os << '\n';
  return os;
}

// For each non-escaping closure, record the indices of arguments that
// require dynamic enforcement.
//
// A note on closure cycles: local functions can be recursive, creating closure
// cycles. DynamicCaptures ignores such cycles, simply processing the call graph
// top-down. This relies on a simple rule: if a captured variable is passed as a
// box a local function (presumably because the function escapes), then it must
// also be passed as a box to any other local function called by the
// first. Therefore, if any capture escapes in a closure cycle, then it must be
// passed as a box in all closures within the cycle. DynamicCaptures does not
// care about boxes, because they are always dynamically enforced.
class DynamicCaptures {
  // This only maps functions that have at least one inout_aliasable argument.
  llvm::DenseMap<SILFunction *, SmallVector<unsigned, 4>> dynamicCaptureMap;

  DynamicCaptures(DynamicCaptures &) = delete;

public:
  DynamicCaptures() {}

  void recordCapture(AddressCapture capture) {
    LLVM_DEBUG(llvm::dbgs() << "Dynamic Capture: " << capture);

    // *NOTE* For dynamically replaceable local functions, getCalleeFunction()
    // returns nullptr. This assert verifies the assumption that a captured
    // local variable can never be promoted to capture-by-address for
    // dynamically replaceable local functions.
    auto callee = capture.site.getCalleeFunction();
    assert(callee && "cannot locate function ref for nonescaping closure");

    auto &dynamicArgs = dynamicCaptureMap[callee];
    if (!llvm::is_contained(dynamicArgs, capture.calleeArgIdx))
      dynamicArgs.push_back(capture.calleeArgIdx);
  }

  bool isDynamic(SILFunctionArgument *arg) const {
    // This closure may be the head of a closure cycle. That's ok, because we
    // only care about whether this argument escapes in the calling function
    // this is *not* part of the cycle. If the capture escapes anywhere in the
    // cycle, then it is passed as a box to all closures in that cycle.

    auto pos = dynamicCaptureMap.find(arg->getFunction());
    if (pos == dynamicCaptureMap.end())
      return false;

    auto &dynamicArgs = pos->second;
    return llvm::is_contained(dynamicArgs, arg->getIndex());
  }
};

// Track box arguments of functions that the box is passed to without escaping:
// local functions that capture a variable by box and are called directly, and
// closures that capture a box and are only called, or passed to functions that
// only call them.
//
// Static enforcement must be equivalent to dynamic enforcement. A box passed
// to a call does not escape if the callee does not escape its box argument, no
// other argument of the call reaches the box, and any access in progress at the
// call either cannot conflict with an access during the call, or conflicts on
// every path through the call, so that DiagnoseStaticExclusivity reports it.
//
// The callee argument is recorded as dynamic if the box has potentially escaped
// at some call site or if the callee has unknown uses. A call that passes a box
// to a dynamic argument is an escape of the box in the caller. This is solved
// as a fixed point, so that an access in the caller is only static if all
// accesses to the same box within the callees are also static.
class BoxArguments {
  // Memoized results of mayEscape().
  llvm::DenseMap<SILFunctionArgument *, bool> escapeCache;

  // Memoized results of closureArgumentMayEscape().
  llvm::DenseMap<SILFunctionArgument *, bool> closureEscapeCache;

  // Callee box arguments that require dynamic enforcement.
  llvm::DenseSet<SILFunctionArgument *> dynamicArgs;

  // Box arguments of direct calls that are analyzed by SelectEnforcement.
  llvm::SmallPtrSet<Operand *, 8> boxCalls;

  BoxArguments(BoxArguments &) = delete;

  BoxAccessSummary &boxSummary;

  // The accesses to each box argument within its function.
  llvm::DenseMap<SILFunctionArgument *, SmallVector<BeginAccessInst *, 4>>
      argumentAccesses;

public:
  BoxArguments(BoxAccessSummary &boxSummary) : boxSummary(boxSummary) {}

  /// Collect the formal accesses to the storage of \p box within its function.
  static void findAccesses(SILValue box,
                           SmallVectorImpl<BeginAccessInst *> &accesses) {
    SmallVector<SILValue, 8> worklist;
    llvm::SmallPtrSet<SILValue, 8> visited;
    auto push = [&](SILValue value) {
      if (visited.insert(value).second)
        worklist.push_back(value);
    };
    push(box);
    while (!worklist.empty()) {
      SILValue value = worklist.pop_back_val();
      for (Operand *use : value->getUses()) {
        SILInstruction *user = use->getUser();
        switch (user->getKind()) {
        case SILInstructionKind::BeginBorrowInst:
        case SILInstructionKind::CopyValueInst:
        case SILInstructionKind::MoveValueInst:
        case SILInstructionKind::MarkUninitializedInst:
        case SILInstructionKind::ProjectBoxInst:
        case SILInstructionKind::MarkUnresolvedNonCopyableValueInst:
          push(cast<SingleValueInstruction>(user));
          continue;
        case SILInstructionKind::BeginAccessInst:
          accesses.push_back(cast<BeginAccessInst>(user));
          continue;
        default:
          continue;
        }
      }
    }
  }

  /// Return true if \p inst is within the scope of \p access. Set \p
  /// mustBeInScope if \p access is in progress whenever \p inst executes.
  static bool isInAccessScope(BeginAccessInst *access, SILInstruction *inst,
                              bool &mustBeInScope) {
    mustBeInScope = false;
    // Scan a block starting at \p start. Return true if inst is found before
    // the access ends, and set \p ended if the access ends.
    auto scan = [&](SILBasicBlock::iterator start, SILBasicBlock *block,
                    bool &ended) {
      for (auto it = start; it != block->end(); ++it) {
        if (&*it == inst)
          return true;
        if (auto *end = dyn_cast<EndAccessInst>(&*it)) {
          if (end->getBeginAccess() == access) {
            ended = true;
            return false;
          }
        }
      }
      return false;
    };
    bool ended = false;
    SILBasicBlock *accessBlock = access->getParent();
    if (scan(std::next(access->getIterator()), accessBlock, ended)) {
      mustBeInScope = true;
      return true;
    }
    if (ended)
      return false;

    BasicBlockWorklist worklist(access->getFunction());
    for (SILBasicBlock *succ : accessBlock->getSuccessorBlocks())
      worklist.pushIfNotVisited(succ);
    while (SILBasicBlock *block = worklist.pop()) {
      ended = false;
      if (scan(block->begin(), block, ended))
        return true;
      if (ended)
        continue;
      for (SILBasicBlock *succ : block->getSuccessorBlocks())
        worklist.pushIfNotVisited(succ);
    }
    return false;
  }

  /// Return true if static enforcement of the accesses to the box whose
  /// reference root is \p boxRoot is equivalent to dynamic enforcement at \p
  /// apply, given the accesses to the box within the caller, \p accesses.
  ///
  /// The callee may access the box, through a box argument or a closure that
  /// captures the box. An access in progress at the call that may conflict
  /// with such an access is only checked statically if the conflict happens
  /// every time the call executes. Then DiagnoseStaticExclusivity reports it.
  /// Otherwise, only dynamic enforcement can check it.
  bool isCallSafe(FullApplySite apply, SILValue boxRoot,
                  ArrayRef<BeginAccessInst *> accesses) {
    BoxAccesses during = boxSummary.getAccessesDuringCall(apply, boxRoot);
    if (!during.mayAccess)
      return true;

    bool mayConflict = false;
    for (BeginAccessInst *access : accesses) {
      if (access->getEnforcement() == SILAccessEnforcement::Unsafe)
        continue;

      bool mustBeInScope;
      if (!isInAccessScope(access, apply.getInstruction(), mustBeInScope))
        continue;

      SILAccessKind kind = access->getAccessKind();
      if (mustBeInScope && during.mustConflictWith(kind) &&
          apply.getModule().getOptions().EnforceExclusivityStatic) {
        return true;
      }
      mayConflict |= during.mayConflictWith(kind);
    }
    if (mayConflict) {
      LLVM_DEBUG(llvm::dbgs() << "    Call may conflict with an access: "
                              << *apply.getInstruction());
    }
    return !mayConflict;
  }

  /// The accesses to box argument \p arg within its function.
  ArrayRef<BeginAccessInst *> getArgumentAccesses(SILFunctionArgument *arg) {
    auto found = argumentAccesses.try_emplace(arg);
    if (found.second)
      findAccesses(arg, found.first->second);
    return found.first->second;
  }

  /// If \p oper passes a box as an argument to a function_ref callee, return
  /// the callee's corresponding argument. This includes a box captured by a
  /// partial_apply.
  static SILFunctionArgument *getCalleeBoxArgument(Operand &oper) {
    if (!oper.get()->getType().is<SILBoxType>())
      return nullptr;

    return getCalleeArgument(oper);
  }

  /// If \p oper is an argument of an apply of a function defined in this
  /// module, return the callee's corresponding argument.
  static SILFunctionArgument *getCalleeArgument(Operand &oper) {
    auto site = ApplySite::isa(oper.getUser());
    if (!site || !site.isArgumentOperand(oper))
      return nullptr;

    // This excludes dynamically replaceable callees. A function that is
    // available externally may be replaced by a different implementation.
    SILFunction *callee = site.getReferencedFunctionOrNull();
    if (!callee || !callee->isDefinition() ||
        isAvailableExternally(callee->getLinkage())) {
      return nullptr;
    }
    return cast<SILFunctionArgument>(
        callee->getArgument(site.getSubstCalleeArgIndex(oper)));
  }

  /// Return true if \p boxOper is the only argument of its apply that may
  /// reach the box's storage.
  ///
  /// The callee's static accesses to its box argument can only be checked
  /// against other accesses through the same argument. If the same storage is
  /// also reachable through another argument, such as the box passed again, an
  /// address projected from the box, or a closure that captures either, then
  /// two accesses in the callee may overlap without being diagnosed. Such a
  /// call is an escape of the box.
  static bool isOnlyArgumentForBox(Operand &boxOper) {
    return isOnlyArgumentForBox(boxOper, findReferenceRoot(boxOper.get()));
  }

  /// Return true if no argument of \p oper's apply other than \p oper may reach
  /// the storage of the box whose reference root is \p boxRoot.
  ///
  /// An address within an access scope is an access in progress at the call,
  /// which isCallSafe checks.
  static bool isOnlyArgumentForBox(Operand &oper, SILValue boxRoot) {
    auto site = ApplySite(oper.getUser());
    llvm::SmallPtrSet<SILValue, 8> visited;
    for (Operand &otherOper : site.getArgumentOperands()) {
      if (&otherOper == &oper)
        continue;

      SILValue value = otherOper.get();
      if (value->getType().isAddress() &&
          isa<BeginAccessInst>(getAccessScope(value))) {
        continue;
      }
      if (mayReachBox(value, boxRoot, visited)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "    Box passed with an alias: " << *oper.getUser());
        return false;
      }
    }
    return true;
  }

  /// Return true if the box argument of a direct call may escape the callee.
  bool mayEscape(SILFunctionArgument *arg) {
    auto pos = escapeCache.find(arg);
    if (pos != escapeCache.end())
      return pos->second;

    bool escapes = computeMayEscape(arg);
    escapeCache[arg] = escapes;
    return escapes;
  }

  /// Find the calls during which \p closure may run, given that it captures
  /// the box whose reference root is \p boxRoot. Return false if the closure
  /// may escape, so that it may run at any time.
  ///
  /// An escaping closure that captures a box does not escape if it is only
  /// called, or passed to callees that only call it or pass it on. No other
  /// argument of those calls may reach the box.
  bool findClosureCalls(SingleValueInstruction *closure, SILValue boxRoot,
                        SmallVectorImpl<SILInstruction *> &calls) {
    SmallVector<SILValue, 4> worklist;
    llvm::SmallPtrSet<SILValue, 4> visited;
    auto push = [&](SILValue value) {
      if (visited.insert(value).second)
        worklist.push_back(value);
    };
    push(closure);
    while (!worklist.empty()) {
      SILValue value = worklist.pop_back_val();
      for (Operand *use : value->getUses()) {
        SILInstruction *user = use->getUser();
        if (isClosureForwarding(use)) {
          push(cast<SingleValueInstruction>(user));
          continue;
        }
        if (isIncidentalClosureUse(user))
          continue;

        if (auto apply = FullApplySite::isa(user)) {
          if (apply.isCalleeOperand(*use)) {
            if (isOnlyArgumentForBox(*use, boxRoot)) {
              calls.push_back(user);
              continue;
            }
            LLVM_DEBUG(llvm::dbgs() << "    Closure escape: " << *user);
            return false;
          }
          auto *calleeArg = getCalleeArgument(*use);
          if (calleeArg && !closureArgumentMayEscape(calleeArg) &&
              isOnlyArgumentForBox(*use, boxRoot)) {
            calls.push_back(user);
            continue;
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "    Closure escape: " << *user);
        return false;
      }
    }
    return true;
  }

  /// Return true if the closure argument of a direct call may escape the
  /// callee.
  bool closureArgumentMayEscape(SILFunctionArgument *arg) {
    auto pos = closureEscapeCache.find(arg);
    if (pos != closureEscapeCache.end())
      return pos->second;

    bool escapes = computeClosureArgumentMayEscape(arg);
    closureEscapeCache[arg] = escapes;
    return escapes;
  }

  void recordBoxCall(Operand *oper) { boxCalls.insert(oper); }

  bool isBoxCall(Operand *oper) const { return boxCalls.count(oper); }

  bool recordDynamic(SILFunctionArgument *arg) {
    if (!dynamicArgs.insert(arg).second)
      return false;

    LLVM_DEBUG(llvm::dbgs() << "Dynamic Box Argument: " << *arg << " of "
                            << arg->getFunction()->getName() << '\n');
    return true;
  }

  bool isDynamic(SILFunctionArgument *arg) const {
    return dynamicArgs.count(arg);
  }

private:
  // Return true if \p value may refer to the storage of the box whose
  // reference root is \p boxRoot: the box itself, an address within it, or a
  // closure that captures either.
  static bool mayReachBox(SILValue value, SILValue boxRoot,
                          llvm::SmallPtrSetImpl<SILValue> &visited) {
    if (!visited.insert(value).second)
      return false;

    SILType type = value->getType();
    if (type.isAddress()) {
      AccessBase base = AccessBase::compute(value);
      if (!base || base.getKind() == AccessBase::Unidentified)
        return true;
      return base.getKind() == AccessBase::Box &&
             findReferenceRoot(base.getReference()) == boxRoot;
    }
    if (type.is<SILBoxType>())
      return findReferenceRoot(value) == boxRoot;

    if (auto objectType = type.getOptionalObjectType())
      type = objectType;
    auto fnType = type.getAs<SILFunctionType>();
    if (!fnType)
      return false;

    // The closure stored in an escaping block can't be found.
    if (!fnType->isNoEscape() &&
        fnType->getRepresentation() == SILFunctionTypeRepresentation::Block) {
      return true;
    }
    TinyPtrVector<PartialApplyInst *> closures;
    findClosuresForFunctionValue(value, closures);
    for (PartialApplyInst *closure : closures) {
      for (Operand &captured : closure->getArgumentOperands()) {
        if (mayReachBox(captured.get(), boxRoot, visited))
          return true;
      }
    }
    return false;
  }

  // Visit all uses of the box argument, following it into direct callees.
  bool computeMayEscape(SILFunctionArgument *root) {
    SmallVector<SILValue, 8> worklist;
    llvm::SmallPtrSet<SILValue, 8> visited;
    auto push = [&](SILValue value) {
      if (visited.insert(value).second)
        worklist.push_back(value);
    };
    push(root);
    while (!worklist.empty()) {
      SILValue value = worklist.pop_back_val();
      if (auto *arg = dyn_cast<SILFunctionArgument>(value)) {
        if (arg->getArgumentConvention() !=
            SILArgumentConvention::Direct_Guaranteed) {
          return true;
        }
      }
      for (Operand *use : value->getUses()) {
        SILInstruction *user = use->getUser();
        switch (user->getKind()) {
        case SILInstructionKind::ProjectBoxInst:
        case SILInstructionKind::DebugValueInst:
        case SILInstructionKind::EndBorrowInst:
        case SILInstructionKind::DestroyValueInst:
          continue;
        case SILInstructionKind::BeginBorrowInst:
        case SILInstructionKind::CopyValueInst:
        case SILInstructionKind::MoveValueInst:
          push(cast<SingleValueInstruction>(user));
          continue;
        default:
          break;
        }
        if (auto apply = FullApplySite::isa(user)) {
          auto *calleeArg = getCalleeBoxArgument(*use);
          auto *callerArg =
              dyn_cast<SILFunctionArgument>(findReferenceRoot(value));
          if (calleeArg && callerArg && isOnlyArgumentForBox(*use) &&
              isCallSafe(apply, callerArg, getArgumentAccesses(callerArg))) {
            push(calleeArg);
            continue;
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "    Box argument escape: " << *user);
        return true;
      }
    }
    return false;
  }

  // Return true if \p use forwards a closure value to its user's result.
  static bool isClosureForwarding(Operand *use) {
    switch (use->getUser()->getKind()) {
    case SILInstructionKind::BeginBorrowInst:
    case SILInstructionKind::CopyValueInst:
    case SILInstructionKind::MoveValueInst:
    case SILInstructionKind::ConvertFunctionInst:
    case SILInstructionKind::ConvertEscapeToNoEscapeInst:
      return true;
    case SILInstructionKind::MarkDependenceInst:
      return use->getOperandNumber() == MarkDependenceInst::Dependent;
    default:
      return false;
    }
  }

  // Return true if \p user neither calls nor escapes a closure.
  static bool isIncidentalClosureUse(SILInstruction *user) {
    return isa<DestroyValueInst>(user) || isa<EndBorrowInst>(user) ||
           isa<DebugValueInst>(user) || isa<DeallocStackInst>(user) ||
           isa<DestroyNotEscapedClosureInst>(user);
  }

  // Visit all uses of the closure argument, following it into direct callees.
  bool computeClosureArgumentMayEscape(SILFunctionArgument *root) {
    if (!root->getType().is<SILFunctionType>())
      return true;

    SmallVector<SILValue, 8> worklist;
    llvm::SmallPtrSet<SILValue, 8> visited;
    auto push = [&](SILValue value) {
      if (visited.insert(value).second)
        worklist.push_back(value);
    };
    push(root);
    while (!worklist.empty()) {
      SILValue value = worklist.pop_back_val();
      for (Operand *use : value->getUses()) {
        SILInstruction *user = use->getUser();
        if (isClosureForwarding(use)) {
          push(cast<SingleValueInstruction>(user));
          continue;
        }
        if (isIncidentalClosureUse(user))
          continue;

        if (auto apply = FullApplySite::isa(user)) {
          if (apply.isCalleeOperand(*use))
            continue;

          auto *calleeArg = getCalleeArgument(*use);
          if (calleeArg && calleeArg->getType().is<SILFunctionType>()) {
            push(calleeArg);
            continue;
          }
        }
        LLVM_DEBUG(llvm::dbgs() << "    Closure argument escape: " << *user);
        return true;
      }
    }
    return false;
  }
};
} // anonymous namespace

namespace {
class SelectEnforcement {
  // Reference back to the known dynamically enforced non-escaping closure
  // arguments in this module. Parent scopes are processed before the closures
  // they reference.
  DynamicCaptures &dynamicCaptures;

  BoxArguments &boxArguments;

  AllocBoxInst *Box;

  /// A state for tracking escape information about a variable.
  /// StateMap only has entries for blocks for which the variable
  /// has potentially escaped at exit.
  struct State {
    bool IsInWorklist = false;

    // At least one of the following must be true.
    bool HasEscape = false;
    bool HasPotentiallyEscapedAtEntry = false;

    // In a more advanced problem, this could easily be passed a State.
    bool adjustForEscapeInPredecessor() {
      bool updateSuccessors = false;

      if (!HasPotentiallyEscapedAtEntry) {
        HasPotentiallyEscapedAtEntry = true;
        updateSuccessors = !HasEscape;
      }

      return updateSuccessors;
    }
  };
  llvm::DenseMap<SILBasicBlock*, State> StateMap;

  /// All the accesses of Box in the function.
  SmallVector<BeginAccessInst*, 8> Accesses;

  /// All the non-escaping closure captures of the Boxed value in this function.
  SmallVector<AddressCapture, 8> Captures;

  /// A use that passes the box to a function without escaping it: a direct
  /// call, or a closure that captures the box. Calls are the calls during
  /// which the function may access the box: the direct call, or the calls that
  /// invoke the closure or take it as an argument.
  struct BoxUse {
    Operand *operand;
    SmallVector<SILInstruction *, 2> calls;

    SILFunctionArgument *getCalleeArgument() const {
      return BoxArguments::getCalleeBoxArgument(*operand);
    }
  };

  /// All the uses that pass the box to a function without escaping it, unless
  /// the function's argument requires dynamic enforcement.
  SmallVector<BoxUse, 4> BoxUses;

  /// All the uses of the box that always escape it.
  SmallVector<SILInstruction *, 4> BoxEscapes;

  /// All the escapes in the function.
  SmallPtrSet<SILInstruction*, 8> Escapes;

  /// A worklist we use for various purposes.
  SmallVector<SILBasicBlock*, 8> Worklist;

public:
  SelectEnforcement(DynamicCaptures &dc, BoxArguments &ba, AllocBoxInst *box)
      : dynamicCaptures(dc), boxArguments(ba), Box(box) {}

  AllocBoxInst *getBox() const { return Box; }

  /// Find all accesses, captures, escapes, and calls that take the box.
  void analyze();

  /// Compute the escaping blocks given the current dynamic box arguments and
  /// record any box arguments that require dynamic enforcement. Return true if
  /// a new dynamic box argument was found.
  bool selectBoxArguments();

  /// Update all the accesses and captures once the box arguments are final.
  void updateAccesses();

private:
  void analyzeUsesOfBox(SingleValueInstruction *source);
  // Used for project_box and mark_must_initialize.
  void analyzeProjection(SingleValueInstruction *project);

  /// Note that the given instruction is a use of the box (or a use of
  /// a projection from it) in which the address escapes.
  void noteEscapingUse(SILInstruction *inst);

  void propagateEscapes();
  void propagateEscapesFrom(SILBasicBlock *bb);

  bool hasPotentiallyEscapedAt(SILInstruction *inst);

  typedef llvm::SmallSetVector<SILBasicBlock*, 8> BlockSetVector;
  void findBlocksAccessedAcross(EndAccessInst *endAccess,
                                BlockSetVector &blocksAccessedAcross);
  bool hasPotentiallyEscapedAtAnyReachableBlock(
    BeginAccessInst *access, BlockSetVector &blocksAccessedAcross);

  void computeEscapes();

  void updateAccess(BeginAccessInst *access);
  void updateCapture(AddressCapture capture);
};
} // end anonymous namespace

void SelectEnforcement::analyze() {
  LLVM_DEBUG(llvm::dbgs() << "  Box: " << *Box);

  analyzeUsesOfBox(Box);
  if (BoxUses.empty())
    return;

  // A call that may conflict with an access in progress is an escape. This
  // includes accesses whose enforcement is already known.
  SmallVector<BeginAccessInst *, 8> allAccesses;
  BoxArguments::findAccesses(Box, allAccesses);
  llvm::erase_if(BoxUses, [&](const BoxUse &use) {
    for (SILInstruction *call : use.calls) {
      if (!boxArguments.isCallSafe(FullApplySite(call), Box, allAccesses)) {
        LLVM_DEBUG(llvm::dbgs() << "    Escape: " << *use.operand->getUser());
        BoxEscapes.push_back(use.operand->getUser());
        return true;
      }
    }
    boxArguments.recordBoxCall(use.operand);
    return false;
  });
}

// Set up and run the data-flow problem.
void SelectEnforcement::computeEscapes() {
  StateMap.clear();
  Escapes.clear();
  for (SILInstruction *escape : BoxEscapes)
    noteEscapingUse(escape);

  // The function may access the box dynamically during each of the calls.
  for (const BoxUse &use : BoxUses) {
    if (boxArguments.isDynamic(use.getCalleeArgument())) {
      for (SILInstruction *call : use.calls)
        noteEscapingUse(call);
    }
  }
  propagateEscapes();
}

bool SelectEnforcement::selectBoxArguments() {
  computeEscapes();

  bool changed = false;
  for (const BoxUse &use : BoxUses) {
    if (llvm::any_of(use.calls, [&](SILInstruction *call) {
          return hasPotentiallyEscapedAt(call);
        })) {
      changed |= boxArguments.recordDynamic(use.getCalleeArgument());
    }
  }
  return changed;
}

// FIXME: This should cover a superset of AllocBoxToStack's findUnexpectedBoxUse
// to avoid perturbing codegen. They should be sharing the same analysis.
void SelectEnforcement::analyzeUsesOfBox(SingleValueInstruction *source) {
  // Collect accesses rooted off of projections.
  for (auto use : source->getUses()) {
    auto user = use->getUser();

    if (auto bbi = dyn_cast<BeginBorrowInst>(user)) {
      analyzeUsesOfBox(bbi);
      continue;
    }

    if (auto mui = dyn_cast<MarkUninitializedInst>(user)) {
      analyzeUsesOfBox(mui);
      continue;
    }

    // A copy of the box only escapes it through its own uses.
    if (isa<CopyValueInst>(user) || isa<MoveValueInst>(user)) {
      analyzeUsesOfBox(cast<SingleValueInstruction>(user));
      continue;
    }

    if (auto projection = dyn_cast<ProjectBoxInst>(user)) {
      analyzeProjection(projection);
      continue;
    }
      
    // Ignore certain other uses that do not capture the value.
    if (isa<StrongRetainInst>(user) || isa<StrongReleaseInst>(user) ||
        isa<DestroyValueInst>(user) || isa<DeallocBoxInst>(user) ||
        isa<EndBorrowInst>(user))
      continue;

    // A direct call does not escape the box if the callee doesn't, and no
    // other argument of the call reaches the box.
    if (FullApplySite::isa(user)) {
      if (auto *calleeArg = BoxArguments::getCalleeBoxArgument(*use)) {
        if (!boxArguments.mayEscape(calleeArg) &&
            BoxArguments::isOnlyArgumentForBox(*use)) {
          BoxUses.push_back({use, {user}});
          continue;
        }
      }
    }

    // A closure that captures the box does not escape it if the closure body
    // doesn't, no other capture reaches the box, and the closure itself does
    // not escape: it is only called, or passed to functions that only call it.
    if (auto *closure = dyn_cast<PartialApplyInst>(user)) {
      if (auto *calleeArg = BoxArguments::getCalleeBoxArgument(*use)) {
        SmallVector<SILInstruction *, 2> calls;
        if (!boxArguments.mayEscape(calleeArg) &&
            BoxArguments::isOnlyArgumentForBox(*use) &&
            boxArguments.findClosureCalls(closure, Box, calls)) {
          BoxUses.push_back({use, std::move(calls)});
          continue;
        }
      }
    }

    // Treat everything else as an escape.
    // A Box typically escapes via copy_value.
    LLVM_DEBUG(llvm::dbgs() << "    Escape: " << *user);
    BoxEscapes.push_back(user);
  }
  // Accesses may still be empty if the user of the Box is a partial apply
  // capture and, for some reason, the closure is dead.
}

// Verify that accesses are not nested before mandatory inlining.
// Closure captures should also not be nested within an access.
static void checkUsesOfAccess(BeginAccessInst *access) {
#ifndef NDEBUG
  // These conditions are only true prior to mandatory inlining.
  assert(!access->getFunction()->wasDeserializedCanonical());
  for (auto *use : access->getUses()) {
    auto user = use->getUser();
    assert(!isa<BeginAccessInst>(user));
    assert(!isa<PartialApplyInst>(user) ||
           onlyUsedByAssignOrInit(cast<PartialApplyInst>(user)));
  }
#endif
}

void SelectEnforcement::analyzeProjection(SingleValueInstruction *projection) {
  for (auto *use : projection->getUses()) {
    auto user = use->getUser();

    // Look through mark must check.
    if (auto *mmi = dyn_cast<MarkUnresolvedNonCopyableValueInst>(user)) {
      analyzeProjection(mmi);
      continue;
    }

    // Collect accesses.
    if (auto *access = dyn_cast<BeginAccessInst>(user)) {
      if (access->getEnforcement() == SILAccessEnforcement::Unknown)
        Accesses.push_back(access);

      checkUsesOfAccess(access);

      continue;
    }
    // Handle both partial applies and directly applied non-escaping closures.
    if (ApplySite::isa(user)) {
      AddressCapture capture(*use);
      if (capture.isValid())
        Captures.emplace_back(capture);
      else
        // Only full apply sites can have non-inout_aliasable address arguments,
        // but those aren't actually captures.
        assert(FullApplySite::isa(user));
    }
  }
}

void SelectEnforcement::noteEscapingUse(SILInstruction *inst) {
  // Add it to the escapes set.
  Escapes.insert(inst);

  // Record this point as escaping.
  auto userBB = inst->getParent();
  auto &state = StateMap[userBB];
  if (!state.IsInWorklist) {
    state.HasEscape = true;
    state.IsInWorklist = true;
    Worklist.push_back(userBB);
  }
  assert(state.HasEscape);
  assert(state.IsInWorklist);
}

void SelectEnforcement::propagateEscapes() {
  while (!Worklist.empty()) {
    auto bb = Worklist.pop_back_val();
    auto it = StateMap.find(bb);
    assert(it != StateMap.end() &&
           "block was in worklist but doesn't have a tracking state");
    auto &state = it->second;
    assert(state.HasEscape || state.HasPotentiallyEscapedAtEntry);
    state.IsInWorklist = false;
    propagateEscapesFrom(bb);
  }
}

/// Given that the box potentially escaped before we exited the
/// given block, propagate that information to all of its successors.
void SelectEnforcement::propagateEscapesFrom(SILBasicBlock *bb) {
  assert(StateMap.count(bb));

  // Iterate over the successors of the block.
  for (SILBasicBlock *succ : bb->getSuccessors()) {
    auto &succState = StateMap[succ];

    // If updating the successor changes it in a way that will
    // require us to update its successors, add it to the worklist.
    if (succState.adjustForEscapeInPredecessor()) {
      if (!succState.IsInWorklist) {
        succState.IsInWorklist = true;
        Worklist.push_back(succ);
      }
    }
  }
}

bool SelectEnforcement::hasPotentiallyEscapedAt(SILInstruction *point) {
  auto bb = point->getParent();

  // If we're not tracking anything for the whole block containing
  // the instruction, we're done; it hasn't escaped here.
  auto it = StateMap.find(bb);
  if (it == StateMap.end())
    return false;

  // If the tracking information says there are escapes before entry,
  // we're done; it has potentially escaped.
  const auto &state = it->second;
  if (state.HasPotentiallyEscapedAtEntry)
    return true;

  // Okay, there must be an escape within this block.
  assert(state.HasEscape);
  for (auto ii = point->getIterator(), ie = bb->begin(); ii != ie; ) {
    auto inst = &*--ii;

    // Maybe just record the first escape in the block and see if we
    // come after it?
    if (Escapes.count(inst))
      return true;
  }

  return false;
}

/// Add all blocks to `Worklist` between the given `endAccess` and its
/// `begin_access` in which the access is active at the end of the block.
void SelectEnforcement::findBlocksAccessedAcross(
  EndAccessInst *endAccess, BlockSetVector &blocksAccessedAcross) {

  // Fast path: we're not tracking any escapes.  (But the box should
  // probably have been promoted to the stack in this case.)
  if (StateMap.empty())
    return;

  SILBasicBlock *beginBB = endAccess->getBeginAccess()->getParent();
  if (endAccess->getParent() == beginBB)
    return;

  assert(Worklist.empty());
  Worklist.push_back(endAccess->getParent());
  while (!Worklist.empty()) {
    SILBasicBlock *bb = Worklist.pop_back_val();
    for (auto *predBB : bb->getPredecessorBlocks()) {
      if (!blocksAccessedAcross.insert(predBB)) continue;
      if (predBB == beginBB) continue;
      Worklist.push_back(predBB);
    }
  }
}

bool SelectEnforcement::hasPotentiallyEscapedAtAnyReachableBlock(
  BeginAccessInst *access, BlockSetVector &blocksAccessedAcross) {

  assert(Worklist.empty());
  BasicBlockSet visited(access->getFunction());

  // Don't follow any paths that lead to an end_access.
  for (auto endAccess : access->getEndAccesses())
    visited.insert(endAccess->getParent());

  /// Initialize the worklist with all blocks that exit the access path.
  for (SILBasicBlock *bb : blocksAccessedAcross) {
    for (SILBasicBlock *succBB : bb->getSuccessorBlocks()) {
      if (blocksAccessedAcross.count(succBB)) continue;
      if (visited.insert(succBB))
        Worklist.push_back(succBB);
    }
  }

  while (!Worklist.empty()) {
    SILBasicBlock *bb = Worklist.pop_back_val();
    assert(visited.contains(bb));

    // If we're tracking information for this block, there's an escape.
    if (StateMap.count(bb))
      return true;

    // Add all reachable successors.
    for (SILBasicBlock *succ : bb->getSuccessors()) {
      if (visited.insert(succ))
        Worklist.push_back(succ);
    }
  }

  // No reachable block has an escape.
  return false;
}

void SelectEnforcement::updateAccesses() {
  LLVM_DEBUG(llvm::dbgs() << "  Box: " << *Box);
  for (auto *access : Accesses) {
    LLVM_DEBUG(llvm::dbgs() << "    Access: " << *access);
    updateAccess(access);
  }
  for (AddressCapture &capture : Captures) {
    LLVM_DEBUG(llvm::dbgs() << "    Capture: " << capture);
    updateCapture(capture);
  }
}

void SelectEnforcement::updateAccess(BeginAccessInst *access) {
  assert(access->getEnforcement() == SILAccessEnforcement::Unknown);

  // Check whether the variable escaped before any of the end_accesses.
  BlockSetVector blocksAccessedAcross;
  for (auto endAccess : access->getEndAccesses()) {
    if (hasPotentiallyEscapedAt(endAccess))
      return setDynamicEnforcement(access);

    // Add all blocks to blocksAccessedAcross between begin_access and this
    // end_access.
    findBlocksAccessedAcross(endAccess, blocksAccessedAcross);
  }
  assert(blocksAccessedAcross.empty()
         || blocksAccessedAcross.count(access->getParent()));

  // For every path through this access that doesn't reach an end_access, check
  // if any block reachable from that path can see an escaped value.
  if (hasPotentiallyEscapedAtAnyReachableBlock(access, blocksAccessedAcross)) {
    setDynamicEnforcement(access);
    return;
  }
  // Otherwise, use static enforcement.
  setStaticEnforcement(access);
}

void SelectEnforcement::updateCapture(AddressCapture capture) {
  auto captureIfEscaped = [&](SILInstruction *user) {
    // A closure passed to the call that escapes the box, such as a call that
    // also takes the box itself, may access the box during that call.
    if (Escapes.count(user) || hasPotentiallyEscapedAt(user))
      dynamicCaptures.recordCapture(capture);
  };
  SingleValueInstruction *PAIUser = dyn_cast<PartialApplyInst>(capture.site);
  if (!PAIUser) {
    // This is a full apply site. Immediately record the capture and return.
    captureIfEscaped(capture.site.getInstruction());
    return;
  }
  // For partial applies, check all use points of the closure.
  llvm::SmallSetVector<SingleValueInstruction *, 8> worklist;
  auto visitUse = [&](Operand *oper) {
    auto *user = oper->getUser();
    if (FullApplySite::isa(user)) {
      // A call is considered a closure access regardless of whether it calls
      // the closure or accepts the closure as an argument.
      captureIfEscaped(user);
      return;
    }
    switch (user->getKind()) {
    case SILInstructionKind::ConvertEscapeToNoEscapeInst:
    case SILInstructionKind::MarkDependenceInst:
    case SILInstructionKind::ConvertFunctionInst:
    case SILInstructionKind::BeginBorrowInst:
    case SILInstructionKind::CopyValueInst:
    case SILInstructionKind::EnumInst:
    case SILInstructionKind::StructInst:
    case SILInstructionKind::TupleInst:
    case SILInstructionKind::PartialApplyInst:
      // Propagate the closure.
      worklist.insert(cast<SingleValueInstruction>(user));
      return;
    case SILInstructionKind::StrongRetainInst:
    case SILInstructionKind::StrongReleaseInst:
    case SILInstructionKind::DebugValueInst:
    case SILInstructionKind::DestroyValueInst:
    case SILInstructionKind::RetainValueInst:
    case SILInstructionKind::ReleaseValueInst:
    case SILInstructionKind::EndBorrowInst:
    // partial_apply [stack] is matched with dealloc_stack.
    case SILInstructionKind::DeallocStackInst:
      // Benign use.
      return;
    case SILInstructionKind::TupleExtractInst:
    case SILInstructionKind::StructExtractInst:
    case SILInstructionKind::AssignInst:
    case SILInstructionKind::BranchInst:
    case SILInstructionKind::CondBranchInst:
    case SILInstructionKind::ReturnInst:
    case SILInstructionKind::StoreInst:
      // These are all valid partial_apply users, however we don't expect them
      // to occur with non-escaping closures. Handle them conservatively just in
      // case they occur.
      LLVM_FALLTHROUGH;
    default:
      LLVM_DEBUG(llvm::dbgs() << "    Unrecognized partial_apply user: "
                              << *user);

      // Handle unknown uses conservatively by assuming a capture.
      captureIfEscaped(user);
    }
  };
  while (true) {
    for (auto *oper : PAIUser->getUses())
      visitUse(oper);
    if (worklist.empty())
      break;
    PAIUser = worklist.pop_back_val();
  }
}

namespace {

// Model the kind of access needed based on analyzing the access's source.
// This is either determined to be static or dynamic, or requires further
// analysis of a boxed variable.
struct SourceAccess {
  enum { StaticAccess, DynamicAccess, BoxAccess } kind;
  AllocBoxInst *allocBox;

  static SourceAccess getStaticAccess() { return {StaticAccess, nullptr}; }
  static SourceAccess getDynamicAccess() { return {DynamicAccess, nullptr}; }

  static SourceAccess getBoxedAccess(AllocBoxInst *inst) {
    return {BoxAccess, inst};
  }
};

/// The pass.
///
/// This can't be a SILFunctionTransform because DynamicCaptures need to be
/// recorded while analyzing a closure's parent scopes before processing the
/// closures.
///
/// TODO: Make this a "ClosureTransform". See the file-level comments above.
class AccessEnforcementSelection : public SILModuleTransform {
  // Track the known dynamically enforced non-escaping closure
  // arguments in this module. Parent scopes are processed before the closures
  // they reference.
  std::unique_ptr<DynamicCaptures> dynamicCaptures;

  // Track the box arguments of directly called functions that require dynamic
  // enforcement. All boxes in the module are processed before any accesses to
  // box arguments.
  std::unique_ptr<BoxArguments> boxArguments;

  // The accesses to boxes during calls.
  std::unique_ptr<BoxAccessSummary> boxSummary;

  // The enforcement selection for each box in the module.
  std::vector<std::unique_ptr<SelectEnforcement>> boxSelections;

#ifndef NDEBUG
  // Book-keeping to verify that a box is processed before all of its accesses
  // and captures are seen.
  llvm::DenseSet<AllocBoxInst *> handledBoxes;
#endif

public:
  void run() override;

protected:
  void analyzeBoxes(SILFunction *F);
  void selectBoxes();
  void processFunction(SILFunction *F);
  SourceAccess getAccessKindForBox(SILValue boxOperand);
  SourceAccess getSourceAccess(SILValue address);
  void handleApply(ApplySite apply);
  void handleAccess(BeginAccessInst *access);
};

void AccessEnforcementSelection::run() {
  auto *CSA = getAnalysis<ClosureScopeAnalysis>();
  ClosureFunctionOrder closureOrder(CSA);
  closureOrder.compute();

  dynamicCaptures = std::make_unique<DynamicCaptures>();
  boxSummary =
      std::make_unique<BoxAccessSummary>(getAnalysis<AccessSummaryAnalysis>());
  boxArguments = std::make_unique<BoxArguments>(*boxSummary);
  SWIFT_DEFER {
    dynamicCaptures.reset();
    boxArguments.reset();
    boxSummary.reset();
    boxSelections.clear();
#ifndef NDEBUG
    handledBoxes.clear();
#endif
  };

  // Analyze all boxes first, so that every call that passes a box is known
  // before selecting enforcement for box arguments.
  for (SILFunction &function : *getModule()) {
    analyzeBoxes(&function);
  }
  selectBoxes();

  for (SILFunction *function : closureOrder.getTopDownFunctions()) {
    this->processFunction(function);
  }
}

// Analyze all boxes. Even if they aren't accessed in this function, they may
// still have captures that require dynamic enforcement because the box has
// escaped prior to the capture.
void AccessEnforcementSelection::analyzeBoxes(SILFunction *F) {
  if (F->isExternalDeclaration() || F->wasDeserializedCanonical())
    return;

  LLVM_DEBUG(llvm::dbgs() << "Box Analysis in " << F->getName() << "\n");

  for (SILBasicBlock &bb : *F) {
    for (SILInstruction &inst : bb) {
      if (auto *box = dyn_cast<AllocBoxInst>(&inst)) {
        boxSelections.push_back(std::make_unique<SelectEnforcement>(
            *dynamicCaptures, *boxArguments, box));
        boxSelections.back()->analyze();
        assert(handledBoxes.insert(box).second);
      }
    }
  }
}

// Find the box arguments that require dynamic enforcement because a call
// passes a box that may have escaped, or because the function has unknown
// uses. Then select enforcement for all boxes.
void AccessEnforcementSelection::selectBoxes() {
  auto recordDynamicArguments = [&](SILFunction *function) {
    if (!function->isDefinition())
      return;
    for (SILArgument *arg : function->getArguments()) {
      if (arg->getType().is<SILBoxType>())
        boxArguments->recordDynamic(cast<SILFunctionArgument>(arg));
    }
  };
  // Box arguments that are passed on to a callee's box argument.
  SmallVector<std::pair<SILFunctionArgument *, SILFunctionArgument *>, 8>
      forwardedArgs;

  for (SILFunction &function : *getModule()) {
    if (function.isPossiblyUsedExternally() ||
        function.isDynamicallyReplaceable()) {
      recordDynamicArguments(&function);
    }
    for (SILBasicBlock &bb : function) {
      for (SILInstruction &inst : bb) {
        // Any use of a function other than calling it is an unknown use.
        if (auto *functionRef = dyn_cast<FunctionRefBaseInst>(&inst)) {
          bool onlyCalled =
              isa<FunctionRefInst>(functionRef) &&
              llvm::all_of(functionRef->getUses(), [](Operand *use) {
                auto site = ApplySite::isa(use->getUser());
                return site && site.getCalleeOperand() == use;
              });
          if (!onlyCalled)
            recordDynamicArguments(
                functionRef->getInitiallyReferencedFunction());
          continue;
        }
        auto site = ApplySite::isa(&inst);
        if (!site)
          continue;

        for (Operand &oper : site.getArgumentOperands()) {
          auto *calleeArg = BoxArguments::getCalleeBoxArgument(oper);
          if (!calleeArg || boxArguments->isBoxCall(&oper))
            continue;

          // A box argument passed to a direct call that does not escape it.
          if (FullApplySite::isa(&inst) &&
              !boxArguments->mayEscape(calleeArg) &&
              BoxArguments::isOnlyArgumentForBox(oper)) {
            SILValue root = findReferenceRoot(oper.get());
            auto *arg = dyn_cast<SILFunctionArgument>(root);
            if (arg && arg->getType().is<SILBoxType>()) {
              forwardedArgs.push_back({arg, calleeArg});
              continue;
            }
          }
          boxArguments->recordDynamic(calleeArg);
        }
      }
    }
  }
  // Iterate until no more box arguments become dynamic. A dynamic argument may
  // cause a box to escape at a call, and an escaped box may cause an argument
  // to be dynamic.
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &boxSelection : boxSelections)
      changed |= boxSelection->selectBoxArguments();

    // A forwarded box argument is dynamic if the caller's argument is. The
    // caller's argument is also dynamic if the callee's argument is, so that
    // the caller's caller treats the call as an escape.
    for (auto [callerArg, calleeArg] : forwardedArgs) {
      if (boxArguments->isDynamic(callerArg))
        changed |= boxArguments->recordDynamic(calleeArg);
      if (boxArguments->isDynamic(calleeArg))
        changed |= boxArguments->recordDynamic(callerArg);
    }
  }
#ifndef NDEBUG
  SILFunction *currentFunction = nullptr;
#endif
  for (auto &boxSelection : boxSelections) {
    LLVM_DEBUG(if (boxSelection->getBox()->getFunction() != currentFunction) {
      currentFunction = boxSelection->getBox()->getFunction();
      llvm::dbgs() << "Box Enforcement Selection in "
                   << currentFunction->getName() << "\n";
    });
    boxSelection->updateAccesses();
  }
}

void AccessEnforcementSelection::
processFunction(SILFunction *F) {
  if (F->isExternalDeclaration())
    return;

  LLVM_DEBUG(llvm::dbgs() << "Access Enforcement Selection in " << F->getName()
                          << "\n");

  // Deserialized functions, which have been mandatory inlined, no longer meet
  // the structural requirements on access markers required by this pass.
  if (F->wasDeserializedCanonical())
    return;

  // Perform an RPO walk so that boxes are always processed before their access.
  auto *PO = getAnalysis<PostOrderAnalysis>()->get(F);
  for (SILBasicBlock *bb : PO->getReversePostOrder()) {
    for (auto ii = bb->begin(), ie = bb->end(); ii != ie;) {
      SILInstruction *inst = &*ii;
      ++ii;

      // Boxes have already been handled by selectBoxes.
      if (auto access = dyn_cast<BeginAccessInst>(inst))
        handleAccess(access);

      else if (auto access = dyn_cast<BeginUnpairedAccessInst>(inst))
        assert(access->getEnforcement() == SILAccessEnforcement::Dynamic);

      // Check for unboxed captures in both partial_applies and direct
      // applications of non-escaping closures.
      else if (auto apply = ApplySite::isa(inst))
        handleApply(apply);
    }
  }
  invalidateAnalysis(F, SILAnalysis::InvalidationKind::Instructions);
}

SourceAccess AccessEnforcementSelection::getAccessKindForBox(SILValue source) {
  if (auto *BBI = dyn_cast<BeginBorrowInst>(source))
    source = BBI->getOperand();
  if (auto *MUI = dyn_cast<MarkUninitializedInst>(source))
    source = MUI->getOperand();

  // A box argument is statically enforced unless some caller passes a box
  // that may have escaped.
  if (auto *arg = dyn_cast<SILFunctionArgument>(source)) {
    if (arg->getType().is<SILBoxType>() && !boxArguments->isDynamic(arg))
      return SourceAccess::getStaticAccess();
  }
  // If we didn't allocate the box, assume that we need to use
  // dynamic enforcement.
  // TODO: use static enforcement in certain provable cases.
  auto box = dyn_cast<AllocBoxInst>(source);
  if (!box)
    return SourceAccess::getDynamicAccess();

  return SourceAccess::getBoxedAccess(box);
}

SourceAccess AccessEnforcementSelection::getSourceAccess(SILValue address) {
  // Recurse through MarkUninitializedInst.
  if (auto *mui = dyn_cast<MarkUninitializedInst>(address))
    return getSourceAccess(mui->getOperand());

  // Recurse through mark must check.
  if (auto *mmci = dyn_cast<MarkUnresolvedNonCopyableValueInst>(address))
    return getSourceAccess(mmci->getOperand());

  // Recur through moveonlywrapper_to_copyable_addr or vice versa.
  if (auto *m = dyn_cast<MoveOnlyWrapperToCopyableAddrInst>(address))
    return getSourceAccess(m->getOperand());
  if (auto *c = dyn_cast<CopyableToMoveOnlyWrapperAddrInst>(address))
    return getSourceAccess(c->getOperand());

  // Recurse through drop_deinit.
  if (auto *ddi = dyn_cast<DropDeinitInst>(address))
    return getSourceAccess(ddi->getOperand());

  // Recurse through moveonlywrapper_to_copyable_box.
  if (auto *m = dyn_cast<MoveOnlyWrapperToCopyableBoxInst>(address))
    return getAccessKindForBox(m->getOperand());

  if (auto box = dyn_cast<ProjectBoxInst>(address))
    return getAccessKindForBox(box->getOperand());

  if (auto arg = dyn_cast<SILFunctionArgument>(address)) {
    switch (arg->getArgumentConvention()) {
    case SILArgumentConvention::Indirect_Inout:
      // `inout` arguments are checked on the caller side, either statically
      // or dynamically if necessary. The @inout does not alias and cannot
      // escape within the callee, so static enforcement is always sufficient.
      return SourceAccess::getStaticAccess();

    case SILArgumentConvention::Indirect_InoutAliasable:
      if (dynamicCaptures->isDynamic(arg))
        return SourceAccess::getDynamicAccess();

      return SourceAccess::getStaticAccess();
    case SILArgumentConvention::Indirect_In:
    case SILArgumentConvention::Indirect_In_Guaranteed:
      // @in/@in_guaranteed cannot be mutably accessed, mutably captured, or
      // passed as inout. @in/@in_guaranteed may be captured @inout_aliasable.
      // (This is fairly horrible, but presumably Sema/SILGen made sure a copy
      // wasn't needed?)
      //  
      // FIXME: When we have borrowed arguments, a "read" needs to be enforced
      // on the caller side.
      return SourceAccess::getStaticAccess();

    case SILArgumentConvention::Indirect_Out:
      // We use an initialized 'out' argument as a parameter.
      return SourceAccess::getStaticAccess();

    default:
      llvm_unreachable("Expecting an inout argument.");
    }
  }
  // If we're not accessing a box or argument, we must've lowered to a stack
  // element. Other sources of access are either outright dynamic (GlobalAddr,
  // RefElementAddr), or only exposed after mandatory inlining (nested
  // dependent BeginAccess).
  //
  // Running before diagnostic constant propagation requires handling 'undef'.
  assert(isa<AllocStackInst>(address) || isa<SILUndef>(address));
  return SourceAccess::getStaticAccess();
}

void AccessEnforcementSelection::handleApply(ApplySite apply) {
  auto calleeTy = apply.getOrigCalleeType();
  SILFunctionConventions calleeConv(
      calleeTy, SILAddressConventions::forFunction(*apply.getFunction()));

  for (Operand &oper : apply.getArgumentOperands()) {
    AddressCapture capture(oper);
    if (!capture.isValid())
      continue;

    // This is a non-escaping closure argument. If the argument requires dynamic
    // access, record that in dynamicCaptures.
    auto sourceAccess = getSourceAccess(oper.get());
    switch (sourceAccess.kind) {
    case SourceAccess::StaticAccess:
      // If the captured variable does not require dynamic enforcement, then
      // there's no need to track it.
      break;
    case SourceAccess::DynamicAccess: {
      dynamicCaptures->recordCapture(capture);
      break;
    }
    case SourceAccess::BoxAccess:
      // Captures of box projections are handled during SelectEnforcement, which
      // determines the access enforcement for all users of a box. Within
      // SelectEnforcement, we know whether the box has escaped before the
      // capture. Here there's just nothing to do.
      assert(handledBoxes.count(sourceAccess.allocBox));
      break;
    }
  }
}

void AccessEnforcementSelection::handleAccess(BeginAccessInst *access) {
  if (access->getEnforcement() != SILAccessEnforcement::Unknown)
    return;

  auto sourceAccess = getSourceAccess(access->getOperand());
  switch (sourceAccess.kind) {
  case SourceAccess::StaticAccess:
    setStaticEnforcement(access);
    break;
  case SourceAccess::DynamicAccess:
    setDynamicEnforcement(access);
    break;
  case SourceAccess::BoxAccess:
    llvm_unreachable("All boxes must have already been selected.");
  }
}

} // end anonymous namespace

SILTransform *swift::createAccessEnforcementSelection() {
  return new AccessEnforcementSelection();
}
