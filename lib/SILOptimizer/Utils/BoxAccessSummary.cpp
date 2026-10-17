//===--- BoxAccessSummary.cpp - Accesses to a box during a call -----------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2026 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "box-access-summary"
#include "swift/SILOptimizer/Utils/BoxAccessSummary.h"
#include "swift/SIL/InstructionUtils.h"
#include "swift/SIL/MemAccessUtils.h"
#include "swift/SIL/SILArgument.h"
#include "swift/SIL/SILInstruction.h"
#include "swift/SILOptimizer/Analysis/AccessSummaryAnalysis.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace swift;

static SILAccessKind strongest(SILAccessKind a, SILAccessKind b) {
  return a == SILAccessKind::Read ? b : a;
}

void BoxAccesses::addMay(SILAccessKind kind) {
  mayAccess = mayAccess ? strongest(*mayAccess, kind) : kind;
}

void BoxAccesses::addMust(SILAccessKind kind, SILLocation loc) {
  addMay(kind);
  if (!mustAccess ||
      (*mustAccess == SILAccessKind::Read && kind != SILAccessKind::Read)) {
    mustAccess = kind;
    mustLoc = loc;
  }
}

void BoxAccesses::mergeMay(const BoxAccesses &other) {
  if (other.mayAccess)
    addMay(*other.mayAccess);
}

void BoxAccesses::mergeMust(const BoxAccesses &other) {
  mergeMay(other);
  if (other.mustAccess)
    addMust(*other.mustAccess, other.mustLoc);
}

BoxAccessSummary::BoxAccessSummary(AccessSummaryAnalysis *ASA) : ASA(ASA) {}

BoxAccessSummary::~BoxAccessSummary() {}

/// Return the callee's argument for \p oper if the callee's body is known.
static SILFunctionArgument *getCalleeArgument(FullApplySite apply,
                                              Operand &oper) {
  // This excludes dynamically replaceable callees.
  SILFunction *callee = apply.getReferencedFunctionOrNull();
  if (!callee || !callee->isDefinition())
    return nullptr;

  return cast<SILFunctionArgument>(
      callee->getArgument(apply.getSubstCalleeArgIndex(oper)));
}

static CanSILFunctionType getClosureType(SILValue value) {
  SILType type = value->getType();
  if (auto objectType = type.getOptionalObjectType())
    type = objectType;
  return type.getAs<SILFunctionType>();
}

/// Return true if a call that never returns ends the path through \p block
/// before \p end, or anywhere in the block if \p end is null. Before
/// NoReturnFolding, such a call need not be followed by unreachable.
static bool hasNoReturnCall(SILBasicBlock *block, SILInstruction *end) {
  for (SILInstruction &inst : *block) {
    if (&inst == end)
      break;
    if (auto apply = FullApplySite::isa(&inst)) {
      if (apply.isCalleeNoReturn())
        return true;
    }
  }
  return false;
}

bool BoxAccessSummary::executesOnAllPaths(SILInstruction *inst) {
  SILBasicBlock *instBlock = inst->getParent();
  if (hasNoReturnCall(instBlock, inst))
    return false;

  // Every path from the entry must reach instBlock. Visit the blocks reachable
  // without passing through instBlock. If any of them exits the function,
  // including by trapping, or they form a cycle, then a path avoids inst.
  enum class State { Visiting, Done };
  llvm::DenseMap<SILBasicBlock *, State> states;
  SmallVector<std::pair<SILBasicBlock *, unsigned>, 16> stack;
  auto visit = [&](SILBasicBlock *block) {
    states[block] = State::Visiting;
    stack.push_back({block, 0});
    return !block->getSuccessors().empty() && !hasNoReturnCall(block, nullptr);
  };
  auto *entry = inst->getFunction()->getEntryBlock();
  if (entry == instBlock)
    return true;

  if (!visit(entry))
    return false;

  while (!stack.empty()) {
    auto &[block, succIdx] = stack.back();
    auto succs = block->getSuccessors();
    if (succIdx == succs.size()) {
      states[block] = State::Done;
      stack.pop_back();
      continue;
    }
    SILBasicBlock *succ = succs[succIdx++].getBB();
    if (succ == instBlock)
      continue;

    auto found = states.find(succ);
    if (found != states.end()) {
      if (found->second == State::Visiting)
        return false;
      continue;
    }
    if (!visit(succ))
      return false;
  }
  return true;
}

BoxAccesses BoxAccessSummary::getBoxArgumentAccesses(SILFunctionArgument *arg) {
  auto found = boxArgumentCache.try_emplace(arg, std::nullopt);
  if (!found.second) {
    if (found.first->second)
      return *found.first->second;

    // A recursive call. Assume the worst for the may accesses, and nothing
    // for the must accesses.
    BoxAccesses recursive;
    recursive.addMay(SILAccessKind::Modify);
    return recursive;
  }
  BoxAccesses result = computeBoxArgumentAccesses(arg);
  boxArgumentCache[arg] = result;
  return result;
}

ClosureInvocation
BoxAccessSummary::getClosureArgumentInvocation(SILFunctionArgument *arg) {
  auto found = closureArgumentCache.try_emplace(arg, std::nullopt);
  if (!found.second) {
    if (found.first->second)
      return *found.first->second;

    return ClosureInvocation{/*mayInvoke*/ true, /*mustInvoke*/ false};
  }
  ClosureInvocation result = computeClosureArgumentInvocation(arg);
  closureArgumentCache[arg] = result;
  return result;
}

BoxAccesses
BoxAccessSummary::computeBoxArgumentAccesses(SILFunctionArgument *arg) {
  BoxAccesses result;

  // Addresses within the box, and whether they are the whole variable.
  SmallVector<std::pair<SILValue, bool>, 8> addresses;
  SmallVector<SILValue, 8> boxes;
  llvm::SmallPtrSet<SILValue, 8> visited;
  auto pushBox = [&](SILValue value) {
    if (visited.insert(value).second)
      boxes.push_back(value);
  };
  pushBox(arg);
  while (!boxes.empty()) {
    SILValue box = boxes.pop_back_val();
    for (Operand *use : box->getUses()) {
      SILInstruction *user = use->getUser();
      switch (user->getKind()) {
      case SILInstructionKind::BeginBorrowInst:
      case SILInstructionKind::CopyValueInst:
      case SILInstructionKind::MoveValueInst:
      case SILInstructionKind::MarkUninitializedInst:
        pushBox(cast<SingleValueInstruction>(user));
        continue;
      case SILInstructionKind::ProjectBoxInst:
        addresses.push_back({cast<ProjectBoxInst>(user), true});
        continue;
      case SILInstructionKind::DestroyValueInst:
      case SILInstructionKind::EndBorrowInst:
      case SILInstructionKind::DebugValueInst:
      case SILInstructionKind::DeallocBoxInst:
        continue;
      default:
        break;
      }
      if (auto apply = FullApplySite::isa(user)) {
        if (!apply.isCalleeOperand(*use)) {
          auto *calleeArg = getCalleeArgument(apply, *use);
          if (calleeArg && calleeArg->getType().is<SILBoxType>()) {
            BoxAccesses calleeAccesses = getBoxArgumentAccesses(calleeArg);
            if (executesOnAllPaths(user))
              result.mergeMust(calleeAccesses);
            else
              result.mergeMay(calleeAccesses);
            continue;
          }
        }
      }
      // The box may be accessed through an unknown use, such as a closure
      // that captures it.
      LLVM_DEBUG(llvm::dbgs() << "  Unknown box use: " << *user);
      result.addMay(SILAccessKind::Modify);
    }
  }
  while (!addresses.empty()) {
    auto [address, isWholeVariable] = addresses.pop_back_val();
    for (Operand *use : address->getUses()) {
      SILInstruction *user = use->getUser();
      switch (user->getKind()) {
      case SILInstructionKind::BeginAccessInst: {
        auto *access = cast<BeginAccessInst>(user);
        if (access->getEnforcement() == SILAccessEnforcement::Unsafe)
          continue;

        SILAccessKind kind = access->getAccessKind();
        bool isWholeAccess =
            isWholeVariable && ASA->findSubPathAccessed(access) ==
                                   access->getModule().getIndexTrieRoot();
        if (isWholeAccess && executesOnAllPaths(access))
          result.addMust(kind, access->getLoc());
        else
          result.addMay(kind);
        continue;
      }
      case SILInstructionKind::MarkUnresolvedNonCopyableValueInst:
        addresses.push_back(
            {cast<SingleValueInstruction>(user), isWholeVariable});
        continue;
      case SILInstructionKind::StructElementAddrInst:
      case SILInstructionKind::TupleElementAddrInst:
        addresses.push_back({cast<SingleValueInstruction>(user), false});
        continue;
      default:
        break;
      }
      // An address passed to a call, such as a non-escaping closure that
      // captures it, may be accessed during the call.
      if (ApplySite::isa(user)) {
        result.addMay(SILAccessKind::Modify);
        continue;
      }
      // Other uses, such as initialization, are not formal accesses.
    }
  }
  return result;
}

ClosureInvocation
BoxAccessSummary::computeClosureArgumentInvocation(SILFunctionArgument *arg) {
  ClosureInvocation result;
  SmallVector<SILValue, 8> worklist;
  llvm::SmallPtrSet<SILValue, 8> visited;
  auto push = [&](SILValue value) {
    if (visited.insert(value).second)
      worklist.push_back(value);
  };
  push(arg);
  while (!worklist.empty()) {
    SILValue value = worklist.pop_back_val();
    for (Operand *use : value->getUses()) {
      SILInstruction *user = use->getUser();
      switch (user->getKind()) {
      case SILInstructionKind::BeginBorrowInst:
      case SILInstructionKind::CopyValueInst:
      case SILInstructionKind::MoveValueInst:
      case SILInstructionKind::ConvertFunctionInst:
      case SILInstructionKind::ConvertEscapeToNoEscapeInst:
        push(cast<SingleValueInstruction>(user));
        continue;
      case SILInstructionKind::MarkDependenceInst:
        if (use->getOperandNumber() == MarkDependenceInst::Dependent) {
          push(cast<SingleValueInstruction>(user));
          continue;
        }
        break;
      case SILInstructionKind::DestroyValueInst:
      case SILInstructionKind::EndBorrowInst:
      case SILInstructionKind::DebugValueInst:
      case SILInstructionKind::DeallocStackInst:
      case SILInstructionKind::DestroyNotEscapedClosureInst:
        continue;
      default:
        break;
      }
      if (auto apply = FullApplySite::isa(user)) {
        if (apply.isCalleeOperand(*use)) {
          result.mayInvoke = true;
          if (executesOnAllPaths(user))
            result.mustInvoke = true;
          continue;
        }
        auto *calleeArg = getCalleeArgument(apply, *use);
        if (calleeArg && getClosureType(calleeArg)) {
          ClosureInvocation calleeInvocation =
              getClosureArgumentInvocation(calleeArg);
          result.mayInvoke |= calleeInvocation.mayInvoke;
          if (calleeInvocation.mustInvoke && executesOnAllPaths(user))
            result.mustInvoke = true;
          continue;
        }
      }
      // An unknown use may invoke the closure.
      result.mayInvoke = true;
    }
  }
  return result;
}

void BoxAccessSummary::addClosureAccesses(SILValue closureValue,
                                          SILValue boxRoot,
                                          ClosureInvocation invocation,
                                          BoxAccesses &result) {
  if (!invocation.mayInvoke)
    return;

  auto fnType = getClosureType(closureValue);
  if (!fnType)
    return;

  // The closure stored in an escaping block can't be found.
  if (!fnType->isNoEscape() &&
      fnType->getRepresentation() == SILFunctionTypeRepresentation::Block) {
    result.addMay(SILAccessKind::Modify);
    return;
  }
  TinyPtrVector<PartialApplyInst *> closures;
  findClosuresForFunctionValue(closureValue, closures);
  for (PartialApplyInst *closure : closures) {
    for (Operand &captured : closure->getArgumentOperands()) {
      SILValue value = captured.get();
      if (!value->getType().is<SILBoxType>() ||
          findReferenceRoot(value) != boxRoot) {
        continue;
      }
      SILFunction *callee = closure->getReferencedFunctionOrNull();
      if (!callee || !callee->isDefinition()) {
        result.addMay(SILAccessKind::Modify);
        continue;
      }
      auto *calleeArg = cast<SILFunctionArgument>(callee->getArgument(
          ApplySite(closure).getSubstCalleeArgIndex(captured)));
      BoxAccesses closureAccesses = getBoxArgumentAccesses(calleeArg);
      if (invocation.mustInvoke)
        result.mergeMust(closureAccesses);
      else
        result.mergeMay(closureAccesses);
    }
  }
}

BoxAccesses BoxAccessSummary::getAccessesDuringCall(FullApplySite apply,
                                                    SILValue boxRoot) {
  BoxAccesses result;
  for (Operand &oper : apply.getArgumentOperands()) {
    SILValue value = oper.get();
    if (value->getType().is<SILBoxType>()) {
      if (findReferenceRoot(value) != boxRoot)
        continue;

      auto *calleeArg = getCalleeArgument(apply, oper);
      if (!calleeArg) {
        result.addMay(SILAccessKind::Modify);
        continue;
      }
      // The callee's must accesses happen during every execution of the call.
      result.mergeMust(getBoxArgumentAccesses(calleeArg));
      continue;
    }
    if (!getClosureType(value))
      continue;

    ClosureInvocation invocation{/*mayInvoke*/ true, /*mustInvoke*/ false};
    if (auto *calleeArg = getCalleeArgument(apply, oper))
      invocation = getClosureArgumentInvocation(calleeArg);

    addClosureAccesses(value, boxRoot, invocation, result);
  }
  // A closure that is the callee is invoked by the call.
  addClosureAccesses(apply.getCallee(), boxRoot,
                     ClosureInvocation{/*mayInvoke*/ true, /*mustInvoke*/ true},
                     result);
  return result;
}
