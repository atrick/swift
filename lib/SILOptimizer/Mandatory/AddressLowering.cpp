//===--- AddressLowering.cpp - Lower SIL address-only types. --------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2021 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
///
/// This pass lowers SILTypes. On completion, the SILType of every SILValue is
/// its SIL storage type. A SIL storage type is always an address type for
/// values that require indirect storage at the LLVM IR level. Consequently,
/// this is a mandatory IRGen preparation pass.
///
/// TODO: Much of the current implementation complexity, including most of the
/// general helper routines, stems from handling calls with multiple return
/// values as tuples. Once those calls are properly represented as instructions
/// with multiple results, then the implementation complexity will fall
/// away. See the code tagged "TODO: Multi-Value".
///
/// TODO: Some of the current complexity stems from SILPhiArgument being use for
/// non-phi arguments. See the code tagged "TODO: Phi".
///
/// TODO: Some of the current complexity stems from handling non-ownership
/// SIL. If we decide to force OSSA up until this pass, then that complexity
/// will fall away. See the code tagged "TODO: OSSA".
///
/// ## State
///
/// A `valueStorageMap` maps each opaque SIL value to its storage information.
///
/// ## Step #1: Map opaque values
///
/// Populate `valueStorageMap` in forward order (RPO), giving each opaque value
/// an ordinal position. Eventually, this ordinal may be used to reason about
/// live intervals to allow more storage reuse with interference checking.
///
/// ## Step #2: Allocate storage
///
/// In reverse order (PO), allocate the parent storage object for each opaque
/// value.
///
/// If the value's use composes a parent object from this value (struct, tuple,
/// enum), and use's storage dominates this value, then mark the value's
/// storage as a projection from the use's storage.
///
/// If the value is a subobject extraction (struct_extract, tuple_extract,
/// open_existential_value, unchecked_enum_data), then mark the value's storage
/// as a projection from the def.
///
/// Opaque value's that are the root of all projection paths now have their
/// `storageAddress` assigned to an `alloc_stack` or an argument. Opaque value's
/// that are projections do not yet have a `storageAddress`.
///
/// After allocating storage for all non-argument values, allocate block
/// argument storage. This is handled by a BlockArgumentStorageOptimizer that
/// checks for interference among the branch operands and reuses storage
/// allocated to other values.
///
/// ## Step #3. Rewrite opaque values
///
/// In forward order (RPO), rewrite each opaque value definition, and all its
/// uses. This generally involves creating a new `_addr` variant of the
/// instruction and obtaining the storage address from the `valueStorageMap`.
///
/// If this value's storage is a projection of the value defined by its
/// composing use, then first generate instructions to materialize the
/// projection. This is a recursive process starting with the root of the
/// projection path.
///
/// A projection path will be materialized once for the leaf subobject. When
/// this happens, the `storageAddress` will be assigned for any intermediate
/// projection paths. When those values are rewritten, their `storageAddress`
/// will already be available.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "address-lowering"
#include "BlockArgumentStorage.h"
#include "swift/Basic/BlotSetVector.h"
#include "swift/Basic/Range.h"
#include "swift/SIL/BasicBlockUtils.h"
#include "swift/SIL/DebugUtils.h"
#include "swift/SIL/PrettyStackTrace.h"
#include "swift/SIL/SILArgument.h"
#include "swift/SIL/SILBuilder.h"
#include "swift/SIL/SILVisitor.h"
#include "swift/SILOptimizer/Analysis/PostOrderAnalysis.h"
#include "swift/SILOptimizer/PassManager/Transforms.h"
#include "swift/SILOptimizer/Utils/InstOptUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"

using namespace swift;
using llvm::SmallSetVector;

llvm::cl::opt<bool>
    OptimizeOpaqueAddressLowering("optimize-opaque-address-lowering",
                                  llvm::cl::init(true));

//===----------------------------------------------------------------------===//
// General SIL helpers.
//===----------------------------------------------------------------------===//

namespace {
struct GetUser {
  SILInstruction *operator()(Operand *oper) const { return oper->getUser(); }
};
} // namespace

// TODO: Fix LLVM's map_range to take the function return type explicitly, then
// remove this helper.
static iterator_range<llvm::mapped_iterator<ValueBase::use_iterator, GetUser>>
makeUserRange(SILValue val) {
  return make_range(llvm::map_iterator(val->use_begin(), GetUser()),
                    llvm::map_iterator(val->use_end(), GetUser()));
}

/// Visit all "actual" call results.
/// Stop when the visitor returns `false`.
///
/// TODO: Multi-Value. Remove this helper.
static void visitCallResults(ApplyInst *applyInst,
                             llvm::function_ref<bool(SILValue)> visitor) {
  if (applyInst->getType().is<TupleType>()) {
    for (auto *operand : applyInst->getUses()) {
      if (auto extract = dyn_cast<TupleExtractInst>(operand->getUser()))
        if (!visitor(extract))
          break;
    }
  } else
    visitor(applyInst);
}

/// Get the argument of a try_apply.
///
/// TODO: Multi-Value: Only relevant for tuple pseudo results.
static SILPhiArgument *getTryApplyPseudoResult(TryApplyInst *TAI) {
  auto *argBB = TAI->getNormalBB();
  assert(argBB->getNumArguments() == 1);
  return argBB->getPhiArguments()[0];
}

/// Get the SIL value that represents all of the given call's results. It may be
/// the call itself, a "fake" tuple without storage, or a block argument.
///
/// TODO: Multi-Value: Only relevant for tuple pseudo results.
SILValue getCallPseudoResult(ApplySite apply) {
  return isa<ApplyInst>(apply)
             ? SILValue(cast<SingleValueInstruction>(apply.getInstruction()))
             : SILValue(getTryApplyPseudoResult(cast<TryApplyInst>(apply)));
}

/// Return true if the given value is either a "fake" tuple that represents all
/// of a call's results or an empty tuple of no results. This may return true
/// for either tuple_inst or a block argument.
///
/// TODO: Multi-Value. Calls are SILValues, but when the result type is a tuple,
/// the call value does not represent a real value with storage. This is a bad
/// situation for address lowering because there's no way to tell from any given
/// value whether its legal to assign storage to that value. As a result, the
/// implementation of call lowering doesn't fall out naturally from the
/// algorithm that lowers values to storage.
static bool isPseudoCallResult(SILValue value) {
  if (isa<ApplyInst>(value))
    return value->getType().is<TupleType>();

  auto *bbArg = dyn_cast<SILPhiArgument>(value);
  if (!bbArg)
    return false;

  auto *predBB = bbArg->getParent()->getSinglePredecessorBlock();
  if (!predBB)
    return false;

  return isa<TryApplyInst>(predBB->getTerminator())
         && bbArg->getType().is<TupleType>();
}

/// Return true if this is a pseudo-return value.
static bool isPseudoReturnValue(SILValue value) {
  auto *TI = dyn_cast<TupleInst>(value);
  if (!TI)
    return false;

  return TI->hasOneUse() && isa<ReturnInst>(TI->use_begin()->getUser());
}

/// Return the value associated with the user of an address-only or indirectly
/// returned tuple element.
///
/// The given operand's user must be a TupleInst. For normal tuples, the value
/// is the tuple itself. For returned tuples, the value is the indirect function
/// argument. This requires indirect function arguments to be rewritten first
/// (see insertIndirectReturnArgs).
/// 
/// TODO: Multi-Value; this should all go away.
static SILValue getTupleElementUserValue(Operand *oper) {
  auto *TI = cast<TupleInst>(oper->getUser());
  if (!TI->hasOneUse() || !isa<ReturnInst>(TI->use_begin()->getUser()))
    return TI;

  unsigned resultIdx = TI->getElementIndex(oper);

  SILFunction *F = TI->getFunction();
  SILFunctionConventions loweredFnConv(
      F->getLoweredFunctionType(),
      SILModuleConventions::getLoweredAddressConventions());
  assert(loweredFnConv.getResults().size() == TI->getElements().size());
  unsigned indirectResultIdx = 0;
  for (SILResultInfo result : loweredFnConv.getResults().slice(0, resultIdx)) {
    if (loweredFnConv.isSILIndirect(result))
      ++indirectResultIdx;
  }
  // Cannot call F->getIndirectSILResults here because that API uses the
  // function conventions before address lowering.
  return F->getArguments()[indirectResultIdx];
}

// Check if this is a copy->store pair and return the StoreInst.
//
// We could check that no instructions write to memory (or destroy the copy
// source) in between--but the goal here is not to discover copy coalescing
// opportunities. I expect a copy-propagation prepass to do that for us. That
// prepass should have already identified the fused copy->operand pairs and
// inserted the copy just before the store, struct, enum, tuple, etc.
// The non-store cases are handled naturally as projections. Store is special
// because its memory location can be written by any instruction.
//
// TODO: OSSA. This would be more robust with a 'store [copy]' instead of the
// 'copy'/'store' pair.
static bool isStoreCopy(SILValue value) {
  auto *copyInst = dyn_cast<CopyValueInst>(value);
  if (!copyInst)
    return false;

  if (!copyInst->hasOneUse())
    return false;

  // Handle stores and aggregate composition.
  auto *storeInst = dyn_cast<StoreInst>(value->getSingleUse()->getUser());
  if (!storeInst)
    return false;

  auto storePos = SILBasicBlock::iterator(storeInst);
  return storeInst->getSrc() == copyInst && &*std::prev(storePos) == copyInst;
}

//===----------------------------------------------------------------------===//
// ValueStorageMap: Map Opaque/Resilient SILValues to abstract storage units.
//===----------------------------------------------------------------------===//

ValueStorage &ValueStorageMap::insertValue(SILValue value) {
  auto hashResult =
      valueHashMap.insert(std::make_pair(value, valueVector.size()));
  (void)hashResult;
  assert(hashResult.second && "SILValue already mapped");

  valueVector.emplace_back(value, ValueStorage());

  return valueVector.back().storage;
}

void ValueStorageMap::replaceValue(SILValue oldValue, SILValue newValue) {
  // Currently, replacement is only allowed for values that originally have no
  // defining instruction (block args), that avoids the problem in which other
  // storage has a use projection from this storage identified by an operand
  // number. In the case of block arguments, the corresponding terminating
  // instruction can only be rewritten if it is impossible to project storage
  // to any of its operands. (e.g. try_apply and switch_enum can be
  // rewrittern, branch and cond_branch cannot).
  assert(!oldValue->getDefiningInstruction());

  auto pos = valueHashMap.find(oldValue);
  assert(pos != valueHashMap.end());
  unsigned ordinal = pos->second;
  valueHashMap.erase(pos);

  auto hashResult = valueHashMap.insert(std::make_pair(newValue, ordinal));
  (void)hashResult;
  assert(hashResult.second && "SILValue already mapped");

  valueVector[ordinal].value = newValue;
}

void ValueStorageMap::dump() {
  llvm::dbgs() << "ValueStorageMap:\n";
  for (unsigned ordinal : indices(valueVector)) {
    auto &valStoragePair = valueVector[ordinal];
    llvm::dbgs() << "value: ";
    valStoragePair.value->dump();
    auto &storage = valStoragePair.storage;
    if (storage.isUseProjection) {
      llvm::dbgs() << "  use projection: ";
      if (!storage.isRewritten)
        valueVector[storage.projectedStorageID].value->dump();
    } else if (storage.isDefProjection) {
      llvm::dbgs() << "  def projection: ";
      if (!storage.isRewritten)
        valueVector[storage.projectedStorageID].value->dump();
    }
    if (storage.storageAddress) {
      llvm::dbgs() << "  storage: ";
      storage.storageAddress->dump();
    }
  }
}

//===----------------------------------------------------------------------===//
// AddressLoweringState: shared state for the pass's analysis and transforms.
//===----------------------------------------------------------------------===//

namespace {
struct AddressLoweringState {
  SILFunction *F;
  SILFunctionConventions loweredFnConv;

  // Dominators remain valid throughout this pass.
  DominanceInfo *domInfo;

  // All opaque values and associated storage.
  ValueStorageMap valueStorageMap;
  // All call sites with formally indirect SILArgument or SILResult conventions.
  SmallBlotSetVector<ApplySite, 16> indirectApplies;
  // All function-exiting terminators (return or throw instructions).
  SmallVector<SILInstruction *, 8> exitingInsts;
  // Delete these instructions after performing transformations. They must
  // already be dead (no remaining users) before being added to this set.
  SmallSetVector<SILInstruction *, 16> instsToDelete;

#ifndef NDEBUG
  // Calls are removed after everything else. This set contains all original
  // calls with multiple results, where at least one result is indirect.
  //
  // TODO: Multi-Value. Until calls are deleted like all other values, they
  // need to be carefully orchestrated w.r.t. their tuple values.
  SmallSetVector<ApplyInst *, 16> callsToDelete;
#endif

  AddressLoweringState(SILFunction *F, DominanceInfo *domInfo)
      : F(F),
        loweredFnConv(F->getLoweredFunctionType(),
                      SILModuleConventions::getLoweredAddressConventions(F->getModule())),
        domInfo(domInfo) {}

  bool isDead(SILInstruction *inst) const { return instsToDelete.count(inst); }

  void markDead(SILInstruction *inst) {
#ifndef NDEBUG
    for (auto result : inst->getResults())
      for (Operand *use : result->getUses())
        assert(instsToDelete.count(use->getUser()));
#endif
    instsToDelete.insert(inst);
  }

  SILBuilder getBuilder(SILBasicBlock::iterator insertPt,
                        SILInstruction *originalInst = nullptr) {
    if (!originalInst)
      originalInst = &*insertPt;
    SILBuilder B(insertPt);
    B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
    B.setOpenedArchetypesTracker(&openedArchetypesTracker);
    B.setCurrentDebugScope(originalInst->getDebugScope());
    return B;
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// OpaqueValueVisitor: Map OpaqueValues to ValueStorage.
//===----------------------------------------------------------------------===//

/// Before populating the ValueStorageMap, replace each value-typed argument to
/// the current function with an address-typed argument by inserting a temporary
/// load instruction.
static void convertIndirectFunctionArgs(AddressLoweringState &pass) {
  // Insert temporary argument loads at the top of the function.
  SILBuilder argBuilder = pass.getBuilder(pass.F->getEntryBlock()->begin());

  auto fnConv = pass.F->getConventions();
  unsigned argIdx = fnConv.getSILArgIndexOfFirstParam();
  for (SILParameterInfo param :
       pass.F->getLoweredFunctionType()->getParameters()) {

    if (param.isFormalIndirect() && !fnConv.isSILIndirect(param)) {
      SILArgument *arg = pass.F->getArgument(argIdx);
      SILType addrType = arg->getType().getAddressType();
      LoadInst *loadArg = argBuilder.createLoad(
          SILValue(arg).getLoc(), SILUndef::get(addrType, *pass.F),
          LoadOwnershipQualifier::Unqualified);

      arg->replaceAllUsesWith(loadArg);
      assert(!pass.valueStorageMap.contains(arg));

      arg = arg->getParent()->replaceFunctionArgument(
          arg->getIndex(), addrType, ValueOwnershipKind::Any, arg->getDecl());

      loadArg->setOperand(arg);

      // Indirect calling convention may be used for loadable types. In that
      // case, generating the argument loads is sufficient.
      if (addrType.isAddressOnly(*pass.F)) {
        auto &storage = pass.valueStorageMap.insertValue(loadArg);
        storage.storageAddress = arg;
        storage.isRewritten = true;
      }
    }
    ++argIdx;
  }
  assert(argIdx
         == fnConv.getSILArgIndexOfFirstParam() + fnConv.getNumSILArguments());
}

/// Before populating the ValueStorageMap, insert function arguments for any
/// @out result type. Return the number of indirect result arguments added.
static unsigned insertIndirectReturnArgs(AddressLoweringState &pass) {
  auto &ctx = pass.F->getModule().getASTContext();
  unsigned argIdx = 0;
  for (auto resultTy : pass.loweredFnConv.getIndirectSILResultTypes()) {
    auto bodyResultTy = pass.F->mapTypeIntoContext(resultTy);
    auto var = new (ctx)
        ParamDecl(ParamDecl::Specifier::InOut, SourceLoc(), SourceLoc(),
                  ctx.getIdentifier("$return_value"), SourceLoc(),
                  ctx.getIdentifier("$return_value"), pass.F->getDeclContext());

    SILFunctionArgument *funcArg = pass.F->begin()->insertFunctionArgument(
        argIdx, bodyResultTy.getAddressType(), ValueOwnershipKind::Any, var);
    // Insert function results into valueStorageMap so that the caller storage
    // can be projected onto values inside the function as use projections.
    auto &storage = pass.valueStorageMap.insertValue(funcArg);
    // This is the only case where a value defines its own storage.
    storage.storageAddress = funcArg;
    storage.isRewritten = true;
      
    ++argIdx;
  }
  assert(argIdx == pass.loweredFnConv.getNumIndirectSILResults());
  return argIdx;
}

namespace {
/// Collect all opaque/resilient values, inserting them in `valueStorageMap` in
/// RPO order.
///
/// Collect all call arguments with formally indirect SIL argument convention in
/// `indirectOperands` and formally indirect SIL results in `indirectResults`.
///
/// TODO: Perform linear-scan style in-place stack slot coloring by keeping
/// track of each value's last use.
class OpaqueValueVisitor {
  AddressLoweringState &pass;
  PostOrderFunctionInfo postorderInfo;

public:
  explicit OpaqueValueVisitor(AddressLoweringState &pass)
      : pass(pass), postorderInfo(pass.F) {}

  void mapValueStorage();

protected:
  void checkForIndirectApply(ApplySite applySite);
  void visitValue(SILValue value);
};
} // end anonymous namespace

/// Top-level entry. Populates AddressLoweringState's `valueStorageMap`,
/// `indirectApplies`, and `exitingInsts`.
///
/// Find all Opaque/Resilient SILValues and add them
/// to valueStorageMap in RPO.
void OpaqueValueVisitor::mapValueStorage() {
  for (auto *BB : postorderInfo.getReversePostOrder()) {
    if (BB->getTerminator()->isFunctionExiting())
      pass.exitingInsts.push_back(BB->getTerminator());

    // Opaque function arguments have already been replaced.
    if (BB != pass.F->getEntryBlock()) {
      for (auto *arg : BB->getArguments()) {
        if (isPseudoCallResult(arg))
          continue;

        visitValue(arg);
      }
    }
    for (auto &II : *BB) {
      pass.openedArchetypesTracker.registerOpenedArchetypes(&II);

      if (auto apply = ApplySite::isa(&II))
        checkForIndirectApply(apply);

      for (auto result : II.getResults()) {
        if (isPseudoCallResult(result) || isPseudoReturnValue(result))
          continue;

        visitValue(result);
      }
    }
  }
}

/// Populate `indirectApplies`.
void OpaqueValueVisitor::checkForIndirectApply(ApplySite applySite) {
  auto calleeConv = applySite.getSubstCalleeConv();
  unsigned calleeArgIdx = applySite.getCalleeArgIndexOfFirstAppliedArg();
  for (Operand &operand : applySite.getArgumentOperands()) {
    if (operand.get()->getType().isObject()) {
      auto argConv = calleeConv.getSILArgumentConvention(calleeArgIdx);
      if (argConv.isIndirectConvention()) {
        pass.indirectApplies.insert(applySite);
        break;
      }
    }
    ++calleeArgIdx;
  }
  if (applySite.getSubstCalleeType()->hasIndirectFormalResults())
    pass.indirectApplies.insert(applySite);
}

/// If `value` is address-only, add it to the `valueStorageMap`.
void OpaqueValueVisitor::visitValue(SILValue value) {
  if (value->getType().isObject()
      && value->getType().isAddressOnly(*pass.F)) {
    if (pass.valueStorageMap.contains(value)) {
      assert(!ApplySite::isa(value) && "!!!how did an apply get here???");
      assert(ApplySite::isa(value)
             || isa<SILFunctionArgument>(
                    pass.valueStorageMap.getStorage(value).storageAddress));
      return;
    }
    pass.valueStorageMap.insertValue(value);
  }
}

/// Top-level entry point.
///
/// Prepare the SIL by rewriting function arguments and returns.
/// Initialize the ValueStorageMap with an entry for each opaque value in the
/// function.
static void prepareValueStorage(AddressLoweringState &pass) {
  // Fixup this function's argument types with temporary loads.
  convertIndirectFunctionArgs(pass);

  // Create a new function argument for each indirect result.
  insertIndirectReturnArgs(pass);

  // Populate valueStorageMap.
  OpaqueValueVisitor(pass).mapValueStorage();
}

//===----------------------------------------------------------------------===//
// Storage allocation "oracles".
//
// These queries determine whether storage for a SILValue can be projected from
// its operands or uses. The answer must be consistent across both
// OpaqueStorageAllocation and AddressMaterialization.
// ===---------------------------------------------------------------------===//

/// Def-projection oracle: Return the operand whose source is the aggregate
/// value that is extracted into the given subobject value, or nullptr.
///
/// Invariant:
///   `getProjectedDefOperand(value) != nullptr`
/// if-and-only-if
///   `pass.valueStorageMap.getStorage(value).isDefProjection`
static Operand *getProjectedDefOperand(SILValue value) {
  switch (value->getKind()) {
  default:
    return nullptr;

  case ValueKind::TupleExtractInst: {

    auto *TEI = cast<TupleExtractInst>(value);
    // TODO: Multi-Value: TupleExtract from an apply are handled specially until
    // we have multi-result calls. Force them to allocate storage.
    if (ApplySite::isa(TEI->getOperand()))
      return nullptr;

    LLVM_FALLTHROUGH;
  }
  case ValueKind::StructExtractInst:
  case ValueKind::OpenExistentialValueInst:
  case ValueKind::OpenExistentialBoxValueInst:
    assert(value.getOwnershipKind() == ValueOwnershipKind::Guaranteed);
    return &cast<SingleValueInstruction>(value)->getAllOperands()[0];

  // TODO: Phi: handle terminator results. This has nothing to do with phis.
  case ValueKind::SILPhiArgument: {
    auto *bbArg = cast<SILPhiArgument>(value);
    auto *predBB = bbArg->getParent()->getSinglePredecessorBlock();
    if (!predBB)
      return nullptr;
    auto *SEI = dyn_cast<SwitchEnumInst>(predBB->getTerminator());
    if (!SEI)
      return nullptr;
    return &SEI->getAllOperands()[0];
  }
  case ValueKind::UncheckedEnumDataInst: {
    // FIXME: !!!
    llvm_unreachable("unchecked_enum_data unimplemented");
  }
  }
}

/// Return the operand whose source is borrowed (not consumed) by this value.
static Operand *getBorrowedDefOperand(SILValue value) {
  // A store copy borrows is source just long enough to store it without
  // its source being destroyed in between.
  if (isStoreCopy(value))
    return &cast<CopyValueInst>(value)->getOperandRef();

  return nullptr;
}

/// Return true of the given instruction either composes an aggregate from its
/// operands or forwards its operands to arguments.
///
/// Use projection oracle (predicate).
///
/// Invariant: For all operands, def -> use
/// If
///   `pass.valueStorageMap.getStorage(def).isUseProjection`
/// Then
///   `canProjectFromUse(use)`
static bool canProjectFromUse(SILInstruction *composedUser) {
  switch (composedUser->getKind()) {
  default:
    return false;

  // @in operands never need their own storage since they are non-mutating
  // uses. They always reuse the storage allocated for their operand. So it
  // wouldn't make sense to "project" out of the apply argument.
  case SILInstructionKind::ApplyInst:
  case SILInstructionKind::TryApplyInst:
    return false;

  // Return instructions can project from the caller's storage to the returned
  // value.
  case SILInstructionKind::ReturnInst:
    return true;

  // structs an enums are straightforward compositions.
  case SILInstructionKind::StructInst:
  case SILInstructionKind::EnumInst:
    return true;

  // A tuple is either a composition or forwards its element through a return
  // through function argument storage. Either way, its element can be a
  // use projection.
  case SILInstructionKind::TupleInst:
    return true;

  // init_existential_value composes an existential value, but may depends on
  // opened archetypes. The caller will need to check that storage dominates
  // the opened types.
  case SILInstructionKind::InitExistentialValueInst:
    return true;

  // Terminators that supply block arguments are handled separately by
  // BlockArgumentStorageOptimizer.
  case SILInstructionKind::BranchInst:
  case SILInstructionKind::CondBranchInst:
    return false;
  }
  // TODO: SwitchValueInst, CheckedCastValueBranchInst.
}

/// Return the SILValue that may be associated with storage for the given
/// operand's non-branch user.
///
/// Def/use projection oracle (operand/value association).
///
/// If no SILValues is returned, then there can be no storage projection
/// from either the operand's source to its use (def projection), or from its
/// use to its source (use projection).
///
/// Returning a valid SILValue does not indicate the existence of a
/// projection; only that *if* there is use or def projection, then that can
/// be determined by checking the returned value's storage.
///
/// Branches are handled seperately because ValueStorage does not record their
/// operand index, and branch projections aren't realized until all other SSA
/// values are rewritten.
///
/// FIXME: We should handle BlockArguments that project their payload here like
/// normal operands and only treat BlockArguments separately when they originate
/// from BranchInst or CondBranchInst.
static SILValue getStorageValueForNonBranchUse(Operand *operand) {
  SILInstruction *user = operand->getUser();

  // Tuples are special because they may represent returned values.
  // TODO: Multi-Value; this check should go away.
  if (auto *TI = dyn_cast<TupleInst>(user))
    return getTupleElementUserValue(operand);

  // The caller must provide argument storage.
  if (ApplySite::isa(user))
    return SILValue();

  if (auto *singleVal = dyn_cast<SingleValueInstruction>(user))
    return singleVal;

  switch (user->getKind()) {
  default:
    return SILValue();

  case SILInstructionKind::ReturnInst: {
    auto *RI = cast<ReturnInst>(user);

    SILFunctionConventions loweredFnConv(
        RI->getFunction()->getLoweredFunctionType(),
        SILModuleConventions::getLoweredAddressConventions());
    assert(loweredFnConv.getNumIndirectSILResults() == 1);
    (void)loweredFnConv;

    // Cannot call getIndirectSILResults here because that API uses the
    // function conventions before address lowering.
    return RI->getFunction()->getArguments()[0];
  }
  case SILInstructionKind::TryApplyInst:
  case SILInstructionKind::BranchInst:
  case SILInstructionKind::CondBranchInst:
  case SILInstructionKind::SwitchEnumInst:
    // Projections through branches are intentionally ignored here.
    return SILValue();
  }
}

void ValueStorageMap::setComposingUseProjection(Operand *oper) {
  auto &storage = getStorage(oper->get());
  SILValue useVal = getStorageValueForNonBranchUse(oper);
  assert(useVal && "expected an operand of a composed aggregate");
  assert(!storage.isAllocated());
  storage.projectedStorageID = getOrdinal(useVal);
  storage.projectedOperandNum = oper->getOperandNumber();
  storage.isUseProjection = true;
}

void ValueStorageMap::setBranchUseProjection(Operand *oper,
                                             SILPhiArgument *bbArg) {
  assert(isa<TermInst>(oper->getUser()));
  auto &storage = getStorage(oper->get());
  assert(!storage.isAllocated());
  assert(storage.projectedOperandNum == ValueStorage::InvalidOper);
  storage.projectedStorageID = getOrdinal(bbArg);
  storage.isUseProjection = true;
}

bool ValueStorageMap::isNonBranchUseProjection(Operand *oper) const {
  auto hashPos = valueHashMap.find(oper->get());
  if (hashPos == valueHashMap.end())
    return false;

  auto &srcStorage = valueVector[hashPos->second].storage;
  if (!srcStorage.isUseProjection)
    return false;

  return srcStorage.projectedOperandNum == oper->getOperandNumber();
}

//===----------------------------------------------------------------------===//
// OpaqueStorageAllocation: Generate alloc_stack and address projections for all
// abstract storage locations.
//===----------------------------------------------------------------------===//

namespace {
/// Allocate storage on the stack for every opaque value defined in this
/// function in RPO order. If the definition is an argument of this function,
/// simply replace the function argument with an address representing the
/// caller's storage.
///
/// TODO: shrink lifetimes by inserting alloc_stack at the dominance LCA and
/// finding the lifetime boundary with a simple backward walk from uses.
class OpaqueStorageAllocation {
  AddressLoweringState &pass;

public:
  explicit OpaqueStorageAllocation(AddressLoweringState &pass) : pass(pass) {}

  void allocateOpaqueStorage();

protected:
  void allocateForValue(SILValue value);
  bool findProjectionFromUser(SILValue value,
                              ArrayRef<SILValue> incomingValues);

  bool findProjectionFromUser(SILValue value) {
    return findProjectionFromUser(value, ArrayRef<SILValue>(value));
  }

  bool checkStorageDominates(AllocStackInst *allocInst,
                             ArrayRef<SILValue> incomingValues);

  void allocateForBBArg(SILPhiArgument *bbArg);

  void deallocateValue(SILValue value);

  AllocStackInst *createStackAllocation(SILValue value);

  void createStackAllocationStorage(SILValue value) {
    pass.valueStorageMap.getStorage(value).storageAddress =
        createStackAllocation(cast<SingleValueInstruction>(value));
  }
};
} // end anonymous namespace

/// Top-level entry point: allocate storage for all opaque/resilient values.
void OpaqueStorageAllocation::allocateOpaqueStorage() {
  // Create an AllocStack for every opaque value defined in the function.  Visit
  // values in post-order to create storage for aggregates before subobjects.
  for (auto &valueStorageI : reversed(pass.valueStorageMap)) {
    SILValue value = valueStorageI.value;
    if (!isa<SILArgument>(value))
      allocateForValue(value);
  }
  // Only allocate block arguments after all SSA values have been
  // allocated. allocatedForValue assumes SSA form without checking
  // interference. At that point, multiple SILValues can share storage via
  // projections, but the storage is still singly defined. However,
  // allocateForBB may coalesce multiple values feeding a block argument, or
  // even a single value across multiple loop iterations. The burden for
  // checking inteference is entirely on allocateForBBArg.
  for (auto &valueStorageI : reversed(pass.valueStorageMap)) {
    SILValue value = valueStorageI.value;
    auto *bbArg = dyn_cast<SILPhiArgument>(value);
    if (!bbArg) {
      assert(valueStorageI.storage.isAllocated());
      continue;
    }
    allocateForBBArg(bbArg);
  }
}

/// Allocate storage for a single opaque/resilient value.
void OpaqueStorageAllocation::allocateForValue(SILValue value) {
  // Function arguments are preallocated. Block arguments must be deferred.
  assert(!isa<SILArgument>(value));

  // Pseudo call results have no storage.
  assert(!isPseudoCallResult(value));

  // Pseudo return values have no storage.
  assert(!isPseudoReturnValue(value));

  auto &storage = pass.valueStorageMap.getStorage(value);

  // Fake loads for incoming function arguments are already rewritten; so are
  // outgoing function arguments.
  if (storage.isRewritten)
    return;

  assert(!storage.isAllocated());

  // Check for values that inherently project storage from their operand.
  if (auto *storageOper = getProjectedDefOperand(value)) {
    pass.valueStorageMap.setExtractedDefOperand(storageOper);
    return;
  }
  // Check for values that borrow their operand. e.g. This may be a copy that is
  // immediately consumed by a store.
  if (auto *storageOper = getBorrowedDefOperand(value)) {
    pass.valueStorageMap.setBorrowedDefOperand(storageOper);
    return;
  }
  // Attempt to reuse a user's storage.
  if (findProjectionFromUser(value))
    return;

  // Eagerly create stack allocation. This way any operands can check
  // alloc_stack dominance before their storage is coalesced with this
  // value. Unfortunately, this alloc_stack may be dead if we later coalesce
  // this value's storage with a branch use.
  createStackAllocationStorage(value);
}

/// Find a use of this value that can provide storage for this value.
/// \param value is the value the needs storage.
/// \param incomingValues is a Range of SILValues (e.g. ArrayRef<SILValue>),
/// that all need the storage to be available in their scope.
bool OpaqueStorageAllocation::
findProjectionFromUser(SILValue value, ArrayRef<SILValue> incomingValues) {
  // Def-projections take precedence.
  assert(!getProjectedDefOperand(value));

  for (Operand *use : value->getUses()) {
    if (!canProjectFromUse(use->getUser()))
      continue;

    // Get the user's value, whose storage we will project from.
    SILValue userValue = getStorageValueForNonBranchUse(use);

    // Recurse through all storage projections to find the uniquely allocated
    // storage.
    auto &storage = pass.valueStorageMap.getNonProjectionStorage(userValue);

    SILValue addr = storage.storageAddress;
    if (auto *stackInst = dyn_cast<AllocStackInst>(addr)) {
      if (!checkStorageDominates(stackInst, incomingValues))
        continue;
    } else
      assert(isa<SILFunctionArgument>(addr));

    LLVM_DEBUG(llvm::dbgs() << "  PROJECT "; use->getUser()->dump();
               llvm::dbgs() << "  into "; value->dump());

    pass.valueStorageMap.setComposingUseProjection(use);
    return true;
  }
  return false;
}

bool OpaqueStorageAllocation::
checkStorageDominates(AllocStackInst *allocInst,
                      ArrayRef<SILValue> incomingValues) {

  for (SILValue incomingValue : incomingValues) {
    if (auto *defInst = incomingValue->getDefiningInstruction()) {
      if (!pass.domInfo->properlyDominates(allocInst, defInst))
        return false;
      continue;
    }
    auto *bbArg = cast<SILPhiArgument>(incomingValue);
    // For block arguments, the storage def's block must structly dominate
    // the argument's block.
    if (!pass.domInfo->properlyDominates(incomingValue,
                                         &*bbArg->getParent()->begin())) {
      return false;
    }
  }
  return true;
}

// Allocate storage for a BB arg. Unlike normal values, this checks all the
// incoming values to determine whether any are also candidates for
// projection.
//
// TODO: consider coalescing storage from function arguments with block args.
void OpaqueStorageAllocation::allocateForBBArg(SILPhiArgument *bbArg) {
  if (auto *predBB = bbArg->getParent()->getSinglePredecessorBlock()) {
    if (auto *SEI = dyn_cast<SwitchEnumInst>(predBB->getTerminator())) {
      pass.valueStorageMap.setExtractedBlockArg(SEI->getOperand(), bbArg);
      return;
    }
    // try_apply is handled differently. If it returns a tuple, then the bbarg
    // has no storage. If it returns a single value then the bb arg has
    // storage, but that storage isn't projected onto any incoming value.
    if (isa<TryApplyInst>(predBB->getTerminator())) {
      // TODO: Multi-Value calls.
      if (!bbArg->getType().is<TupleType>()) {
        if (!findProjectionFromUser(bbArg)) {
          createStackAllocationStorage(bbArg);
        }
      }
      return;
    }
  }
  // BlockArgumentStorageOptimizer computes the incoming values of a basic
  // block argument that can share storage with the block argument. The
  // algorithm processes all incoming values at once, so it is is run when
  // visiting the block argument.
  //
  // The incoming value projections are computed first to give them
  // priority. Then we determine if the block argument itself can share
  // storage with one of its users, given that it may already have projections
  // to incoming values.
  //
  // The single incoming value case (including try_apply results) will be
  // immediately pruned by computeArgumentProjections--it will always be a
  // projection of its block argument.
  auto argStorageResult =
      BlockArgumentStorageOptimizer(pass.valueStorageMap, bbArg)
          .computeArgumentProjections();

  SmallVector<SILValue, 4> incomingValues;
  auto incomingValRange = argStorageResult.getIncomingValueRange();
  incomingValues.resize(argStorageResult.getArgumentProjections().size());
  for (SILValue val : incomingValRange)
    incomingValues.push_back(val);
  
  if (!findProjectionFromUser(bbArg, incomingValues))
    createStackAllocationStorage(bbArg);

  // Regardless of whether we projected from a user or allocated storage,
  // provide this storage to all the incoming values that can reuse it.
  for (Operand *argOper : argStorageResult.getArgumentProjections()) {
    deallocateValue(argOper->get());
    pass.valueStorageMap.setBranchUseProjection(argOper, bbArg);
  }
}

// Unfortunately, we create alloc_stack instructions for SSA values before
// coalescing block arguments. This temporary storage now needs to be removed.
void OpaqueStorageAllocation::deallocateValue(SILValue value) {
  auto &storage = pass.valueStorageMap.getStorage(value);
  auto *allocInst = cast<AllocStackInst>(storage.storageAddress);
  storage.storageAddress = nullptr;

  // It's only use should be dealloc_stacks.
  for (Operand *use : allocInst->getUses())
    cast<DeallocStackInst>(use->getUser())->eraseFromParent();

  allocInst->eraseFromParent();
}

// Create alloc_stack that dominates `value` and jointly-postdominating
// dealloc_stack instructions.  Nesting will be fixed later.
//
// This may split critical edges. If so, it updates pass.domInfo.
AllocStackInst *OpaqueStorageAllocation::
createStackAllocation(SILValue value) {
  SILType allocTy = value->getType();
  auto *defInst = value->getDefiningInstruction();
  auto createAllocStack = [&](SILInstruction *allocPoint,
                              ArrayRef<SILInstruction *> deallocPoints) {
    SILBuilderWithScope allocBuilder(allocPoint);
    allocBuilder.setSILConventions(
        SILModuleConventions::getLoweredAddressConventions());
    allocBuilder.setOpenedArchetypesTracker(&pass.openedArchetypesTracker);
    // FIXME: debug scope of block arguments?
    if (defInst)
      allocBuilder.setCurrentDebugScope(defInst->getDebugScope());

    AllocStackInst *allocInstr =
        allocBuilder.createAllocStack(allocPoint->getLoc(), allocTy);

    for (SILInstruction *deallocPoint : deallocPoints) {
      SILBuilderWithScope B(deallocPoint);
      if (defInst)
        B.setCurrentDebugScope(defInst->getDebugScope());
      B.createDeallocStack(deallocPoint->getLoc(), allocInstr);
    }
    return allocInstr;
  };

  if (allocTy.isOpenedExistential()) {
    // Opened archetypes should always be produced by an Instruction.
    assert(defInst);
    // OpenExistential-ish instructions have guaranteed lifetime--they project
    // their operand's storage--so should never reach here.
    assert(!getOpenedArchetypeOf(defInst));
    // For all other instructions, just allocate storage immediately before
    // the value is defined.
    DeadEndBlocks DEBlocks(pass.F);
    llvm::SmallVector<SILInstruction *, 8> UsePoints;
    ValueLifetimeAnalysis VLA(defInst);
    ValueLifetimeAnalysis::Frontier Frontier;
    // This updates DominanceInfo if it splits a critical edge.
    VLA.computeFrontier(Frontier, ValueLifetimeAnalysis::AllowToModifyCFG,
                        &DEBlocks, pass.domInfo);
    return createAllocStack(defInst, Frontier);
  }
  return createAllocStack(&*pass.F->begin()->begin(), pass.exitingInsts);
}

//===----------------------------------------------------------------------===//
// AddressMaterialization - materialize storage addresses, generate
// projections.
//===----------------------------------------------------------------------===//

namespace {
/// Materialize the address of a value's storage. For values that are directly
/// mapped to a storage location, simply return the mapped `AllocStackInst`.
/// For subobjects emit any necessary `_addr` projections using the provided
/// `SILBuilder`.
///
/// This is a common utility for ParameterRewriter, ApplyRewriter,
/// AddressOnlyDefRewriter, and AddressOnlyUseRewriter.
class AddressMaterialization {
  AddressLoweringState &pass;
  SILBuilder &B;

public:
  AddressMaterialization(AddressLoweringState &pass, SILBuilder &B)
      : pass(pass), B(B) {}

  SILValue initializeOperandMem(Operand *operand);

  SILValue materializeAddress(SILValue origValue);

  SILValue materializeSwitchCase(SILPhiArgument *bbArg,
                                 ValueStorage &caseStorage);
  SILValue materializeProjectionFromDef(SILValue origValue);
  SILValue materializeProjectionFromNonBranchUse(Operand *operand);
};
} // anonymous namespace

/// Given the operand of an aggregate instruction (struct, tuple, enum),
/// materialize an address pointing to memory for this operand and ensure that
/// this memory is initialized with the subobject. Generates the address
/// projection and copy if needed.
SILValue AddressMaterialization::initializeOperandMem(Operand *operand) {
  SILValue def = operand->get();
  SILValue destAddr;
  if (def->getType().isAddressOnly(*pass.F)) {
    ValueStorage &storage = pass.valueStorageMap.getStorage(def);
    // Source value should already be rewritten.
    assert(storage.isRewritten);
    if (pass.valueStorageMap.isNonBranchUseProjection(operand))
      destAddr = storage.storageAddress;
    else {
      destAddr = materializeProjectionFromNonBranchUse(operand);
      B.createCopyAddr(operand->getUser()->getLoc(), storage.storageAddress,
                       destAddr, IsTake, IsInitialization);
    }
  } else {
    destAddr = materializeProjectionFromNonBranchUse(operand);
    B.createStore(operand->getUser()->getLoc(), operand->get(), destAddr,
                  StoreOwnershipQualifier::Unqualified);
  }
  return destAddr;
}

/// Return the address of the storage for `origValue`. This may involve
/// materializing projections.
///
/// As a side effect, record the materialized address as storage for
/// origValue.
SILValue AddressMaterialization::materializeAddress(SILValue origValue) {
  ValueStorage &storage = pass.valueStorageMap.getStorage(origValue);

  if (storage.storageAddress)
    return storage.storageAddress;

  // Handle a value that composes a user (struct/tuple/enum) or forward through
  // a block argument.
  if (storage.isUseProjection) {
    SILValue useVal = pass.valueStorageMap.getProjectedStorage(storage).value;
    if (auto *defInst = useVal->getDefiningInstruction()) {
      storage.storageAddress = materializeProjectionFromNonBranchUse(
        &defInst->getAllOperands()[storage.projectedOperandNum]);

    } else {
      // origValue is projected from either a block or function argument.
      assert(isa<SILArgument>(useVal));
      storage.storageAddress = materializeAddress(useVal);
    }
    return storage.storageAddress;
  }
  // Handle a value that is extracted from an aggregate.
  assert(storage.isDefProjection);
  if (auto *bbArg = dyn_cast<SILPhiArgument>(origValue))
    return materializeSwitchCase(bbArg, storage);

  if (isStoreCopy(origValue)) {
    storage.storageAddress =
      materializeAddress(cast<CopyValueInst>(origValue)->getOperand());
    return storage.storageAddress;
  }

  storage.storageAddress = materializeProjectionFromDef(origValue);
  return storage.storageAddress;
}

/// Materialize the address of a subobject.
///
/// \param origValue is be the value associated with the subobject
/// storage. Normally it will be the operand's user, except when it is a block
/// argument for a switch_enum.
SILValue
AddressMaterialization::materializeProjectionFromDef(SILValue origValue) {
  switch (origValue->getKind()) {
  default:
    llvm_unreachable("Unexpected projection from def.");

  case ValueKind::OpenExistentialValueInst:
  case ValueKind::OpenExistentialBoxValueInst:
    // Unlike struct_extract, tuple_extract, and switch_enum, there's no need to
    // materialize open_existential for non-opaque values.
    llvm_unreachable("open_existential should only be materialized once");

  case ValueKind::StructExtractInst: {
    auto *extractInst = cast<StructExtractInst>(origValue);
    SILValue srcAddr = materializeAddress(extractInst->getOperand());

    return B.createStructElementAddr(extractInst->getLoc(), srcAddr,
                                     extractInst->getField(),
                                     extractInst->getType().getAddressType());
  }
  case ValueKind::TupleExtractInst: {
    auto *extractInst = cast<TupleExtractInst>(origValue);
    SILValue srcAddr = materializeAddress(extractInst->getOperand());

    return B.createTupleElementAddr(extractInst->getLoc(), srcAddr,
                                    extractInst->getFieldNo(),
                                    extractInst->getType().getAddressType());
  }
  case ValueKind::SILPhiArgument: {
    // unchecked_take_enum_data_addr is destructive. It cannot be materialized
    // on demand.
    llvm_unreachable("Unimplemented switch_enum optimization");
  }
  case ValueKind::UncheckedEnumDataInst: {
    llvm_unreachable("unchecked_enum_data unimplemented"); //!!!
  }
  }
}

/// Materialize the address of a switch_enum case. This destroys the enum
/// storage, so it can only be called once per switch case.
SILValue
AddressMaterialization::materializeSwitchCase(SILPhiArgument *bbArg,
                                              ValueStorage &caseStorage) {
  assert(!caseStorage.isAllocated() && "only destroy the enum once.");
  
  auto *destBB = bbArg->getParent();
  Operand *switchOper = getProjectedDefOperand(bbArg);
  auto *SEI = cast<SwitchEnumInst>(switchOper->getUser());
  auto eltDecl = SEI->getUniqueCaseForDestination(destBB);
  assert(eltDecl && "No unique case found for destination block");

  SILValue enumAddr = materializeAddress(switchOper->get());
  caseStorage.storageAddress =
    B.createUncheckedTakeEnumDataAddr(SEI->getLoc(), enumAddr, eltDecl.get());
  return caseStorage.storageAddress;
}

/// Materialize the address of a subobject composing this operand's use. The
/// operand's user is an aggregate (struct, tuple, enum,
/// init_existential_value), or a terminator that reuses storage from a block
/// argument.
SILValue AddressMaterialization::materializeProjectionFromNonBranchUse(
    Operand *operand) {
  SILInstruction *user = operand->getUser();

  assert(!isa<TermInst>(user) || isa<ReturnInst>(user));

  switch (user->getKind()) {
  default:
    LLVM_DEBUG(user->dump());
    llvm_unreachable("Unexpected projection from use.");
  case SILInstructionKind::EnumInst: {
    auto *enumInst = cast<EnumInst>(user);
    SILValue enumAddr = materializeAddress(enumInst);
    return B.createInitEnumDataAddr(enumInst->getLoc(), enumAddr,
                                    enumInst->getElement(),
                                    operand->get()->getType().getAddressType());
  }
  case SILInstructionKind::InitExistentialValueInst: {
    auto *initExistentialValue = cast<InitExistentialValueInst>(user);
    SILValue containerAddr = materializeAddress(initExistentialValue);
    auto canTy = initExistentialValue->getFormalConcreteType();
    auto opaque = Lowering::AbstractionPattern::getOpaque();
    auto &concreteTL = pass.F->getTypeLowering(opaque, canTy);
    return B.createInitExistentialAddr(
        initExistentialValue->getLoc(), containerAddr, canTy,
        concreteTL.getLoweredType(), initExistentialValue->getConformances());
  }
  case SILInstructionKind::ReturnInst: {
    assert(pass.loweredFnConv.hasIndirectSILResults());
    return pass.F->getArguments()[0];
  }
  case SILInstructionKind::StructInst: {
    auto *structInst = cast<StructInst>(user);

    auto fieldIter = structInst->getStructDecl()->getStoredProperties().begin();
    std::advance(fieldIter, operand->getOperandNumber());

    SILValue structAddr = materializeAddress(structInst);
    return B.createStructElementAddr(
        structInst->getLoc(), structAddr, *fieldIter,
        operand->get()->getType().getAddressType());
  }
  case SILInstructionKind::TupleInst: {
    auto *tupleInst = cast<TupleInst>(user);
    // Function return values.
    if (tupleInst->hasOneUse()
        && isa<ReturnInst>(tupleInst->use_begin()->getUser())) {
      unsigned resultIdx = tupleInst->getElementIndex(operand);
      assert(resultIdx < pass.loweredFnConv.getNumIndirectSILResults());
      // Cannot call getIndirectSILResults here because that API uses the
      // original function type.
      return pass.F->getArguments()[resultIdx];
    }
    SILValue tupleAddr = materializeAddress(tupleInst);
    return B.createTupleElementAddr(tupleInst->getLoc(), tupleAddr,
                                    operand->getOperandNumber(),
                                    operand->get()->getType().getAddressType());
  }
  }
}

//===----------------------------------------------------------------------===//
// ParameterRewriter - rewrite call sites with indirect arguments.
//===----------------------------------------------------------------------===//

static void insertStackDeallocation(AllocStackInst *allocInst,
                                    SILBasicBlock::iterator insertionPoint) {
  SILBuilder deallocBuilder(insertionPoint);
  deallocBuilder.setSILConventions(
      SILModuleConventions::getLoweredAddressConventions());
  deallocBuilder.setCurrentDebugScope(allocInst->getDebugScope());
  deallocBuilder.createDeallocStack(allocInst->getLoc(), allocInst);
}

/// Deallocate temporary call-site stack storage.
///
/// `argLoad` is non-null for @out args that are loaded.
static void insertStackDeallocationAtCall(AllocStackInst *allocInst,
                                          SILInstruction *applyInst,
                                          SILInstruction *argLoad) {
  SILInstruction *lastUse = argLoad ? argLoad : applyInst;

  switch (applyInst->getKind()) {
  case SILInstructionKind::ApplyInst: {
    insertStackDeallocation(allocInst, std::next(lastUse->getIterator()));
    break;
  }
  case SILInstructionKind::TryApplyInst: {
    auto *TAI = cast<TryApplyInst>(applyInst);
    if (&*lastUse == applyInst)
      insertStackDeallocation(allocInst, TAI->getNormalBB()->begin());
    else
      insertStackDeallocation(allocInst, std::next(lastUse->getIterator()));

    insertStackDeallocation(allocInst, TAI->getErrorBB()->begin());
    break;
  }
  case SILInstructionKind::PartialApplyInst:
    llvm_unreachable("partial apply cannot have indirect results.");
  default:
    llvm_unreachable("not implemented for this instruction!");
  }
}

namespace {
/// Rewrite an apply parameter, lowering its indirect SIL arguments.
///
/// This rewritees one parameter at a time, replacing the incoming
/// object arguments with address-type arguments.
class ParameterRewriter {
  AddressLoweringState &pass;
  ApplySite apply;
  SILBuilderWithScope argBuilder;
  AddressMaterialization addrMat;

public:
  ParameterRewriter(ApplySite origCall, AddressLoweringState &pass)
    : pass(pass), apply(origCall), argBuilder(origCall.getInstruction()),
      addrMat(pass, argBuilder)
  {
    argBuilder.setSILConventions(
      SILModuleConventions::getLoweredAddressConventions(origCall.getModule()));
  }
  void rewriteParameters();
  void rewriteIndirectParameter(Operand *operand);
};
} // end anonymous namespace

/// Rewrite any indirect parameter in place.
void ParameterRewriter::rewriteParameters() {
  // Rewrite all incoming indirect operands.
  unsigned calleeArgIdx = apply.getCalleeArgIndexOfFirstAppliedArg();
  for (Operand &operand : apply.getArgumentOperands()) {
    if (operand.get()->getType().isObject()) {
      auto argConv =
          apply.getSubstCalleeConv().getSILArgumentConvention(calleeArgIdx);
      if (argConv.isIndirectConvention())
        rewriteIndirectParameter(&operand);
    }
    ++calleeArgIdx;
  }
}

/// Rewrite a formally indirect parameter in place.
/// Update the operand to the incoming value's storage address.
/// After this, the SIL argument types no longer match SIL function conventions.
///
/// Temporary argument storage may be created for loadable values.
///
/// Note: Temporary argument storage does not own its value. If the argument
/// is owned, the stored value should already have been copied.
void ParameterRewriter::rewriteIndirectParameter(Operand *operand) {
  SILValue argValue = operand->get();

  if (argValue->getType().isAddressOnly(*pass.F)) {
    ValueStorage &storage = pass.valueStorageMap.getStorage(argValue);
    // Source value should already be rewritten.
    assert(storage.isRewritten);
    operand->set(storage.storageAddress);
    return;
  }
  // Allocate temporary storage for a loadable operand.
  AllocStackInst *allocInstr =
      argBuilder.createAllocStack(apply.getLoc(), argValue->getType());

  argBuilder.createStore(apply.getLoc(), argValue, allocInstr,
                         StoreOwnershipQualifier::Unqualified);

  operand->set(allocInstr);

  insertStackDeallocationAtCall(allocInstr, apply.getInstruction(),
                                /*argLoad=*/nullptr);
}

//===----------------------------------------------------------------------===//
// ApplyRewriter - rewrite call sites with indirect results.
//===----------------------------------------------------------------------===//

namespace {
/// Rewrite an Apply, lowering its indirect SIL results.
///
/// Once any result needs to be rewritten, then the entire apply is
/// replaced. Creates new indirect result arguments for this function to
/// represent the caller's storage.
class ApplyRewriter {
  AddressLoweringState &pass;
  // Use argBuilder for building incoming arguments and materializing
  // addresses.
  SILBuilderWithScope argBuilder;
  // Use resultBuilder for loading results.
  SILBuilder resultBuilder;
  AddressMaterialization addrMat;
  SILFunctionConventions origCalleeConv;
  SILFunctionConventions loweredCalleeConv;
  SILLocation callLoc;

  // This apply site mutates when the new apply instruction is generated.
  ApplySite apply;

public:
  ApplyRewriter(ApplySite origCall, AddressLoweringState &pass)
      : pass(pass), argBuilder(origCall.getInstruction()),
        resultBuilder(*origCall.getFunction()), addrMat(pass, argBuilder),
        origCalleeConv(origCall.getSubstCalleeConv()),
        loweredCalleeConv(origCall.getSubstCalleeType(),
                          SILModuleConventions::getLoweredAddressConventions()),
        callLoc(origCall.getLoc()), apply(origCall) {

    argBuilder.setSILConventions(
        SILModuleConventions::getLoweredAddressConventions());
    resultBuilder.setSILConventions(
      SILModuleConventions::getLoweredAddressConventions());
    resultBuilder.setCurrentDebugScope(origCall.getDebugScope());
    resultBuilder.setInsertionPoint(getCallResultInsertionPoint());
  }

  void convertApplyWithIndirectResults();

protected:
  SILBasicBlock::iterator getCallResultInsertionPoint() {
    if (isa<ApplyInst>(apply))
      return std::next(SILBasicBlock::iterator(apply.getInstruction()));

    auto *bb = cast<TryApplyInst>(apply)->getNormalBB();
    return bb->begin();
  }

  void getOriginalDirectResultValues(
    SmallVectorImpl<SILValue> &directResultValues);

  void canonicalizeResults(MutableArrayRef<SILValue> directResultValues,
                           ArrayRef<Operand *> nonCanonicalUses);

  void
  getNewCallArgsAndResults(SmallVectorImpl<SILValue> &newCallArgs,
                           SmallVectorImpl<unsigned> &newDirectResultIndices);

  SILValue materializeIndirectResultAddress(SILValue origDirectResultVal,
                                            SILType argTy);

  ApplyInst *genApply(ApplyInst *AI, ArrayRef<SILValue> newCallArgs);
  SILValue rewriteTryApply(TryApplyInst *TAI, ArrayRef<SILValue> newCallArgs);

  void rewriteResult(SILInstruction *useInst,
                     ArrayRef<unsigned> newDirectResultIndices,
                     SILValue origCallResult, SILValue newCallResult);

  void removeCall(ApplyInst *AI);
};
} // end anonymous namespace

/// Top-level entry: Allocate storage for formally indirect results at the given
/// call site. Create a new call instruction with indirect SIL arguments.
void ApplyRewriter::convertApplyWithIndirectResults() {
  // Gather information from the old apply before rewriting it.

  // List of new call arguments.
  SmallVector<SILValue, 8> newCallArgs(loweredCalleeConv.getNumSILArguments());
  // Map the original result indices to new result indices.
  SmallVector<unsigned, 8> newDirectResultIndices(
      origCalleeConv.getNumDirectSILResults());
  getNewCallArgsAndResults(newCallArgs, newDirectResultIndices);
  
  // Collect the original uses before potentialily removing the call
  // pseudo-result (e.g. try_apply's bb arg).
  SmallVector<SILInstruction *, 8> origUsers(
    makeUserRange(getCallPseudoResult(apply)));

  switch (apply.getInstruction()->getKind()) {
  case SILInstructionKind::ApplyInst: {
    auto *origCallInst = cast<ApplyInst>(apply.getInstruction());
    ApplyInst *newCallInst = genApply(origCallInst, newCallArgs);

    // Replace all unmapped uses of the original call with uses of the new call.
    for (SILInstruction *useInst : origUsers) {
      rewriteResult(useInst, newDirectResultIndices,
                    getCallPseudoResult(origCallInst),
                    getCallPseudoResult(newCallInst));
    }
    this->apply = ApplySite(newCallInst);
    // If this call won't be visited as an opaque value def, mark it deleted.
    // Note: Simply checking whether this call is address-only is insufficient,
    // because the address-only use may be have been dead to begin with.
    if (!pass.valueStorageMap.contains(origCallInst)) {
      pass.markDead(origCallInst);
      //!!!removeCall(origCallInst);
    }
    return;
  }
  case SILInstructionKind::TryApplyInst: {
    // this->apply will be updated with the new try_apply instruction.
    // The returned origResult is a placeHolder mapped to the original result
    // storage.
    SILValue origResult = rewriteTryApply(
        cast<TryApplyInst>(apply.getInstruction()), newCallArgs);
    // Replace unmapped uses of the original call with uses of the new call.
    for (SILInstruction *useInst : origUsers) {
      rewriteResult(useInst, newDirectResultIndices, origResult,
                    getCallPseudoResult(apply));
    }
    return;
  }
  case SILInstructionKind::PartialApplyInst:
    llvm_unreachable("partial_apply cannot have indirect results.");
  default:
    llvm_unreachable("Unexpected call type.");
  };
}

// Gather the original direct return values.
// Canonicalize results so no user uses more than one result.
void ApplyRewriter::getOriginalDirectResultValues(
  SmallVectorImpl<SILValue> &origDirectResultValues) {

  SILValue origCallValue = getCallPseudoResult(apply);
  SmallVector<Operand *, 4> nonCanonicalUses;
  if (origCallValue->getType().is<TupleType>()) {
    for (Operand *operand : origCallValue->getUses()) {
      if (auto *extract = dyn_cast<TupleExtractInst>(operand->getUser()))
        origDirectResultValues[extract->getFieldNo()] = extract;
      else
        nonCanonicalUses.push_back(operand);
    }
    if (!nonCanonicalUses.empty())
      canonicalizeResults(origDirectResultValues, nonCanonicalUses);
  } else {
    // This call has a single result. Convert it to an indirect
    // result. (convertApplyWithIndirectResults is only invoked for calls with
    // at least one indirect result). An unused result can remain
    // unmapped. Temporary storage will be allocated later when fixing up the
    // call's uses.
    assert(origDirectResultValues.size() == 1);
    origDirectResultValues[0] = origCallValue;
  }
}

// Canonicalize call result uses. Treat each result of a multi-result call as
// an independent value. Currently, SILGen may generate tuple_extract for each
// result but generate a single destroy_value for the entire tuple of
// results. This makes it impossible to reason about each call result as an
// independent value according to the callee's function type.
//
// directResultValues has an entry for each tuple extract corresponding to
// that result if one exists. This function will add an entry to
// directResultValues whenever it needs to materialize a TupleExtractInst.
void ApplyRewriter::canonicalizeResults(
    MutableArrayRef<SILValue> directResultValues,
    ArrayRef<Operand *> nonCanonicalUses) {

  for (Operand *operand : nonCanonicalUses) {
    auto *destroyInst = dyn_cast<DestroyValueInst>(operand->getUser());
    if (!destroyInst)
      llvm::report_fatal_error("Simultaneous use of multiple call results.");

    for (unsigned resultIdx : indices(directResultValues)) {
      SILValue result = directResultValues[resultIdx];
      if (!result) {
        SILBuilder resultBuilder(getCallResultInsertionPoint());
        resultBuilder.setSILConventions(
            SILModuleConventions::getLoweredAddressConventions());
        resultBuilder.setCurrentDebugScope(
            apply.getInstruction()->getDebugScope());
        result = resultBuilder.createTupleExtract(
            apply.getInstruction()->getLoc(), getCallPseudoResult(apply),
            resultIdx);
        directResultValues[resultIdx] = result;
      }
      SILBuilderWithScope B(destroyInst);
      B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
      B.emitDestroyValueOperation(destroyInst->getLoc(), result);
    }
    destroyInst->eraseFromParent();
  }
}

// Get the new call instruction's SIL argument list and a map from old direct
// result indices to new direct result indices.
void ApplyRewriter::getNewCallArgsAndResults(
    SmallVectorImpl<SILValue> &newCallArgs,
    SmallVectorImpl<unsigned> &newDirectResultIndices) {

  // List of the original results.
  SmallVector<SILValue, 4> origDirectResultValues(
      origCalleeConv.getNumDirectSILResults());
  getOriginalDirectResultValues(origDirectResultValues);

  // Indices used to populate newDirectResultIndices.
  unsigned oldDirectResultIdx = 0, newDirectResultIdx = 0;

  // The index of the next indirect result argument.
  unsigned newResultArgIdx =
      loweredCalleeConv.getSILArgIndexOfFirstIndirectResult();

  // Visit each result. Redirect results that are now indirect by calling
  // materializeIndirectResultAddress. Results that remain direct will be
  // redirected later. Populate newCallArgs and newDirectResultIndices.
  for_each(
      apply.getSubstCalleeType()->getResults(), origDirectResultValues,
      [&](SILResultInfo resultInfo, SILValue origDirectResultVal) {
        // Assume that all original results are direct in SIL.
        assert(!origCalleeConv.isSILIndirect(resultInfo));

        if (loweredCalleeConv.isSILIndirect(resultInfo)) {
          SILValue indirectResultAddr = materializeIndirectResultAddress(
              origDirectResultVal, loweredCalleeConv.getSILType(resultInfo));
          // Record the new indirect call argument.
          newCallArgs[newResultArgIdx++] = indirectResultAddr;
          // Leave a placeholder for indirect results.
          newDirectResultIndices[oldDirectResultIdx++] = ~0;
        } else {
          // Record the new direct result, and advance the direct result
          // indices.
          newDirectResultIndices[oldDirectResultIdx++] = newDirectResultIdx++;
        }
        // replaceAllUses will be called later to handle direct results that
        // remain direct results of the new call instruction.
      });

  // Append the existing call arguments to the SIL argument list. They were
  // already lowered to addresses by rewriteIncomingArgument.
  assert(newResultArgIdx == loweredCalleeConv.getSILArgIndexOfFirstParam());
  unsigned origArgIdx = apply.getSubstCalleeConv().getSILArgIndexOfFirstParam();
  for (unsigned endIdx = newCallArgs.size(); newResultArgIdx < endIdx;
       ++newResultArgIdx, ++origArgIdx) {
    newCallArgs[newResultArgIdx] = apply.getArgument(origArgIdx);
  }
}

/// Return the storage address for the indirect result corresponding to the
/// given original result value. Allocate temporary argument storage for any
/// indirect results that are unmapped because they are loadable or unused.
///
/// origDirectResultVal may be nullptr for unused results.
SILValue
ApplyRewriter::materializeIndirectResultAddress(SILValue origDirectResultVal,
                                                SILType argTy) {

  if (origDirectResultVal
      && origDirectResultVal->getType().isAddressOnly(*pass.F)) {

    // For normal calls, the tuple_extract result may not have been visited yet.
    addrMat.materializeAddress(origDirectResultVal);

    auto &storage = pass.valueStorageMap.getStorage(origDirectResultVal);
    storage.markRewritten();
    return storage.storageAddress;
  }
  // Allocate temporary call-site storage for an unused or loadable result.
  SILInstruction *origCallInst = apply.getInstruction();
  SILLocation loc = origCallInst->getLoc();
  auto *allocInst = argBuilder.createAllocStack(loc, argTy);
  LoadInst *loadInst = nullptr;
  if (origDirectResultVal) {
    // Build results outside-in to next stack allocations.
    SILBuilder resultBuilder(getCallResultInsertionPoint());
    resultBuilder.setSILConventions(
        SILModuleConventions::getLoweredAddressConventions(pass.F->getModule()));
    resultBuilder.setCurrentDebugScope(origCallInst->getDebugScope());
    if (!origDirectResultVal->use_empty()) {
      // This is a formally indirect argument, but is loadable.
      loadInst = resultBuilder.createLoad(loc, allocInst,
                                          LoadOwnershipQualifier::Unqualified);
      origDirectResultVal->replaceAllUsesWith(loadInst);
    }
    if (auto *resultInst = origDirectResultVal->getDefiningInstruction())
      pass.markDead(resultInst);
  }
  insertStackDeallocationAtCall(allocInst, origCallInst, loadInst);
  return SILValue(allocInst);
}

ApplyInst *ApplyRewriter::genApply(ApplyInst *AI,
                                   ArrayRef<SILValue> newCallArgs) {
  return argBuilder.createApply(
      callLoc, apply.getCallee(), apply.getSubstitutionMap(), newCallArgs,
      AI->isNonThrowing(), AI->getSpecializationInfo());
}

/// Replace the given try_apply with a new try_apply using the given
/// newCallArgs. Return the value representing the original (address-only)
/// result of the try_apply, which will now be replaced with a fake load.
/// If the original result was not address-only return an invalid SILValue.
///
/// Update this->apply with the new call instruction.
SILValue ApplyRewriter::rewriteTryApply(TryApplyInst *TAI,
                                        ArrayRef<SILValue> newCallArgs) {

  auto *newCallInst = argBuilder.createTryApply(
      callLoc, apply.getCallee(), apply.getSubstitutionMap(), newCallArgs,
      TAI->getNormalBB(), TAI->getErrorBB(), TAI->getSpecializationInfo());

  SILPhiArgument *resultArg = getTryApplyPseudoResult(TAI);

  auto replacePhi = [&](SILValue newResultVal) {
    SILType resultTy = loweredCalleeConv.getSILResultType();
    auto ownership = resultTy.isTrivial(*pass.F)
                         ? ValueOwnershipKind::Any
                         : ValueOwnershipKind::Owned;
    resultArg->replaceAllUsesWith(newResultVal);
    resultArg->getParent()->replacePhiArgument(0, resultTy, ownership,
                                               resultArg->getDecl());
  };
  // Immediately delete the old try_apply (old applies hang around until
  // dead code removal because they define values).
  TAI->eraseFromParent();
  this->apply = ApplySite(newCallInst);

  if (pass.valueStorageMap.contains(resultArg)) {
    // Rewriting the apply with a new result type requires erasing any opaque
    // block arguments.  Create dummy loads to stand in for those block
    // arguments until everything has been rewritten. Just load from undef
    // since, for tuple results, there's no storage address to load from.
    LoadInst *loadArg = resultBuilder.createLoad(
        callLoc,
        SILUndef::get(resultArg->getType().getAddressType(), *pass.F),
        LoadOwnershipQualifier::Unqualified);

    assert(!resultArg->getType().is<TupleType>());

    // Storage was materialized by materializeIndirectResultAddress.
    auto &origStorage = pass.valueStorageMap.getStorage(resultArg);
    assert(origStorage.isRewritten);
    (void)origStorage;

    pass.valueStorageMap.replaceValue(resultArg, loadArg);
    replacePhi(loadArg);
    return loadArg;
  }
  // Loadable results were loaded by materializeIndirectResultAddress.
  assert(resultArg->getIndex() == 0);
  // Temporarily redirect all uses to Undef. They will be fixed in
  // rewriteResults.
  replacePhi(SILUndef::get(resultArg->getType().getAddressType(), *pass.F));
  return SILValue();
}

// Replace any unmapped use of the original call's pseudo-result with a use of
// the new call.
//
// origCallResult is only still valid for single-value calls.
void ApplyRewriter::rewriteResult(SILInstruction *useInst,
                                  ArrayRef<unsigned> newDirectResultIndices,
                                  SILValue origCallResult,
                                  SILValue newCallResult) {
  auto *extractInst = dyn_cast<TupleExtractInst>(useInst);
  if (!extractInst) {
    // If the original call produces a single result, it was replaced when
    // materializing the new call arguments.
    // origCallResult must still be valid because we can't delete a call that
    // produces a single value mapped to storage.
    assert(origCalleeConv.getNumDirectSILResults() == 1);
    assert(pass.valueStorageMap.getStorage(origCallResult).isRewritten);
    return;
  }
  unsigned origResultIdx = extractInst->getFieldNo();
  auto resultInfo = origCalleeConv.getResults()[origResultIdx];

  if (extractInst->getType().isAddressOnly(*pass.F)) {
    // Uses of indirect results will be rewritten by AddressOnlyUseRewriter.
    assert(loweredCalleeConv.isSILIndirect(resultInfo));
    // For try_apply, the operand was temporarily rewritten to undef.
    extractInst->setOperand(newCallResult);
    // Mark the extract as rewritten now so we don't attempt to convert the
    // call again.
    pass.valueStorageMap.getStorage(extractInst).markRewritten();
    return;
  }
  if (loweredCalleeConv.isSILIndirect(resultInfo)) {
    // This loadable indirect use should already be redirected to a load from
    // the argument storage and marked dead.
    assert(extractInst->use_empty() && pass.isDead(extractInst));
    return;
  }
  // Either the new call instruction has only a single direct result, or we
  // map the original tuple field to the new tuple field.
  SILValue newResultVal = newCallResult;
  if (loweredCalleeConv.getNumDirectSILResults() > 1) {
    assert(newCallResult->getType().is<TupleType>());
    newResultVal =
        resultBuilder.createTupleExtract(extractInst->getLoc(), newCallResult,
                                         newDirectResultIndices[origResultIdx]);
  }
  // Since this is a loadable type, there's no associated storage, so erasing
  // the instruction and rewriting is use operands does not invalidate
  // ValueStorage.
  extractInst->replaceAllUsesWith(newResultVal);
  extractInst->eraseFromParent();
}
>>>>>>> 357afa0047e (AddressLowering pass fixes.)

// This call itself has no storage. If it is used by an address_only
// tuple_extract, then it will be marked deleted only after that use is marked
// deleted. Otherwise all its uses must already be dead and it must be marked
// dead immediately.
void ApplyRewriter::removeCall(ApplyInst *AI) {
  for (Operand *use : AI->getUses()) {
    auto *extract = cast<TupleExtractInst>(use->getUser());
    if (pass.valueStorageMap.contains(extract)) {
    // At least one of the results has storage. The call cannot be marked dead
    // until its results are marked dead.
#ifndef NDEBUG
      pass.callsToDelete.insert(AI);
#endif
      return;
    }
    assert(pass.isDead(extract));
  }
  pass.markDead(AI);
}

//===----------------------------------------------------------------------===//
// ReturnRewriter - rewrite return instructions for indirect results.
//===----------------------------------------------------------------------===//

class ReturnRewriter {
  AddressLoweringState &pass;

public:
  ReturnRewriter(AddressLoweringState &pass) : pass(pass) {}

  void rewriteReturns();

protected:
  void rewriteReturn(ReturnInst *returnInst);
};

void ReturnRewriter::rewriteReturns() {
  for (SILInstruction *termInst : pass.exitingInsts) {
    if (auto *returnInst = dyn_cast<ReturnInst>(termInst))
      rewriteReturn(returnInst);
    else
      assert(isa<ThrowInst>(termInst));
  }
}

void ReturnRewriter::rewriteReturn(ReturnInst *returnInst) {
  auto insertPt = SILBasicBlock::iterator(returnInst);
  auto bbStart = returnInst->getParent()->begin();
  while (insertPt != bbStart) {
    --insertPt;
    if (!isa<DeallocStackInst>(*insertPt))
      break;
  }
  SILBuilderWithScope B(insertPt);
  B.setSILConventions(SILModuleConventions::getLoweredAddressConventions(pass.F->getModule()));

  // Gather direct function results.
  unsigned numOrigDirectResults =
      pass.F->getConventions().getNumDirectSILResults();
  SmallVector<SILValue, 8> origDirectResultValues;
  if (numOrigDirectResults == 1)
    origDirectResultValues.push_back(returnInst->getOperand());
  else {
    auto *tupleInst = cast<TupleInst>(returnInst->getOperand());
    origDirectResultValues.append(tupleInst->getElements().begin(),
                                  tupleInst->getElements().end());
    assert(origDirectResultValues.size() == numOrigDirectResults);
  }

  SILFunctionConventions origFnConv(pass.F->getConventions());
  (void)origFnConv;

  // Convert each result.
  SmallVector<SILValue, 8> newDirectResults;
  unsigned newResultArgIdx =
      pass.loweredFnConv.getSILArgIndexOfFirstIndirectResult();

  for_each(
      pass.F->getLoweredFunctionType()->getResults(), origDirectResultValues,
      [&](SILResultInfo resultInfo, SILValue origDirectResultVal) {
        // Assume that all original results are direct in SIL.
        assert(!origFnConv.isSILIndirect(resultInfo));

        if (pass.loweredFnConv.isSILIndirect(resultInfo)) {
          assert(newResultArgIdx
                 < pass.loweredFnConv.getSILArgIndexOfFirstParam());

          SILArgument *resultArg = B.getFunction().getArgument(newResultArgIdx);
          SILType resultTy = origDirectResultVal->getType();
          if (resultTy.isAddressOnly(*pass.F)) {
            ValueStorage &storage =
                pass.valueStorageMap.getStorage(origDirectResultVal);
            assert(storage.isRewritten);
            SILValue resultAddr = storage.storageAddress;
            if (resultAddr != resultArg) {
              // Copy the result from local storage into the result argument.
              B.createCopyAddr(returnInst->getLoc(), resultAddr, resultArg,
                               IsTake, IsInitialization);
            }
          } else {
            // Store the result into the result argument.
            B.createStore(returnInst->getLoc(), origDirectResultVal, resultArg,
                          StoreOwnershipQualifier::Unqualified);
          }
          ++newResultArgIdx;
        } else {
          // Record the direct result for populating the result tuple.
          newDirectResults.push_back(origDirectResultVal);
        }
      });
  assert(newDirectResults.size()
         == pass.loweredFnConv.getNumDirectSILResults());
  SILValue newReturnVal;
  if (newDirectResults.empty()) {
    SILType emptyTy = SILType::getPrimitiveObjectType(
        B.getModule().getASTContext().TheEmptyTupleType);
    newReturnVal = B.createTuple(returnInst->getLoc(), emptyTy, {});
  } else if (newDirectResults.size() == 1) {
    newReturnVal = newDirectResults[0];
  } else {
    newReturnVal = B.createTuple(
        returnInst->getLoc(),
        pass.loweredFnConv.getSILResultType(B.getTypeExpansionContext()),
        newDirectResults);
  }
  // Rewrite the returned value.
  SILValue origFullResult = returnInst->getOperand();
  returnInst->setOperand(newReturnVal);
  if (auto *fullResultInst = origFullResult->getDefiningInstruction()) {
    if (!fullResultInst->hasUsesOfAnyResult())
      pass.markDead(fullResultInst);
  }
}

//===----------------------------------------------------------------------===//
// AddressOnlyUseRewriter - rewrite opaque value uses.
//===----------------------------------------------------------------------===//

namespace {
class AddressOnlyUseRewriter
    : SILInstructionVisitor<AddressOnlyUseRewriter> {
  friend SILVisitorBase<AddressOnlyUseRewriter>;
  friend SILInstructionVisitor<AddressOnlyUseRewriter>;

  AddressLoweringState &pass;

  SILBuilder B;
  AddressMaterialization addrMat;

  Operand *currOper;

public:
  explicit AddressOnlyUseRewriter(AddressLoweringState &pass)
      : pass(pass), B(*pass.F), addrMat(pass, B) {
    B.setSILConventions(
      SILModuleConventions::getLoweredAddressConventions(pass.F->getModule()));
  }

  void visitOperand(Operand *operand) {
    currOper = operand;
    // Special handling for opened archetypes because a single result somehow
    // produces both a value of the opened type and the metatype itself :/
    if (operand->getUser()->isTypeDependentOperand(*operand)) {
      CanType openedTy = operand->get()->getType().getASTType();
      assert(openedTy->isOpenedExistential());
      SILValue archetypeDef =
          pass.openedArchetypesTracker.getOpenedArchetypeDef(
              CanArchetypeType(openedTy->castTo<ArchetypeType>()));
      operand->set(archetypeDef);
      return;
    }
    visit(operand->getUser());
  }

protected:
  void markRewritten(SILValue oldValue, SILValue addr) {
    auto &storage = pass.valueStorageMap.getStorage(oldValue);
    storage.storageAddress = addr;
    storage.markRewritten();
  }

  void beforeVisit(SILInstruction *I) {
    LLVM_DEBUG(llvm::dbgs() << "  REWRITE USE "; I->dump());

    B.setInsertionPoint(I);
    B.setCurrentDebugScope(I->getDebugScope());
  }

  void visitSILInstruction(SILInstruction *I) {
    LLVM_DEBUG(I->dump());
    llvm_unreachable("Unimplemented opaque use.");
  }

  // Opaque call argument.
  void visitApplyInst(ApplyInst *applyInst) {
    ParameterRewriter(applyInst, pass).rewriteIndirectParameter(currOper);
  }

  // Opaque branch argument.
  void visitBranchInst(BranchInst *BI) {
    SILValue predStorageAddress =
        pass.valueStorageMap.getStorage(currOper->get()).storageAddress;

    auto *bbArg = cast<SILPhiArgument>(BI->getArgForOperand(currOper));
    SILValue succStorageAddress = addrMat.materializeAddress(bbArg);

    // If this branch argument is not a projection of the block argument, then
    // copy its value in to block arg storage.
    if (predStorageAddress != succStorageAddress) {
      B.createCopyAddr(BI->getLoc(), predStorageAddress, succStorageAddress,
                       IsTake, IsInitialization);
    }
    // Set this operand to Undef. Dead block arguments are removed after
    // rewriting.
    currOper->set(
        SILUndef::get(currOper->get()->getType(), *pass.F));
  }

  // Opaque checked cast source.
  void visitCheckedCastValueBranchInst(
      CheckedCastValueBranchInst *checkedBranchInst) {
    llvm::report_fatal_error("Unimplemented CheckCastValueBranch use.");
  }

  // Copy from an opaque source operand.
  void visitCopyValueInst(CopyValueInst *copyInst) {
    // store-copy pairs are already rewritten.
    if (pass.valueStorageMap.getStorage(copyInst).isRewritten)
      return;
    
    SILValue srcVal = copyInst->getOperand();
    SILValue srcAddr = pass.valueStorageMap.getStorage(srcVal).storageAddress;
    SILValue destAddr = addrMat.materializeAddress(copyInst);
    B.createCopyAddr(copyInst->getLoc(), srcAddr, destAddr, IsNotTake,
                     IsInitialization);
    markRewritten(copyInst, destAddr);
  }

  // Opaque conditional branch argument.
  void visitCondBranchInst(CondBranchInst *CBI) {
    SILValue predStorageAddress =
        pass.valueStorageMap.getStorage(currOper->get()).storageAddress;

    unsigned operIdx = currOper->getOperandNumber();
    bool isTruePath = CBI->isTrueOperandIndex(operIdx);
    unsigned argIdx =
        operIdx
        - (isTruePath ? CBI->getFalseOperands()[0].getOperandNumber()
                      : CBI->getTrueOperands()[0].getOperandNumber());

    auto *destBB = isTruePath ? CBI->getTrueBB() : CBI->getFalseBB();
    auto *bbArg = cast<SILPhiArgument>(destBB->getArgument(argIdx));
    SILValue succStorageAddress = addrMat.materializeAddress(bbArg);

    // If this branch argument is not a projection of the block argument, then
    // copy its value in to block arg storage.
    if (predStorageAddress != succStorageAddress) {
      B.createCopyAddr(CBI->getLoc(), predStorageAddress, succStorageAddress,
                       IsTake, IsInitialization);
    }
    // Set this operand to Undef. Dead block arguments are removed after
    // rewriting.
    //
    // FIXME: Make sure this is done such that ValueStorage projections can't be
    // referenced once we replace an operand with Undef or erase an instruction
    // or argument.
    currOper->set(SILUndef::get(currOper->get()->getType(), *pass.F));

    ValueStorage &storage = pass.valueStorageMap.getStorage(currOper->get());
    currOper->set(storage.storageAddress);
    llvm::report_fatal_error("Untested."); //!!!
  }

  void visitDebugValueInst(DebugValueInst *debugInst) {
    SILValue srcVal = debugInst->getOperand();
    SILValue srcAddr = pass.valueStorageMap.getStorage(srcVal).storageAddress;
    B.createDebugValueAddr(debugInst->getLoc(), srcAddr,
                           *debugInst->getVarInfo());
    pass.markDead(debugInst);
  }

  void visitDeinitExistentialValueInst(
      DeinitExistentialValueInst *deinitExistential) {
    llvm::report_fatal_error("Unimplemented DeinitExsitentialValue use.");
  }

  void visitDestroyValueInst(DestroyValueInst *destroyInst) {
    SILValue srcVal = destroyInst->getOperand();
    SILValue srcAddr = pass.valueStorageMap.getStorage(srcVal).storageAddress;
    B.createDestroyAddr(destroyInst->getLoc(), srcAddr);
    pass.markDead(destroyInst);
  }

  // Opaque enum payload. Handle EnumInst on the def side to handle both opaque
  // and loadable operands.
  void visitEnumInst(EnumInst *enumInst) {}

  // Initialize an existential with an opaque payload.
  // (Handle InitExistentialValue on the def side to handle both opaque
  // and loadable operands.)
  void
  visitInitExistentialValueInst(InitExistentialValueInst *initExistential) {}

  // Opening an opaque existential. Rewrite the opened existentials here on the
  // use-side because it may produce either loadable or address-only types.
  void
  visitOpenExistentialValueInst(OpenExistentialValueInst *openExistential) {
    assert(currOper == getProjectedDefOperand(openExistential));
    SILValue srcAddr =
        pass.valueStorageMap.getStorage(currOper->get()).storageAddress;
    // Mutable access is always by address.
    auto *OEA =
        B.createOpenExistentialAddr(openExistential->getLoc(), srcAddr,
                                    openExistential->getType().getAddressType(),
                                    OpenedExistentialAccess::Immutable);
    // Henceforth track this operned archetype using open_existential_addr.
    pass.openedArchetypesTracker.unregisterOpenedArchetypes(openExistential);
    pass.openedArchetypesTracker.registerOpenedArchetypes(OEA);
    markRewritten(openExistential, OEA);
  }

  void visitOpenExistentialBoxValueInst(
      OpenExistentialBoxValueInst *openExistentialBox) {
    llvm::report_fatal_error("Unimplemented OpenExistentialBox use.");
  }

  void visitReleaseValueInst(ReleaseValueInst *releaseInst) {
    SILValue srcVal = releaseInst->getOperand();
    ValueStorage &storage = pass.valueStorageMap.getStorage(srcVal);
    SILValue srcAddr = storage.storageAddress;

    B.createReleaseValueAddr(releaseInst->getLoc(), srcAddr,
                             releaseInst->getAtomicity());
    pass.markDead(releaseInst);
  }

  void visitRetainValueInst(RetainValueInst *retainInst) {
    SILValue srcVal = retainInst->getOperand();
    ValueStorage &storage = pass.valueStorageMap.getStorage(srcVal);
    SILValue srcAddr = storage.storageAddress;

    B.createRetainValueAddr(retainInst->getLoc(), srcAddr,
                            retainInst->getAtomicity());
    pass.markDead(retainInst);
  }

  void visitReturnInst(ReturnInst *returnInst) {
    // Returns are rewritten for any function with indirect results after opaque
    // value rewriting.
  }

  void visitSelectValueInst(SelectValueInst *selectInst) {
    llvm::report_fatal_error("Unimplemented SelectValue use.");
  }

  // Opaque enum operand to a switch_enum.
  void visitSwitchEnumInst(SwitchEnumInst *SEI);

  void visitStoreInst(StoreInst *storeInst) {
    SILValue srcVal = storeInst->getSrc();
    assert(currOper->get() == srcVal);

    ValueStorage &storage = pass.valueStorageMap.getStorage(srcVal);
    SILValue srcAddr = storage.storageAddress;

    IsTake_t isTake = IsTake;
    if (auto *copy = dyn_cast<CopyValueInst>(srcVal)) {
      if (storage.isDefProjection) {
        SILValue copySrcAddr =
          pass.valueStorageMap.getStorage(copy->getOperand()).storageAddress;
        assert(srcAddr == copySrcAddr && "folded copy should borrow storage");
        (void)copySrcAddr;
        isTake = IsNotTake;
      }
    }
    IsInitialization_t isInit;
    auto qualifier = storeInst->getOwnershipQualifier();
    if (qualifier == StoreOwnershipQualifier::Init)
      isInit = IsInitialization;
    else {
      assert(qualifier == StoreOwnershipQualifier::Assign);
      isInit = IsNotInitialization;
    }
    B.createCopyAddr(storeInst->getLoc(), srcAddr, storeInst->getDest(),
                     isTake, isInit);
    pass.markDead(storeInst);
  }

  // Extract from an opaque struct.
  void visitStructExtractInst(StructExtractInst *extractInst) {
    SILValue extractAddr = addrMat.materializeProjectionFromDef(extractInst);
    if (extractInst->getType().isAddressOnly(*pass.F)) {
      assert(currOper == getProjectedDefOperand(extractInst));
      markRewritten(extractInst, extractAddr);
    } else {
      assert(!pass.valueStorageMap.contains(extractInst));
      LoadInst *loadElement = B.createLoad(extractInst->getLoc(), extractAddr,
                                           LoadOwnershipQualifier::Unqualified);
      extractInst->replaceAllUsesWith(loadElement);
      pass.markDead(extractInst);
    }
  }

  // Opaque struct member.
  void visitStructInst(StructInst *structInst) {
    // Structs are rewritten on the def-side, where both the direct and indirect
    // elements that compose a struct can be handled.
  }

  // Opaque call argument.
  void visitTryApplyInst(TryApplyInst *tryApplyInst) {
    ParameterRewriter(tryApplyInst, pass).rewriteIndirectParameter(currOper);
  }

  // Opaque tuple element.
  void visitTupleInst(TupleInst *tupleInst) {
    // Tuples are rewritten on the def-side, where both the direct and indirect
    // elements that compose a struct can be handled.
  }

  // Extract from an opaque tuple.
  void visitTupleExtractInst(TupleExtractInst *extractInst) {
    // Apply results are rewritten when the result definition is visited.
    if (ApplySite::isa(currOper->get()))
      return;

    SILValue extractAddr = addrMat.materializeProjectionFromDef(extractInst);
    if (extractInst->getType().isAddressOnly(*pass.F)) {
      assert(currOper == getProjectedDefOperand(extractInst));
      markRewritten(extractInst, extractAddr);
    } else {
      assert(!pass.valueStorageMap.contains(extractInst));
      LoadInst *loadElement = B.createLoad(extractInst->getLoc(), extractAddr,
                                           LoadOwnershipQualifier::Unqualified);
      extractInst->replaceAllUsesWith(loadElement);
      pass.markDead(extractInst);
    }
    llvm_unreachable("Untested."); //!!!
  }

  void visitUncheckedBitwiseCast(UncheckedBitwiseCastInst *uncheckedCastInst) {
    llvm_unreachable("Unimplemented UncheckedBitwiseCast use.");
  }

  void visitUnconditionalCheckedCastValueInst(
      UnconditionalCheckedCastValueInst *checkedCastInst) {

    llvm_unreachable("Unimplemented UnconditionalCheckedCast use.");
  }
};
} // end anonymous namespace

//===----------------------------------------------------------------------===//
// rewriteSwitchEnum
//===----------------------------------------------------------------------===//

// Rewrite switch_enum to switch_enum_addr. This requires rewriting all the
// associated block arguments are removed.
void AddressOnlyUseRewriter::visitSwitchEnumInst(SwitchEnumInst *SEI) {
  SILValue enumVal = SEI->getOperand();
  assert(currOper->get() == enumVal);
  
  SILValue enumAddr = addrMat.materializeAddress(enumVal);

  // TODO: We should be able to avoid copying the case elements into a vector.
  SmallVector<std::pair<EnumElementDecl *, SILBasicBlock *>, 8> cases;
  SmallVector<ProfileCounter, 8> caseCounters;

  // Collect switch cases for rewriting and remove block arguments.
  for (unsigned caseIdx : range(SEI->getNumCases())) {
    EnumElementDecl *caseDecl;
    SILBasicBlock *caseBB;
    auto caseRecord = SEI->getCase(caseIdx);
    std::tie(caseDecl, caseBB) = caseRecord;

    cases.push_back(caseRecord);
    caseCounters.push_back(SEI->getCaseCount(caseIdx));

    // Nothing to do for unused case payloads.
    if (caseBB->getPhiArguments().size() == 0)
      continue;

    assert(caseBB->getPhiArguments().size() == 1);
    SILPhiArgument *caseArg = caseBB->getPhiArguments()[0];

    assert(caseDecl->hasAssociatedValues() && "caseBB has a payload argument");

    SILBuilder argBuilder = pass.getBuilder(caseBB->begin());

    if (caseArg->getType().isAddressOnly(*pass.F)) {
      // Replace the block arg with a dummy. As a rule, don't delete anything
      // that may define an opaque value until dead code elimination. Also,
      // conceivably we could have a cycle of switch_enum during rewriting.
      LoadInst *loadArg = argBuilder.createLoad(
          SEI->getLoc(),
          SILUndef::get(caseArg->getType().getAddressType(), *pass.F),
          LoadOwnershipQualifier::Unqualified);

      caseArg->replaceAllUsesWith(loadArg);

      pass.valueStorageMap.replaceValue(caseArg, loadArg);
    } else {
      // Rewrite any non-opaque cases now since we don't visit their defs.
      auto *UTE = argBuilder.createUncheckedTakeEnumDataAddr(
        SEI->getLoc(), enumAddr, caseDecl);
      auto *loadElt = argBuilder.createLoad(
        SEI->getLoc(), UTE, LoadOwnershipQualifier::Unqualified);
      caseArg->replaceAllUsesWith(loadElt);
    }
    caseBB->eraseArgument(0);
  }
  auto *defaultBB = SEI->hasDefault() ? SEI->getDefaultBB() : nullptr;
  auto defaultCounter =
      SEI->hasDefault() ? SEI->getDefaultCount() : ProfileCounter();

  SILBuilderWithScope B(SEI);
  B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
  B.createSwitchEnumAddr(SEI->getLoc(), enumAddr, defaultBB, cases,
                         ArrayRef<ProfileCounter>(caseCounters),
                         defaultCounter);
  SEI->eraseFromParent();
}

//===----------------------------------------------------------------------===//
// AddressOnlyDefRewriter - rewrite opaque value definitions.
//===----------------------------------------------------------------------===//

namespace {
class AddressOnlyDefRewriter
    : SILInstructionVisitor<AddressOnlyDefRewriter> {
  friend SILVisitorBase<AddressOnlyDefRewriter>;
  friend SILInstructionVisitor<AddressOnlyDefRewriter>;

  AddressLoweringState &pass;

  SILBuilder B;
  AddressMaterialization addrMat;

  // This storage pointer is set before visiting a def for convenience, but if
  // the valueStorageMap is mutated it is no longer valid.
  ValueStorage *storage = nullptr;

public:
  explicit AddressOnlyDefRewriter(AddressLoweringState &pass)
      : pass(pass), B(*pass.F), addrMat(pass, B) {
    B.setSILConventions(
       SILModuleConventions::getLoweredAddressConventions(pass.F->getModule()));
  }

  void visitInst(SILInstruction *inst) { visit(inst); }

  // Set the storage address for am opaque block arg and mark it rewritten.
  // Opaque block args are only removed after all instructions are rewritten.
  void rewriteBBArg(SILPhiArgument *bbArg) {
    B.setInsertionPoint(bbArg->getParent()->begin());
    // TODO: No debug scope for arguments?

    auto &storage = pass.valueStorageMap.getStorage(bbArg);
    assert(!storage.isRewritten);
    storage.storageAddress = addrMat.materializeAddress(bbArg);
    storage.markRewritten();
  }

protected:
  void beforeVisit(SILInstruction *I) {
    // This cast succeeds beecause only specific instructions get added to
    // the value storage map.
    storage = &pass.valueStorageMap.getStorage(cast<SingleValueInstruction>(I));

    LLVM_DEBUG(llvm::dbgs() << "REWRITE DEF "; I->dump());
    if (storage->storageAddress)
      LLVM_DEBUG(llvm::dbgs() << "  STORAGE "; storage->storageAddress->dump());

    B.setInsertionPoint(I);
    B.setCurrentDebugScope(I->getDebugScope());
  }

  void visitSILInstruction(SILInstruction *I) {
    LLVM_DEBUG(I->dump());
    llvm_unreachable("Unimplemented opaque def.");
  }

  // Call returning a single opaque value.
  void visitApplyInst(ApplyInst *applyInst) {
    assert(isa<SingleValueInstruction>(applyInst) &&
           "beforeVisit assumes that ApplyInst is an SVI");
    assert(!storage->isRewritten);
    // Completely rewrite the apply instruction, handling any remaining
    // (loadable) indirect parameters, allocating memory for indirect
    // results, and generating a new apply instruction.
    ParameterRewriter(applyInst, pass).rewriteParameters();
    ApplyRewriter(applyInst, pass).convertApplyWithIndirectResults();
  }

  // Copy into an opaque value.
  void visitCopyValueInst(CopyValueInst *copyInst) {
    assert(storage->isRewritten);
  }

  // Define an opaque enum value.
  void visitEnumInst(EnumInst *enumInst) {
    SILValue enumAddr;
    if (enumInst->hasOperand()) {
      addrMat.initializeOperandMem(&enumInst->getOperandRef());

      assert(storage->storageAddress);
      enumAddr = storage->storageAddress;
    } else
      enumAddr = addrMat.materializeAddress(enumInst);

    B.createInjectEnumAddr(enumInst->getLoc(), enumAddr,
                           enumInst->getElement());

    storage->markRewritten();
  }

  // Define an existential.
  void visitInitExistentialValueInst(
      InitExistentialValueInst *initExistentialValue) {

    // Initialize memory for the operand which may be opaque or loadable.
    addrMat.initializeOperandMem(&initExistentialValue->getOperandRef());

    assert(storage->storageAddress);
    storage->markRewritten();
  }

  // Project an opaque value out of an existential.
  void visitOpenExistentialValueInst(
      OpenExistentialValueInst *openExistentialValue) {
    // Rewritten on the use-side because storage is inherited from the source.
    assert(storage->isRewritten);
  }

  // Project an opaque value out of a box-type existential.
  void visitOpenExistentialBoxValueInst(
      OpenExistentialBoxValueInst *openExistentialBox) {
    llvm::report_fatal_error("Unimplemented OpenExistentialBoxValue def.");
  }

  // Load an opaque value.
  void visitLoadInst(LoadInst *loadInst) {
    SILValue addr = addrMat.materializeAddress(loadInst);
    IsTake_t isTake;
    if (loadInst->getOwnershipQualifier() == LoadOwnershipQualifier::Take)
      isTake = IsTake;
    else {
      assert(loadInst->getOwnershipQualifier() == LoadOwnershipQualifier::Copy);
      isTake = IsNotTake;
    }
    // Dummy loads are already mapped to their storage address.
    if (addr != loadInst->getOperand()) {
      B.createCopyAddr(loadInst->getLoc(), loadInst->getOperand(), addr, isTake,
                       IsInitialization);
    }
    storage->markRewritten();
  }

  // Define an opaque struct.
  void visitStructInst(StructInst *structInst) {
    ValueStorage &storage = pass.valueStorageMap.getStorage(structInst);

    // For each element, initialize the operand's memory. Some struct elements
    // may be loadable types.
    for (Operand &operand : structInst->getAllOperands())
      addrMat.initializeOperandMem(&operand);

    storage.markRewritten();
  }

  // Define an opaque tuple.
  void visitTupleInst(TupleInst *tupleInst) {
    ValueStorage &storage = pass.valueStorageMap.getStorage(tupleInst);
    // For each element, initialize the operand's memory. Some tuple elements
    // may be loadable types.
    for (Operand &operand : tupleInst->getAllOperands())
      addrMat.initializeOperandMem(&operand);

    storage.markRewritten();
  }

  // Extract an opaque struct member.
  void visitStructExtractInst(StructExtractInst *extractInst) {
    assert(storage->isRewritten);
  }

  // Extract an opaque tuple element.
  void visitTupleExtractInst(TupleExtractInst *extractInst) {
    // If the source is an opaque tuple, as opposed to a call result, then the
    // extract is rewritten on the use-side.
    if (storage->isRewritten)
      return;

    // This must be an indirect result for an apply that has not yet been
    // rewritten. Rewrite the apply.
    SILValue srcVal = extractInst->getOperand();
    if (auto *AI = dyn_cast<ApplyInst>(srcVal)) {
      // Go ahead and rewrite the apply that produces the tuple now since it was
      // effectively just skipped by the def rewriter.
      ApplyRewriter(AI, pass).convertApplyWithIndirectResults();
      assert(storage->isRewritten);
      return;
    }
    assert(isa<SILPhiArgument>(srcVal));
    // Handle multi-result try_apply. Rewriting the apply itself is deferred
    // since it involves a terminator and bb arg.
    storage->storageAddress = addrMat.materializeAddress(extractInst);
    storage->markRewritten();
  }
};
} // end anonymous namespace

// Rewrite applies with indirect paramters or results of loadable types which
// were not visited during opaque value rewritting.
static void rewriteIndirectApply(ApplySite apply, AddressLoweringState &pass) {
  if (!apply.getSubstCalleeType()->hasIndirectFormalResults()) {
    ParameterRewriter(apply, pass).rewriteParameters();
    return;
  }

#ifndef NDEBUG
  bool isRewritten = false;
  if (auto *AI = dyn_cast<ApplyInst>(apply)) {
    visitCallResults(AI, [&](SILValue result) {
      // Normal applies that produce indirect results have already been
      // rewritten when visiting mapped values.
      if (result->getType().isAddressOnly(*pass.F)) {
        assert(pass.valueStorageMap.getStorage(result).isRewritten);
        isRewritten = true;
        return false;
      }
      return true;
    });
  }
  assert(!isRewritten);
#endif
  // If the call has indirect results and wasn't already rewritten, rewrite it
  // now. This also handles try_apply.
  ParameterRewriter(apply, pass).rewriteParameters();
  ApplyRewriter(apply, pass).convertApplyWithIndirectResults();
}

static void rewriteFunction(AddressLoweringState &pass) {
  AddressOnlyDefRewriter defVisitor(pass);
  AddressOnlyUseRewriter useVisitor(pass);

  // For each opaque value, rewrite its users and its defining instruction.
  for (auto &valueStorageI : pass.valueStorageMap) {
    SILValue valueDef = valueStorageI.value;

    // TODO: Multi-Value: ApplyInst
    if (auto *inst = dyn_cast<SingleValueInstruction>(valueDef))
      defVisitor.visitInst(inst);
    else if (isa<SILFunctionArgument>(valueDef)) {
      // Returned values are represented by their rewritten @out arguments
      // because no opaque value exists.
      assert(valueDef->getType().isAddress());
      continue;
    } else
      defVisitor.rewriteBBArg(cast<SILPhiArgument>(valueDef));

    SmallVector<Operand *, 8> uses(valueDef->getUses());
    for (Operand *oper : uses)
      useVisitor.visitOperand(oper);
  }
  // Rewrite any remaining (loadable) indirect parameters or call results that
  // need to be adjusted for the calling convention change.
  //
  // Also rewrite any try_apply with an indirect result and remove the
  // corresponding block argument.
  while (!pass.indirectApplies.empty()) {
    if (auto apply = pass.indirectApplies.pop_back_val())
      rewriteIndirectApply(apply.getValue(), pass);
  }
  // Rewrite this function's return value now that all opaque values within the
  // function are rewritten. This still depends on a valid ValueStorage
  // projection operands.
  if (pass.F->getLoweredFunctionType()->hasIndirectFormalResults())
    ReturnRewriter(pass).rewriteReturns();
}

// Given an array of terminator operand values, produce an array of
// operands with those corresponding to deadArgIndices stripped out.
static void filterDeadArgs(OperandValueArrayRef origArgs,
                           ArrayRef<unsigned> deadArgIndices,
                           SmallVectorImpl<SILValue> &newArgs) {
  auto nextDeadArgI = deadArgIndices.begin();
  for (unsigned i : indices(origArgs)) {
    if (i == *nextDeadArgI) {
      ++nextDeadArgI;
      continue;
    }
    newArgs.push_back(origArgs[i]);
  }
  assert(nextDeadArgI == deadArgIndices.end());
}

// Rewrite a BranchInst omitting dead arguments.
static void removeBranchArgs(BranchInst *BI,
                             SmallVectorImpl<unsigned> &deadArgIndices,
                             AddressLoweringState &pass) {

  llvm::SmallVector<SILValue, 4> branchArgs;
  filterDeadArgs(BI->getArgs(), deadArgIndices, branchArgs);

  SILBuilderWithScope B(BI);
  B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
  B.createBranch(BI->getLoc(), BI->getDestBB(), branchArgs);

  BI->eraseFromParent();
}

// Rewrite a CondBranchInst omitting dead arguments.
static void removeCondBranchArgs(CondBranchInst *CBI, SILBasicBlock *targetBB,
                                 ArrayRef<unsigned> deadArgIndices,
                                 AddressLoweringState &pass) {
  SmallVector<SILValue, 8> trueArgs;
  SmallVector<SILValue, 8> falseArgs;

  if (targetBB == CBI->getTrueBB())
    filterDeadArgs(CBI->getTrueArgs(), deadArgIndices, trueArgs);
  else
    trueArgs.append(CBI->getTrueArgs().begin(), CBI->getTrueArgs().end());

  if (targetBB == CBI->getFalseBB())
    filterDeadArgs(CBI->getFalseArgs(), deadArgIndices, falseArgs);
  else
    falseArgs.append(CBI->getFalseArgs().begin(), CBI->getFalseArgs().end());

  SILBuilderWithScope B(CBI);
  B.setSILConventions(SILModuleConventions::getLoweredAddressConventions());
  B.createCondBranch(CBI->getLoc(), CBI->getCondition(), CBI->getTrueBB(),
                     trueArgs, CBI->getFalseBB(), falseArgs,
                     CBI->getTrueBBCount(), CBI->getFalseBBCount());
  CBI->eraseFromParent();

  llvm_unreachable("Untested"); //!!!
}

// Remove all opaque block arguments. Their inputs have already been substituted
// with Undef.
//
// This can't be done while the valueStorageMap is still in use, because storage
// may be a projection from a branch operand.
//
// FIXME: Generalize to other terminators.
static void removeOpaqueBBArgs(SILBasicBlock *bb, AddressLoweringState &pass) {
  if (bb->isEntry())
    return;

  SmallVector<unsigned, 16> deadArgIndices;
  for (auto *bbArg : bb->getArguments()) {
    if (bbArg->getType().isAddressOnly(*pass.F))
      deadArgIndices.push_back(bbArg->getIndex());
  }
  if (deadArgIndices.empty())
    return;

  // Iterate while modifying the predecessor's terminators.
  for (auto predI = bb->pred_begin(), nextI = predI, predE = bb->pred_end();
       predI != predE; predI = nextI) {
    ++nextI;
    auto *predTerm = (*predI)->getTerminator();
    switch (predTerm->getTermKind()) {
    default:
      llvm_unreachable("Unexpected block terminator.");

    case TermKind::BranchInst:
      removeBranchArgs(cast<BranchInst>(predTerm), deadArgIndices, pass);
      break;
    case TermKind::CondBranchInst:
      removeCondBranchArgs(cast<CondBranchInst>(predTerm), bb, deadArgIndices,
                           pass);
      break;
    case TermKind::SwitchEnumInst:
      llvm_unreachable("switch_enum arguments are removed when the terminator "
                       "is rewritten.");
    case TermKind::TryApplyInst:
      llvm_unreachable("try_apply arguments are removed when the terminator "
                       "is rewritten.");
      break;
    }
  }
  for (unsigned deadArgIdx : deadArgIndices)
    bb->eraseArgument(deadArgIdx);
}

// pass.instsToDelete now contains instructions that were explicitly marked
// dead. These already have no users. The rest of the address-only definitions
// will be removed bottom-up by visiting valuestorageMap.
//
// Calls are tricky because sometimes they are values, and sometimes they have
// tuple_extracts representing their values. Either way, they still need to be
// deleted in the correct use-def order.
//
// Address-only block arguments that are tied to terminators (switch_enum,
// try_apply) have already been removed and replace with fake load. The phi-like
// block arguments are removed here after all other instructions.
static void deleteRewrittenInstructions(AddressLoweringState &pass) {
  // Add the rest of the instructions to the dead list in post order.
  // FIXME: make sure we cleaned up address-only BB arguments.
  for (auto &valueStorageI : reversed(pass.valueStorageMap)) {
    SILValue val = valueStorageI.value;
    // If the storage was explicitly erased, skip it. (e.g. replaced bb args).
    if (!pass.valueStorageMap.contains(val))
      continue;

#ifndef NDEBUG
    auto &storage = pass.valueStorageMap.getStorage(val);
    // Returned tuples and multi-result calls are currently values without
    // storage.  Everything else must have been rewritten.
    assert(storage.isRewritten);
#endif
    // TODO: Multi-Value: ApplyInst
    auto *deadInst = dyn_cast<SingleValueInstruction>(val);
    if (!deadInst) {
      // Skip arguments. Function arguments are already address types and are
      // not dead. Dead block args are removed later to avoid rewriting all
      // predecessors terminators multiple times.
      assert(isa<SILArgument>(val));
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "DEAD "; deadInst->dump());
#ifndef NDEBUG
    for (auto result : deadInst->getResults())
      for (Operand *operand : result->getUses())
        assert(pass.instsToDelete.count(operand->getUser()));
#endif
    pass.instsToDelete.insert(deadInst);
#if 0 //!!!
    // TODO: Multi-Value. Insert non-value calls into the dead instruction list
    // if all of their results have been deleted.
    auto *extract = dyn_cast<TupleExtractInst>(deadInst);
    if (extract) {
      auto *applyInst = dyn_cast<ApplyInst>(extract->getOperand());
      if (applyInst) {
        if (all_of(applyInst->getUses(),
                   [&](Operand *op) -> bool {
                     return pass.instsToDelete.count(op->getUser());
                   })) {
          DEBUG(llvm::dbgs() << "DEAD "; applyInst->dump());
          pass.instsToDelete.insert(applyInst);
          //!!!assert(pass.callsToDelete.remove(applyInst));
        }
      }
    }
#endif
  }
  assert(pass.callsToDelete.empty());
  
  pass.valueStorageMap.clear();

  // Delete instructions in postorder
  recursivelyDeleteTriviallyDeadInstructions(pass.instsToDelete.takeVector(),
                                             true);

  // Remove block args after removing all instructions that may use them.
  for (auto &bb : *pass.F)
    removeOpaqueBBArgs(&bb, pass);
}

//===----------------------------------------------------------------------===//
// AddressLowering: Top-Level Module Transform.
//===----------------------------------------------------------------------===//

namespace {
// Note: the only reason this is not a FunctionTransform is to change the SIL
// stage for all functions at once.
class AddressLowering : public SILModuleTransform {
  /// The entry point to this module transformation.
  void run() override;

  void runOnFunction(SILFunction *F);
};
} // end anonymous namespace

void AddressLowering::runOnFunction(SILFunction *F) {
  if (!F->isDefinition())
    return;

  PrettyStackTraceSILFunction FuncScope("address-lowering", F);

  LLVM_DEBUG(llvm::dbgs() << "Address Lowering: " << F->getName() << "\n");

  auto *DA = PM->getAnalysis<DominanceAnalysis>();

  AddressLoweringState pass(F, DA->get(F));

  // Rewrite function args and populate pass.valueStorageMap.
  prepareValueStorage(pass);

  // Insert alloc_stack/dealloc_stack.
  //
  // WARNING: This may split critical edges in ValueLifetimeAnalysis.
  OpaqueStorageAllocation allocator(pass);
  allocator.allocateOpaqueStorage();

  LLVM_DEBUG(llvm::dbgs() << "Finished allocating storage.\n"; F->dump();
             pass.valueStorageMap.dump());

  // Rewrite instructions with address-only operands or results.
  rewriteFunction(pass);

  deleteRewrittenInstructions(pass);

  StackNesting().correctStackNesting(F);

  // The CFG may change because of criticalEdge splitting during
  // createStackAllocation or StackNesting.
  invalidateAnalysis(F, SILAnalysis::InvalidationKind::BranchesAndInstructions);
}

/// The entry point to this module transformation.
void AddressLowering::run() {
  if (getModule()->getOptions().EnableSILOpaqueValues) {
    for (auto &F : *getModule())
      runOnFunction(&F);
  }
  // Set the SIL state before the PassManager has a chance to run
  // verification.
  getModule()->setStage(SILStage::Lowered);
}

SILTransform *swift::createAddressLowering() { return new AddressLowering(); }
