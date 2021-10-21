//===--- OwnershipOptUtils.cpp --------------------------------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2020 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//
///
/// \file
///
/// Ownership Utilities that rely on SILOptimizer functionality.
///
//===----------------------------------------------------------------------===//

#include "swift/SILOptimizer/Utils/OwnershipOptUtils.h"

#include "swift/Basic/Defer.h"
#include "swift/SIL/BasicBlockUtils.h"
#include "swift/SIL/InstructionUtils.h"
#include "swift/SIL/LinearLifetimeChecker.h"
#include "swift/SIL/MemAccessUtils.h"
#include "swift/SIL/OwnershipUtils.h"
#include "swift/SIL/Projection.h"
#include "swift/SIL/ScopedAddressUtils.h"
#include "swift/SIL/SILArgument.h"
#include "swift/SIL/SILBuilder.h"
#include "swift/SIL/SILInstruction.h"
#include "swift/SILOptimizer/Utils/CFGOptUtils.h"
#include "swift/SILOptimizer/Utils/InstructionDeleter.h"
#include "swift/SILOptimizer/Utils/ValueLifetime.h"

using namespace swift;

//===----------------------------------------------------------------------===//
//                            Lifetime Completion
//===----------------------------------------------------------------------===//

// When computing lifetime for the initial value, %v1, transitively include all
// dominated reborrows. %phi1 in this example:
//
//     %v1 = ...
//     cond_br bb1, bb2
//   bb1:
//     %b1 = begin_borrow %v1
//     br bb3(%b1)
//   bb2:
//     %b2 = begin_borrow %v1
//     br bb3(%b2)
//   bb3(%phi1):
//     %u1 = %phi1
//     end_borrow %phi1
//     %k1 = destroy_value %v1 // must be below end_borrow %phi1
//
// In the following adjacent reborrow example, when computing lifetime for an
// the a phi (%phi2) transitively include all "adjacent reborrows" (%phi1):
//
//   bb1:
//     %v1 = ...
//     %b1 = begin_borrow %v1
//     br bb3(%b1, %v1)
//   bb2:
//     %v2 = ...
//     %b2 = begin_borrow %v2
//     br bb3(%b2, %v2)
//   bb3(%phi1, %phi2):
//     %u1 = %phi1
//     end_borrow %phi1
//     %k1 = destroy_value %phi1
//
void OSSALifetimeCompletion::recursivelyComputeLiveness(
    SILValue def, PrunedLiveness &liveness) {
  if ((SILPhiArgument *arg = dyn_cast<SILPhiArgument>(value)) && arg->isPhi()) {
    visitAdjacentReborrowsOfPhi(arg, [&](SILPhiArgument *reborrow) {
      recursivelyComputeLiveness(reborow);
    });
  }
  for (Operand *use : def->getUses()) {
    switch (use->getOperandOwnership()) {
    default:
      updateForUse(use->getUser(), use->isLifetimeEnding());
      break;
    case OperandOwnership::NonUse:
      break;
    case OperandOwnership::Borrow:
    case OperandOwnership::Reborrow: {
      BorrowingOperand borrowOper(use);
      auto borrowedValue = borrowOper.getBorrowIntroducingUserResult();

      // Recursively complete the borrow lifetime
      completeOSSLifetime(borrowedValue.value);

      // Visit scope-ending uses only after completing the borrow lifetime
      //
      //!!! what if an owned value dominates a reborrow?
      //
      //!!! this doesn't make sense for reborrows when there is an adjacent
      //!!! owned phi.
      borrowOper.visitScopeEndingUses([&](Operand *end) {
        updateForUse(end->getUser(), /*lifetimeEnding*/ false);
      });
    }
    }
  }
}

/// Visit all points on the value's lifetime boundary that do not end its
/// lifetime.
///
/// \p value must be an owned value or must introduce a local borrow scope.
///
/// If \p value is a dead instruction, then its definining instruction is
/// visited as the last use. If \p value is a dead argument, then it's basic
/// block is visited as the boundary edge.
static void
visitNonLifetimeEndingBoundary(SILValue value,
                               function_ref<void(SILBasicBlock *)> visitEdge,
                               function_ref<void(SILInstruction *)> visitUser) {

  if (value->getOwnershipKind() != OwnershipKind::Owned) {
    BorrowedValue borrowedValue(value);
    assert(borrowedValue && borrowedValue.isLocalScope());
  }

  SmallVector<SILBasicBlock *> discoveredBlocks;
  PrunedLiveness liveness(&discoveredBlocks);
  PrunedLivenessBoundary boundary;

  liveness.initializeDefBlock(value->getParentBlock());
  recursivelyComputeLiveness(value, liveness);

  boundary.compute(liveness);
  if (boundary.empty()) {
    auto *def = value->getDefiningInstruction();
    if (def)
      visitUser(def);
    else
      visitEdge(cast<SILArgument>(value)->getParent());

    return;
  }
  for (SILBasicBlock *edge : boundary.boundaryEdges) {
    visitEdge(edge);
  }
  for (SILInstruction *lastUser : boundary.lastUsers) {
    if (liveness.isInterestingUser(lastUser)
        != PrunedLiveness::LifetimeEndingUse) {
      visitUser(lastUser);
    }
  }
}

static SILInstruction *endOSSALifetime(SILValue value, SILBuilder &builder) {
  auto loc =
    RegularLocation::getAutoGeneratedLocation(builder.getInsertionPointLoc());
  if (value->getOwnershipKind() == OwnershipKind::Owned) {
    return builder.createDestroyValue(loc, value);
  }
  return builder.createEndBorrow(loc, value);
}

//!!! FIXME: must call visitAdjacentReborrowsOfPhi
bool swift::completeNonLexicalLifetime(SILValue value) {
  assert(!value->isLexical());

  //!!!
  // bool trace = value->getFunction()->hasName("$ss18_CocoaArrayWrapperV13_copyContents12initializings16IndexingIteratorVyABG_SitSryyXlG_tF");
  
  bool changed = false;

  auto endLifetimeAtEdge = [&](SILBasicBlock *edge) {
    SILBuilderWithScope builder(edge->begin());
    auto *i = endOSSALifetime(value, builder);
    if (trace)
      i->dump();
    changed = true;
  };
  auto endLifetimeAtUser = [&](SILInstruction *user) {
    SILBuilderWithScope::insertAfter(user, [value, trace](SILBuilder &builder) {
      auto *i = endOSSALifetime(value, builder);
      if (trace)
        i->dump();
    });
    changed = true;
  };
  visitNonLifetimeEndingBoundary(value, endLifetimeAtEdge, endLifetimeAtUser);

  //!!! trace = false;
  return changed;
}

/// End the lifetime of \p value at unreachable instructions.
///
/// Returns true if any new instructions were created to complete the lifetime.
bool swift::completeLexicalLifetime(SILValue value) {
  assert(value->isLexical());

  BasicBlockWorklist deadEndBlocks(value->getFunction());
  visitNonLifetimeEndingBoundary(
      value, [&](SILBasicBlock *edge) { deadEndBlocks.push(edge); },
      [&](SILInstruction *user) { deadEndBlocks.push(user->getParent()); });

  // Forward CFG walk from the non-lifetime-ending boundary to the unreachable
  // instructions.
  bool changed = false;
  while (auto *block = deadEndBlocks.pop()) {
    if (block->succ_empty()) {
      auto *unreachable = cast<UnreachableInst>(block->getTerminator());
      SILBuilderWithScope builder(unreachable);
      endOSSALifetime(value, builder);
      changed = true;
    }
    for (auto *successor : block->getSuccessorBlocks()) {
      deadEndBlocks.push(successor);
    }
  }
  return changed;
}

// TODO: create a fast check for 'mayEndLifetime(SILInstruction *)'. Verify that
// it returns true for every instruction that has a lifetime-ending operand.
void UnreachableLifetimeCompletion::visitUnreachableInst(
    SILInstruction *instruction) {
  auto *block = instruction->getParent();
  bool inReachableBlock = !unreachableBlocks.contains(block);
  // If this instruction's block is already marked unreachable, and
  // updatingLifetimes is not yet set, then this instruction will be visited
  // again later when propagating unreachable blocks.
  if (!inReachableBlock && !updatingLifetimes)
    return;

  for (Operand &operand : instruction->getAllOperands()) {
    if (!operand.isLifetimeEnding())
      continue;

    SILValue value = operand.get();
    SILBasicBlock *defBlock = value->getParentBlock();
    if (unreachableBlocks.contains(defBlock))
      continue;

    auto *def = value->getDefiningInstruction();
    if (def && unreachableInsts.contains(def))
      continue;

    // The operand's definition is still reachable and its lifetime ends on a
    // newly unreachable path.
    //
    // Note: The arguments of a no-return try_apply may still appear reachable
    // here because the try_apply itself is never visited as unreachable, hence
    // its successor blocks are not marked . But it
    // seems harmless to recompute their lifetimes.

    // Insert this unreachable instruction in unreachableInsts if its parent
    // block is not already marked unreachable.
    if (inReachableBlock) {
      unreachableInsts.insert(instruction);
    }
    incompleteValues.insert(value);

    // Add unreachable successors to the forward traversal worklist.
    if (auto *term = dyn_cast<TermInst>(instruction)) {
      for (auto *succBlock : term->getSuccessorBlocks()) {
        if (llvm::all_of(succBlock->getPredecessorBlocks(),
                         [&](SILBasicBlock *predBlock) {
                           if (predBlock == block)
                             return true;

                           return unreachableBlocks.contains(predBlock);
                         })) {
          unreachableBlocks.insert(succBlock);
        }
      }
    }
  }
}

//!!! we don't know the order of the incompleteValues, so completeOSSLifetime
//!!! needs to be changed to recursively complete borrowed values.
bool UnreachableLifetimeCompletion::completeLifetimes() {
  assert(!updatingLifetimes && "don't call this more than once");
  updatingLifetimes = true;

  // Now that all unreachable terminator instructions have been visited,
  // propagate unreachable blocks.
  for (auto blockIt = unreachableBlocks.begin();
       blockIt != unreachableBlocks.end(); ++blockIt) {
    auto *block = *blockIt;
    for (auto &instruction : *block) {
      visitUnreachableInst(&instruction);
    }
  }

  bool changed = false;
  for (auto value : incompleteValues) {
    if (completeOSSALifetime(value) == LifetimeCompletion::WasCompleted) {
      changed = true;
    }
  }
  return changed;
}

//===----------------------------------------------------------------------===//
//                   Basic scope and lifetime extension API
//===----------------------------------------------------------------------===//

void swift::extendOwnedLifetime(SILValue ownedValue,
                                PrunedLivenessBoundary &lifetimeBoundary,
                                InstructionDeleter &deleter) {
  // Gather the current set of destroy_values, which may die.
  SmallSetVector<Operand *, 4> extraConsumes;
  SmallPtrSet<SILInstruction *, 4> extraConsumers;
  for (Operand *use : ownedValue->getUses()) {
    if (use->isConsuming()) {
      extraConsumes.insert(use);
      extraConsumers.insert(use->getUser());
    }
  }
  // Insert or reuse a destroy_value at all last users.
  auto createDestroy = [&](SILBuilder &builder) {
    auto loc = RegularLocation::getAutoGeneratedLocation(
        builder.getInsertionPointLoc());
    auto *destroy = builder.createDestroyValue(loc, ownedValue);
    deleter.getCallbacks().createdNewInst(destroy);
  };
  for (SILInstruction *lastUser : lifetimeBoundary.lastUsers) {
    if (extraConsumers.erase(lastUser))
      continue;

    SILBuilderWithScope::insertAfter(lastUser, createDestroy);
  }
  // Insert a destroy_value at all boundary edges.
  for (SILBasicBlock *edge : lifetimeBoundary.boundaryEdges) {
    SILBuilderWithScope builder(edge->begin());
    createDestroy(builder);
  }
  // Delete or copy extra consumes.
  for (auto *consume : extraConsumes) {
    auto *consumer = consume->getUser();
    if (!extraConsumers.count(consumer))
      continue;

    if (isa<DestroyValueInst>(consumer)) {
      deleter.forceDelete(consumer);
      continue;
    }
    auto loc = RegularLocation::getAutoGeneratedLocation(consumer->getLoc());
    auto *copy = SILBuilderWithScope(consumer).createCopyValue(loc, ownedValue);
    consume->set(copy);
    deleter.getCallbacks().createdNewInst(copy);
  }
}

void swift::extendLocalBorrow(BeginBorrowInst *beginBorrow,
                              PrunedLivenessBoundary &guaranteedBoundary,
                              InstructionDeleter &deleter) {
  // Gather the current set of end_borrows, which may die.
  SmallVector<EndBorrowInst *, 4> endBorrows;
  SmallPtrSet<EndBorrowInst *, 4> deadEndBorrows;
  for (Operand *use : beginBorrow->getUses()) {
    if (auto *endBorrow = dyn_cast<EndBorrowInst>(use->getUser())) {
      endBorrows.push_back(endBorrow);
      deadEndBorrows.insert(endBorrow);
      continue;
    }
    assert(use->getOperandOwnership() != OperandOwnership::EndBorrow
           && use->getOperandOwnership() != OperandOwnership::Reborrow
           && "expecting a purely local borrow scope");
  }
  // Insert or reuse an end_borrow at all last users.
  auto createEndBorrow = [&](SILBuilder &builder) {
    auto loc = RegularLocation::getAutoGeneratedLocation(
        builder.getInsertionPointLoc());
    auto *endBorrow = builder.createEndBorrow(loc, beginBorrow);
    deleter.getCallbacks().createdNewInst(endBorrow);
  };
  for (SILInstruction *lastUser : guaranteedBoundary.lastUsers) {
    if (auto *endBorrow = dyn_cast<EndBorrowInst>(lastUser)) {
      if (deadEndBorrows.erase(endBorrow))
        continue;
    }
    SILBuilderWithScope::insertAfter(lastUser, createEndBorrow);
  }
  // Insert an end_borrow at all boundary edges.
  for (SILBasicBlock *edge : guaranteedBoundary.boundaryEdges) {
    SILBuilderWithScope builder(edge->begin());
    createEndBorrow(builder);
  }
  // Delete dead end_borrows.
  for (auto *endBorrow : endBorrows) {
    if (deadEndBorrows.count(endBorrow))
      deleter.forceDelete(endBorrow);
  }
}

bool swift::computeGuaranteedBoundary(SILValue value,
                                      PrunedLivenessBoundary &boundary) {
  assert(value->getOwnershipKind() == OwnershipKind::Guaranteed);

  // Place end_borrows that cover the load_borrow uses. It is not necessary to
  // cover the outer borrow scope of the extract's operand. If a lexical
  // borrow scope exists for the outer value, which is now in memory, then
  // its alloc_stack will be marked lexical, and the in-memory values will be
  // kept alive until the end of the outer scope.
  SmallVector<Operand *, 4> usePoints;
  bool noEscape = findInnerTransitiveGuaranteedUses(value, &usePoints);

  SmallVector<SILBasicBlock *, 4> discoveredBlocks;
  SSAPrunedLiveness liveness(&discoveredBlocks);
  liveness.initializeDef(value);
  for (auto *use : usePoints) {
    assert(!use->isLifetimeEnding());
    liveness.updateForUse(use->getUser(), /*lifetimeEnding*/ false);
  }
  liveness.computeBoundary(boundary);

  return noEscape;
}

//===----------------------------------------------------------------------===//
//                        GuaranteedOwnershipExtension
//===----------------------------------------------------------------------===//

// Can the OSSA ownership of the \p parentAddress cover all uses of the \p
// childAddress?
GuaranteedOwnershipExtension::Status
GuaranteedOwnershipExtension::checkAddressOwnership(SILValue parentAddress,
                                                    SILValue childAddress) {
  AddressOwnership addressOwnership(parentAddress);
  if (!addressOwnership.hasLocalOwnershipLifetime()) {
    // Indirect Arg, Stack, Global, Unidentified, Yield
    // (these have no reference lifetime to extend).
    return Valid;
  }
  SmallVector<Operand *, 8> childUses;
  if (findTransitiveUsesForAddress(childAddress, &childUses)
      != AddressUseKind::NonEscaping) {
    return Invalid; // pointer escape, so we don't know required lifetime
  }
  SILValue referenceRoot = addressOwnership.getOwnershipReferenceRoot();
  assert(referenceRoot && "expect to find a reference to Box/Class/Tail");

  if (referenceRoot->getOwnershipKind() != OwnershipKind::Guaranteed) {
    // Note: Addresses are normally guarded by a borrow scope. But eventually,
    // an address base can be considered an implicit borrow. This current
    // handles project_box, which is not in a borrow scope (it is sadly modeled
    // as a PointerEscape). But we can treat project_box like an implicit borrow
    // in this context.
    return checkLifetimeExtension(referenceRoot, childUses);
  }
  BorrowedValue parentBorrow(referenceRoot);
  if (!parentBorrow)
    return Invalid; // unexpected borrow introducer

  return checkBorrowExtension(parentBorrow, childUses);
}

// Can the OSSA scope of \p borrow cover all \p newUses?
GuaranteedOwnershipExtension::Status
GuaranteedOwnershipExtension::checkBorrowExtension(
    BorrowedValue borrow, ArrayRef<Operand *> newUses) {

  if (!borrow.isLocalScope())
    return Valid; // arguments have whole-function ownership

  assert(guaranteedLiveness.empty());
  borrow.computeTransitiveLiveness(guaranteedLiveness);

  if (guaranteedLiveness.areUsesWithinBoundary(newUses))
    return Valid; // reuse the borrow scope as-is

  beginBorrow = dyn_cast<BeginBorrowInst>(borrow.value);
  if (!beginBorrow)
    return Invalid; // cannot extend load_borrow without memory lifetime

  // Extend liveness to the new uses before returning any status that leads to
  // transformation.
  for (Operand *use : newUses) {
    guaranteedLiveness.updateForUse(use->getUser(), /*lifetimeEnding*/ false);
  }
  // It is unusual to have a borrow scope that (a) dominates the new uses, (b)
  // does not already cover the new uses, but (c) already has a reborrow for
  // some other reason.
  if (borrow.hasReborrow())
    return Invalid; // Can only extend a local scope up to dominated uses

  auto status = checkLifetimeExtension(beginBorrow->getOperand(), newUses);
  if (status == Valid) {
    // The owned lifetime is adequate, but the borrow scope must be extended.
    return ExtendBorrow;
  }
  return status;
}

GuaranteedOwnershipExtension::Status
GuaranteedOwnershipExtension::checkLifetimeExtension(
    SILValue ownedValue, ArrayRef<Operand *> newUses) {
  assert(ownedLifetime.empty());

  auto ownershipKind = ownedValue->getOwnershipKind();
  if (ownershipKind == OwnershipKind::None)
    return Valid;

  // If the ownedValue is not owned, give up for simplicity. We expect nested
  // borrows to be removed.
  if (ownershipKind != OwnershipKind::Owned)
    return Invalid;

  ownedLifetime.initializeDef(ownedValue);
  for (Operand *use : ownedValue->getUses()) {
    auto *user = use->getUser();
    if (use->isConsuming()) {
      ownedLifetime.updateForUse(user, true);
      ownedConsumeBlocks.push_back(user->getParent());
    }
  }
  if (ownedLifetime.areUsesWithinBoundary(newUses))
    return Valid;

  return ExtendLifetime; // Can't cover newUses without destroy sinking.
}

void GuaranteedOwnershipExtension::transform(Status status) {
  switch (status) {
  case Invalid:
  case Valid:
    return;
  case ExtendBorrow: {
    PrunedLivenessBoundary guaranteedBoundary;
    guaranteedLiveness.computeBoundary(guaranteedBoundary, ownedConsumeBlocks);
    extendLocalBorrow(beginBorrow, guaranteedBoundary, deleter);
    break;
  }
  case ExtendLifetime: {
    ownedLifetime.extendAcrossLiveness(guaranteedLiveness);
    PrunedLivenessBoundary ownedBoundary;
    ownedLifetime.computeBoundary(ownedBoundary, ownedConsumeBlocks);
    extendOwnedLifetime(beginBorrow->getOperand(), ownedBoundary, deleter);
    PrunedLivenessBoundary guaranteedBoundary;
    guaranteedLiveness.computeBoundary(guaranteedBoundary, ownedConsumeBlocks);
    extendLocalBorrow(beginBorrow, guaranteedBoundary, deleter);
    break;
  }
  }
}

//===----------------------------------------------------------------------===//
//                          Utility Helper Functions
//===----------------------------------------------------------------------===//

static void cleanupOperandsBeforeDeletion(SILInstruction *oldValue,
                                          InstModCallbacks &callbacks) {
  SILBuilderWithScope builder(oldValue);
  for (auto &op : oldValue->getAllOperands()) {
    if (!op.isLifetimeEnding()) {
      continue;
    }

    switch (op.get()->getOwnershipKind()) {
    case OwnershipKind::Any:
      llvm_unreachable("Invalid ownership for value");
    case OwnershipKind::Owned: {
      auto *dvi = builder.createDestroyValue(oldValue->getLoc(), op.get());
      callbacks.createdNewInst(dvi);
      continue;
    }
    case OwnershipKind::Guaranteed: {
      // Should only happen once we model destructures as true reborrows.
      auto *ebi = builder.createEndBorrow(oldValue->getLoc(), op.get());
      callbacks.createdNewInst(ebi);
      continue;
    }
    case OwnershipKind::None:
      continue;
    case OwnershipKind::Unowned:
      llvm_unreachable("Unowned object can never be consumed?!");
    }
    llvm_unreachable("Covered switch isn't covered");
  }
}

//===----------------------------------------------------------------------===//
//                      Ownership RAUW Helper Functions
//===----------------------------------------------------------------------===//

// Determine whether it is valid to replace \p oldValue with \p newValue by
// directly checking ownership requirements. This does not determine whether the
// scope of the newValue can be fully extended.
bool OwnershipRAUWHelper::hasValidRAUWOwnership(SILValue oldValue,
                                                SILValue newValue,
                                                ArrayRef<Operand *> oldUses) {
  auto newOwnershipKind = newValue->getOwnershipKind();

  // If the either value is lexical, replacing its uses may result in
  // shortening or lengthening its lifetime in ways that don't respect lexical
  // scope and deinit barriers.
  //
  // Specifically, we have the following cases:
  //
  // +--------+--------+
  // |oldValue|newValue|
  // +--------+--------+
  // |  not   |  not   | legal
  // +--------+--------+
  // |lexical |  not   | illegal
  // +--------+--------+
  // |   *    |lexical | legal so long as it doesn't extend newValue's lifetime
  // +--------+--------+
  if ((oldValue->isLexical() && !newValue->isLexical()) ||
      (newValue->isLexical() &&
       !areUsesWithinLexicalValueLifetime(newValue, oldUses)))
    return false;

  // If our new kind is ValueOwnershipKind::None, then we are fine. We
  // trivially support that. This check also ensures that we can always
  // replace any value with a ValueOwnershipKind::None value.
  if (newOwnershipKind == OwnershipKind::None)
    return true;

  // If our old ownership kind is ValueOwnershipKind::None and our new kind is
  // not, we may need to do more work that has not been implemented yet. So
  // bail.
  //
  // Due to our requirement that types line up, this can only occur given a
  // non-trivial typed value with None ownership. This can only happen when
  // oldValue is a trivial payloaded or no-payload non-trivially typed
  // enum. That doesn't occur that often so we just bail on it today until we
  // implement this functionality.
  if (oldValue->getOwnershipKind() == OwnershipKind::None)
    return false;

  // First check if oldValue is SILUndef. If it is, then we know that:
  //
  // 1. SILUndef (and thus oldValue) must have OwnershipKind::None.
  // 2. newValue is not OwnershipKind::None due to our check above.
  //
  // Thus we know that we would be replacing a value with OwnershipKind::None
  // with a value with non-None ownership. This is a case we don't support, so
  // we can bail now.
  if (isa<SILUndef>(oldValue))
    return false;

  // Ok, we now know that we do not have SILUndef implying that we must be able
  // to get a module from our value since we must have an argument or an
  // instruction.
  auto *m = oldValue->getModule();
  assert(m);

  // If we are in Raw SIL, just bail at this point. We do not support
  // ownership fixups.
  if (m->getStage() == SILStage::Raw)
    return false;

  return true;
}

// Determine whether it is valid to replace \p oldValue with \p newValue and
// extend the lifetime of \p oldValue to cover the new uses.
static bool canFixUpOwnershipForRAUW(SILValue oldValue, SILValue newValue,
                                     OwnershipFixupContext &context) {
  switch (oldValue->getOwnershipKind()) {
  case OwnershipKind::Guaranteed: {
    // Check that the old lifetime can be extended and record the necessary
    // book-keeping in the OwnershipFixupContext.
    context.clear();

    // Check that no transitive uses have a PointerEscape, and record the leaf
    // uses for liveness extension.
    //
    // FIXME: Use findExtendedTransitiveGuaranteedUses and switch the
    // implementation of borrowCopyOverGuaranteedUses to
    // GuaranteedOwnershipExtension.  Utils then, reborrows are considered
    // pointer escapes, causing findTransitiveGuaranteedUses to return false. So
    // they can be ignored.
    auto visitReborrow = [&](Operand *reborrow) {};
    if (!findTransitiveGuaranteedUses(oldValue, context.guaranteedUsePoints,
                                      visitReborrow)) {
      return false;
    }
    return OwnershipRAUWHelper::hasValidRAUWOwnership(
        oldValue, newValue, context.guaranteedUsePoints);
  }
  default: {
    SmallVector<Operand *, 8> ownedUsePoints;
    // If newValue is lexical, find the uses of oldValue so that it can be
    // determined whether the replacement would illegally extend the lifetime
    // of newValue.
    if (newValue->isLexical() &&
        !findUsesOfSimpleValue(oldValue, &ownedUsePoints))
      return false;
    return OwnershipRAUWHelper::hasValidRAUWOwnership(oldValue, newValue,
                                                      ownedUsePoints);
  }
  }
}

bool swift::areUsesWithinLexicalValueLifetime(SILValue value,
                                              ArrayRef<Operand *> uses) {
  assert(value->isLexical());

  // The lexical lifetime of a function argument is the whole body of the
  // function.
  if (isa<SILFunctionArgument>(value))
    return true;

  if (auto borrowedValue = BorrowedValue(value)) {
    auto *function = value->getFunction();
    MultiDefPrunedLiveness liveness(function);
    borrowedValue.computeTransitiveLiveness(liveness);
    DeadEndBlocks deadEndBlocks(function);
    return liveness.areUsesWithinBoundary(uses, &deadEndBlocks);
  }

  return false;
}

//===----------------------------------------------------------------------===//
//                          BorrowedLifetimeExtender
//===----------------------------------------------------------------------===//

/// Model an extended borrow scope, including transitive reborrows. This applies
/// to "local" borrow scopes (begin_borrow, load_borrow, & phi).
///
/// Allow extending the lifetime of an owned value that dominates this borrowed
/// value across that extended borrow scope. This handles uses of reborrows that
/// are not dominated by the owned value by generating phis and copying the
/// borrowed values the reach this borrow scope from non-dominated paths.
///
/// This produces somewhat canonical owned phis, although that isn't a
/// requirement for valid SIL. Given an owned value, a dominated borrowed value,
/// and a reborrow:
///
///     %ownedValue = ...
///     %borrowedValue = ...
///     %reborrow = phi(%borrowedValue, %otherBorrowedValue)
///
/// %otherBorrowedValue will always be copied even if %ownedValue also dominates
/// %otherBorrowedValue, as such:
///
///     %otherCopy = copy_value %borrowedValue
///     %newPhi = phi(%ownedValue, %otherCopy)
///
/// The immediate effect is to produce an unnecessary copy, but it avoids
/// extending %ownedValue's liveness to new paths and hopefully simplifies
/// downstream optimization and debugging. Unnecessary copies could be
/// avoided with simple dominance check if it becomes desirable to do so.
class BorrowedLifetimeExtender {
  BorrowedValue borrowedValue;

  // Owned value currently being extended over borrowedValue.
  SILValue currentOwnedValue;

  InstModCallbacks &callbacks;

  llvm::SmallVector<PhiValue, 4> reborrowedPhis;
  llvm::SmallDenseMap<PhiValue, PhiValue, 4> reborrowedToOwnedPhis;

  /// Check that all reaching operands are handled. This can be removed once the
  /// utility and OSSA representation are stable.
  SWIFT_ASSERT_ONLY_DECL(llvm::SmallDenseSet<PhiOperand, 4> reborrowedOperands);

public:
  /// Precondition: \p borrowedValue must introduce a local borrow scope
  /// (begin_borrow, load_borrow, & phi).
  BorrowedLifetimeExtender(BorrowedValue borrowedValue,
                           InstModCallbacks &callbacks)
      : borrowedValue(borrowedValue), callbacks(callbacks) {
    assert(borrowedValue.isLocalScope() && "expect a valid borrowed value");
  }

  /// Extend \p ownedValue over this extended borrow scope.
  ///
  /// Precondition: \p ownedValue dominates this borrowed value.
  void extendOverBorrowScopeAndConsume(SILValue ownedValue);

protected:
  /// Initially map the reborrowed phi to an invalid value prior to creating the
  /// owned phi.
  void discoverReborrow(PhiValue reborrowedPhi) {
    if (reborrowedToOwnedPhis.try_emplace(reborrowedPhi, PhiValue()).second) {
      reborrowedPhis.push_back(reborrowedPhi);
    }
  }

  /// Remap the reborrowed phi to an valid owned phi after creating it.
  void mapOwnedPhi(PhiValue reborrowedPhi, PhiValue ownedPhi) {
    reborrowedToOwnedPhis[reborrowedPhi] = ownedPhi;
  }

  /// Get the owned value associated with this reborrowed operand, or return an
  /// invalid SILValue indicating that the borrowed lifetime does not reach this
  /// operand.
  SILValue getExtendedOwnedValue(PhiOperand reborrowedOper) {
    // If this operand reborrows the original borrow, then the currentOwned phi
    // reaches it directly.
    SILValue borrowSource = reborrowedOper.getSource();
    if (borrowSource == borrowedValue.value)
      return currentOwnedValue;

    // Check if the borrowed operand's source is already mapped to an owned phi.
    auto reborrowedAndOwnedPhi = reborrowedToOwnedPhis.find(borrowSource);
    if (reborrowedAndOwnedPhi != reborrowedToOwnedPhis.end()) {
      // Return the already-mapped owned phi.
      assert(reborrowedOperands.erase(reborrowedOper));
      return reborrowedAndOwnedPhi->second;
    }
    // The owned value does not reach this reborrowed operand.
    assert(
        !reborrowedOperands.count(reborrowedOper)
        && "reachable borrowed phi operand must be mapped to an owned value");
    return SILValue();
  }

  void analyzeExtendedScope();

  SILValue createCopyAtEdge(PhiOperand reborrowOper);

  void destroyAtScopeEnd(SILValue ownedValue, BorrowedValue pairedBorrow);
};

// Gather all transitive phi-reborrows and check that all the borrowed uses can
// be found with no escapes.
//
// Calls discoverReborrow to populate reborrowedPhis.
void BorrowedLifetimeExtender::analyzeExtendedScope() {
  auto visitReborrow = [&](Operand *endScope) {
    if (auto borrowingOper = BorrowingOperand(endScope)) {
      assert(borrowingOper.isReborrow());

      SWIFT_ASSERT_ONLY(reborrowedOperands.insert(endScope));

      // TODO: if non-phi reborrows are added, handle multiple results.
      discoverReborrow(borrowingOper.getBorrowIntroducingUserResult().value);
    }
    return true;
  };

  bool result = borrowedValue.visitLocalScopeEndingUses(visitReborrow);
  assert(result && "visitReborrow always succeeds, escapes are irrelevant");

  // Note: Iterate in the same manner as findExtendedTransitiveGuaranteedUses(),
  // but using BorrowedLifetimeExtender's own reborrowedPhis.
  for (unsigned idx = 0; idx < reborrowedPhis.size(); ++idx) {
    auto borrowedValue = BorrowedValue(reborrowedPhis[idx]);
    result = borrowedValue.visitLocalScopeEndingUses(visitReborrow);
    assert(result && "visitReborrow always succeeds, escapes are irrelevant");
  }
}

// Insert a copy on this edge. This might not be necessary if the owned
// value dominates this path, but this avoids forcing the owned value to be
// live across new paths.
//
// TODO: consider copying the base of the borrowed value instead of the
// borrowed value directly. It's likely that the copy is used outside of the
// borrow scope, in which case, canonicalizeOSSA will create a copy outside
// the borrow scope anyway. However, we can't be sure that the base is the
// same type.
//
// TODO: consider reusing copies that dominate multiple reborrowed
// operands. However, this requires copying in an earlier block and inserting
// post-dominating destroys, which may be better handled in an ownership phi
// canonicalization pass.
SILValue BorrowedLifetimeExtender::createCopyAtEdge(PhiOperand reborrowOper) {
  auto *branch = reborrowOper.getBranch();
  auto loc = RegularLocation::getAutoGeneratedLocation(branch->getLoc());
  auto *copy = SILBuilderWithScope(branch).createCopyValue(
      loc, reborrowOper.getSource());
  callbacks.createdNewInst(copy);
  return copy;
}

// Destroy \p ownedValue at \p pairedBorrow's scope-ending uses, excluding
// reborrows.
//
// Precondition: ownedValue takes ownership of its value at the same point as
// pairedBorrow. e.g. an owned and guaranteed pair of phis.
void BorrowedLifetimeExtender::destroyAtScopeEnd(SILValue ownedValue,
                                                 BorrowedValue pairedBorrow) {
  pairedBorrow.visitLocalScopeEndingUses([&](Operand *scopeEnd) {
    if (scopeEnd->getOperandOwnership() == OperandOwnership::Reborrow)
      return true;

    auto *endInst = scopeEnd->getUser();
    assert(!isa<TermInst>(endInst) && "branch must be a reborrow");
    auto *destroyPt = &*std::next(endInst->getIterator());
    auto *destroy = SILBuilderWithScope(destroyPt).createDestroyValue(
        destroyPt->getLoc(), ownedValue);
    callbacks.createdNewInst(destroy);
    return true;
  });
}

// Insert and map an owned phi for each reborrowed phi.
//
// For each reborrowed phi, insert a copy on each edge that does not originate
// from the extended borrowedValue.
//
// TODO: If non-phi reborrows are added, they would also need to be
// mapped to their owned counterpart. This means generating new owned
// struct/destructure instructions.
void BorrowedLifetimeExtender::
extendOverBorrowScopeAndConsume(SILValue ownedValue) {
  currentOwnedValue = ownedValue;

  // Populate the reborrowedPhis vector.
  analyzeExtendedScope();

  // Warning: don't use the original callbacks in this function after creating a
  // deleter.
  InstModCallbacks tempCallbacks = callbacks;
  InstructionDeleter deleter(std::move(tempCallbacks));

  // Generate and map the phis with undef operands first, in case of recursion.
  auto undef = SILUndef::get(ownedValue->getType(), *ownedValue->getFunction());
  for (PhiValue reborrowedPhi : reborrowedPhis) {
    auto *phiBlock = reborrowedPhi.phiBlock;
    auto *ownedPhi = phiBlock->createPhiArgument(ownedValue->getType(),
                                                 OwnershipKind::Owned);
    for (auto *predBlock : phiBlock->getPredecessorBlocks()) {
      TermInst *ti = predBlock->getTerminator();
      addNewEdgeValueToBranch(ti, phiBlock, undef, deleter);
    }
    mapOwnedPhi(reborrowedPhi, PhiValue(ownedPhi));
  }
  // Generate copies and set the phi operands.
  for (PhiValue reborrowedPhi : reborrowedPhis) {
    PhiValue ownedPhi = reborrowedToOwnedPhis[reborrowedPhi];
    reborrowedPhi.getValue()->visitIncomingPhiOperands(
        // For each reborrowed operand, get the owned value for that edge,
        // and set the owned phi's operand.
        [&](Operand *reborrowedOper) {
          SILValue ownedVal = getExtendedOwnedValue(reborrowedOper);
          if (!ownedVal) {
            ownedVal = createCopyAtEdge(reborrowedOper);
          }
          TermInst *branch = PhiOperand(reborrowedOper).getBranch();
          branch->getOperandRef(ownedPhi.argIndex).set(ownedVal);
          return true;
        });
  }
  assert(reborrowedOperands.empty() && "not all phi operands are handled");

  // Create destroys at the last uses.
  destroyAtScopeEnd(ownedValue, borrowedValue);
  for (PhiValue reborrowedPhi : reborrowedPhis) {
    PhiValue ownedPhi = reborrowedToOwnedPhis[reborrowedPhi];
    destroyAtScopeEnd(ownedPhi, BorrowedValue(reborrowedPhi));
  }
}

//===----------------------------------------------------------------------===//
//                        Ownership Lifetime Extender
//===----------------------------------------------------------------------===//

namespace {

struct OwnershipLifetimeExtender {
  OwnershipFixupContext &ctx;

  /// Create a new copy of \p value assuming that our caller will clean up the
  /// copy along all paths that go through consuming point. Operationally this
  /// means that the API will insert compensating destroy_value on the copy
  /// along all paths that do not go through consuming point.
  ///
  /// DISCUSSION: If \p consumingPoint is an instruction that forwards \p value,
  /// calling this and then RAUWing with \p value guarantee that \p value will
  /// be consumed by the forwarding instruction's results consuming uses.
  CopyValueInst *createPlusOneCopy(SILValue value,
                                   SILInstruction *consumingPoint);

  /// Create a copy of \p value that covers all of \p range and insert all
  /// needed destroy_values. We assume that no uses in \p range consume \p
  /// value.
  CopyValueInst *createPlusZeroCopy(SILValue value, ArrayRef<Operand *> range) {
    return createPlusZeroCopy<ArrayRef<Operand *>>(value, range);
  }

  /// Create a copy of \p value that covers all of \p range and insert all
  /// needed destroy_values. We assume that all uses in \p range do not consume
  /// \p value.
  ///
  /// We return our copy_value to the user at +0 to show that they do not need
  /// to insert cleanup destroys.
  template <typename RangeTy>
  CopyValueInst *createPlusZeroCopy(SILValue value, const RangeTy &range);

  /// Borrow \p newValue over the extended lifetime of \p borrowedValue.
  BeginBorrowInst *borrowCopyOverScope(SILValue newValue,
                                       BorrowedValue borrowedValue);

  /// Borrow-copy \p newValue over \p guaranteedUses. Copy newValue, borrow the
  /// copy, and extend the lifetime of the borrow-copy over guaranteedUsePoints.
  ///
  /// \p borrowPoint is a value whose definition will be the location of
  /// the new borrow.
  template <typename RangeTy>
  BeginBorrowInst *
  borrowCopyOverGuaranteedUses(SILValue newValue,
                               SILBasicBlock::iterator borrowPoint,
                               RangeTy guaranteedUsePoints);

  template <typename RangeTy>
  BeginBorrowInst *
  borrowCopyOverGuaranteedUsers(SILValue newValue,
                                SILBasicBlock::iterator borrowPoint,
                                RangeTy guaranteedUsers);

  /// Borrow \p newValue over the lifetime of \p guaranteedValue. Return the
  /// new guaranteed value.
  SILValue borrowOverValue(SILValue newValue, SILValue guaranteedValue);

  /// Borrow \p newValue over \p singleGuaranteedUse. Return the
  /// new guaranteed value.
  ///
  /// Precondition: if \p use ends a borrow scope, then \p newValue dominates
  /// the BorrowedValue that begins the scope.
  SILValue borrowOverSingleUse(SILValue newValue,
                               Operand *singleGuaranteedUse);

  SILValue
  borrowOverSingleNonLifetimeEndingUser(SILValue newValue,
                                        SILInstruction *nonLifetimeEndingUser);
};

} // end anonymous namespace

/// Lifetime extend \p value over \p consumingPoint, assuming that \p
/// consumingPoint will consume \p value after the client performs replacement
/// (this implicit destruction on the caller-side makes it a "plus-one"
/// copy). Destroy \p copy on all paths that don't reach \p consumingPoint.
///
/// Precondition: \p value is owned
///
/// Precondition: \p consumingPoint is dominated by \p value
CopyValueInst *
OwnershipLifetimeExtender::createPlusOneCopy(SILValue value,
                                             SILInstruction *consumingPoint) {
  auto *copyPoint = value->getNextInstruction();
  auto loc = copyPoint->getLoc();
  auto *copy = SILBuilderWithScope(copyPoint).createCopyValue(loc, value);

  auto &callbacks = ctx.callbacks;
  callbacks.createdNewInst(copy);

  auto *result = copy;
  findJointPostDominatingSet(
      copy->getParent(), consumingPoint->getParent(),
      // inputBlocksFoundDuringWalk.
      [&](SILBasicBlock *loopBlock) {
        // Since copy dominates consumingPoint, it must be outside the
        // loop. Otherwise backward traversal would have stopped at copyPoint.
        //
        // Create an extra copy when the consumingPoint is inside a loop and the
        // original copy is outside the loop. The new copy will be consumed
        // within the loop in the same block as the consume. The original copy
        // will be destroyed on all paths exiting the loop.
        assert(loopBlock == consumingPoint->getParent());
        auto front = loopBlock->begin();
        SILBuilderWithScope newBuilder(front);
        result = newBuilder.createCopyValue(front->getLoc(), copy);
        callbacks.createdNewInst(result);
      },
      // Leaky blocks that never reach consumingPoint.
      [&](SILBasicBlock *postDomBlock) {
        auto front = postDomBlock->begin();
        SILBuilderWithScope newBuilder(front);
        auto loc = RegularLocation::getAutoGeneratedLocation(front->getLoc());
        auto *dvi = newBuilder.createDestroyValue(loc, copy);
        callbacks.createdNewInst(dvi);
      });
  return result;
}

// A copy_value that we lifetime extend with destroy_value over range. We assume
// all instructions passed into range do not consume value.
template <typename RangeTy>
CopyValueInst *
OwnershipLifetimeExtender::createPlusZeroCopy(SILValue value,
                                              const RangeTy &range) {
  auto *newValInsertPt = value->getDefiningInsertionPoint();
  assert(newValInsertPt);

  CopyValueInst *copy;

  if (!isa<SILArgument>(value)) {
    SILBuilderWithScope::insertAfter(newValInsertPt, [&](SILBuilder &builder) {
      copy = builder.createCopyValue(builder.getInsertionPointLoc(), value);
    });
  } else {
    SILBuilderWithScope builder(newValInsertPt);
    copy = builder.createCopyValue(newValInsertPt->getLoc(), value);
  }

  auto &callbacks = ctx.callbacks;
  callbacks.createdNewInst(copy);

  auto opRange = makeUserRange(range);
  ValueLifetimeAnalysis lifetimeAnalysis(copy, opRange);
  ValueLifetimeBoundary boundary;
  lifetimeAnalysis.computeLifetimeBoundary(boundary);
  boundary.visitInsertionPoints(
      [&](SILBasicBlock::iterator insertPt) {
        SILBuilderWithScope builder(insertPt);
        auto *dvi = builder.createDestroyValue(insertPt->getLoc(), copy);
        callbacks.createdNewInst(dvi);
      });

  return copy;
}

/// Borrow \p newValue over the extended lifetime of \p borrowedValue.
///
/// Precondition: \p newValue dominates borrowedValue.
BeginBorrowInst *
OwnershipLifetimeExtender::borrowCopyOverScope(SILValue newValue,
                                               BorrowedValue borrowedValue) {
  assert(borrowedValue.isLocalScope() && "SILFunctionArg is already handled");

  SILInstruction *borrowPoint = borrowedValue.value->getNextInstruction();
  auto loc = RegularLocation::getAutoGeneratedLocation(borrowPoint->getLoc());
  SILBuilderWithScope builder(borrowPoint);
  auto *copy = builder.createCopyValue(loc, newValue);
  ctx.callbacks.createdNewInst(copy);

  // Extend the new copy's lifetime over borrowedValue's scope and destroy it on
  // all paths through borrowedValue. Since copy is in the same block as
  // borrowedValue, no extra destroys are needed.
  BorrowedLifetimeExtender(borrowedValue, ctx.callbacks)
      .extendOverBorrowScopeAndConsume(copy);

  auto *borrow = builder.createBeginBorrow(loc, copy);
  ctx.callbacks.createdNewInst(borrow);
  return borrow;
}

/// Borrow-copy \p newValue over \p guaranteedUses. Copy newValue, borrow the
/// copy, and extend the lifetime of the borrow-copy over guaranteedUses.
///
/// \p borrowPoint is a the insertion point of the new borrow.
///
/// Precondition: \p newValue dominates \p borrowPoint which dominates \p
/// guaranteedUses
///
/// Precondition: \p guaranteedUses is not empty.
///
/// Precondition: None of \p guaranteedUses are lifetime ending.
template <typename RangeTy>
BeginBorrowInst *OwnershipLifetimeExtender::borrowCopyOverGuaranteedUsers(
    SILValue newValue, SILBasicBlock::iterator borrowPoint,
    RangeTy guaranteedUsers) {

  auto loc = RegularLocation::getAutoGeneratedLocation(borrowPoint->getLoc());

  auto *copy = SILBuilderWithScope(newValue->getNextInstruction())
                   .createCopyValue(loc, newValue);
  auto *borrow = SILBuilderWithScope(borrowPoint).createBeginBorrow(loc, copy);
  ctx.callbacks.createdNewInst(copy);
  ctx.callbacks.createdNewInst(borrow);

  // Create end_borrows at the end of the borrow's lifetime.
  {
    // We don't expect an empty guaranteedUsers. If it happens, then the
    // newly created copy will never be destroyed.
    assert(!guaranteedUsers.empty());

    ValueLifetimeAnalysis lifetimeAnalysis(borrow, guaranteedUsers);
    ValueLifetimeBoundary borrowBoundary;
    lifetimeAnalysis.computeLifetimeBoundary(borrowBoundary);

    borrowBoundary.visitInsertionPoints(
        [&](SILBasicBlock::iterator insertPt) {
          SILBuilderWithScope builder(insertPt);
          // Use an auto-generated location here, because insertPt may have an
          // incompatible LocationKind
          auto loc =
              RegularLocation::getAutoGeneratedLocation(insertPt->getLoc());
          auto *endBorrow = builder.createEndBorrow(loc, borrow);
          ctx.callbacks.createdNewInst(endBorrow);
        });
  }

  // Create destroys at the end of copy's lifetime. This only needs to consider
  // uses that end the borrow scope.
  {
    ValueLifetimeAnalysis lifetimeAnalysis(copy, borrow->getEndBorrows());
    ValueLifetimeBoundary copyBoundary;
    lifetimeAnalysis.computeLifetimeBoundary(copyBoundary);

    copyBoundary.visitInsertionPoints(
        [&](SILBasicBlock::iterator insertPt) {
          SILBuilderWithScope builder(insertPt);
          auto *destroy = builder.createDestroyValue(loc, copy);
          ctx.callbacks.createdNewInst(destroy);
        });
  }
  return borrow;
}

template <typename RangeTy>
BeginBorrowInst *OwnershipLifetimeExtender::borrowCopyOverGuaranteedUses(
    SILValue newValue, SILBasicBlock::iterator borrowPoint,
    RangeTy guaranteedUsePoints) {
  return borrowCopyOverGuaranteedUsers(newValue, borrowPoint,
                                       makeUserRange(guaranteedUsePoints));
}

// Return the borrow position when replacing guaranteedValue with newValue.
//
// Precondition: newValue's block dominates and reaches guaranteedValue's block.
//
// Postcondition: The returned instruction's block is guaranteedValue's block.
//
// If \p newValue and \p guaranteedValue are in the same block, borrow at the
// newValue just in case it is defined later in the block (to avoid scanning
// instructions). Otherwise, borrow in the guaranteedValue's block to avoid
// introducing the borrow scope too early--not only would this require extra
// cleanup, but it would hinder optimization.
static SILBasicBlock::iterator getBorrowPoint(SILValue newValue,
                                              SILValue guaranteedValue) {
  if (newValue->getParentBlock() == guaranteedValue->getParentBlock())
    return newValue->getNextInstruction()->getIterator();

  return guaranteedValue->getNextInstruction()->getIterator();
}

/// Borrow \p newValue over the lifetime of \p guaranteedValue. Return the
/// new guaranteed value or an empty SILValue when there are no uses.
///
/// TODO: determine whether \p newValue's borrow scope already encompasses all
/// uses of \p guaranteedValue and avoid the copy-borrow. Handle the case where
/// \p newValue is a chain of guaranteed ownership-forwarding operations.
///
/// TODO: Consider replacing all of newValue's uses with the new copy of
/// newValue. This may allow newValue's original borrow scope to be removed,
/// which then allows the copy to be removed. The result would be a single
/// borrow scope over all newValue's and guaranteedValue's uses, which is
/// usually preferable to a new copy and separate borrow scope. When doing
/// this, we can use newValue as the borrow point instead of getBorrowPoint.
SILValue
OwnershipLifetimeExtender::borrowOverValue(SILValue newValue,
                                           SILValue guaranteedValue) {
  // Avoid borrowing guaranteed function arguments.
  if (isa<SILFunctionArgument>(newValue) &&
      newValue->getOwnershipKind() == OwnershipKind::Guaranteed) {
    return newValue;
  }
  auto borrowedValue = BorrowedValue(guaranteedValue);
  if (borrowedValue && borrowedValue.isLocalScope()) {
    return borrowCopyOverScope(newValue, borrowedValue);
  }
  if (ctx.guaranteedUsePoints.empty())
    return newValue;

  // FIXME: use GuaranteedOwnershipExtension
  auto borrowPt = getBorrowPoint(newValue, guaranteedValue);
  return borrowCopyOverGuaranteedUses(
      newValue, borrowPt, ArrayRef<Operand *>(ctx.guaranteedUsePoints));
}

// Borrow \p newValue over \p singleGuaranteedUse. Return the new guaranteed
// value.
//
// Precondition: \p newValue dominates \p singleGuaranteedUse.
//
// Precondition: If \p singleGuaranteedUse ends a borrowed lifetime, the \p
// newValue also dominates the beginning of the borrow scope.
//
// If \p singleGuaranteedUse is lifetime-ending, then two forms
// of cleanup are performed, anticipating that singleGuaranteedUse will be
// replaced with the returned value.
//
// 1. Insert an end_borrow for the original borrow at the point of the replaced
// use.
//
// 2. Insert end_borrows for the new borrow at all the original borrow's
// scope-ending uses that aren't being replaced.
SILValue
OwnershipLifetimeExtender::borrowOverSingleUse(SILValue newValue,
                                               Operand *singleGuaranteedUse) {
  // Avoid borrowing guaranteed function arguments.
  if (isa<SILFunctionArgument>(newValue) &&
      newValue->getOwnershipKind() == OwnershipKind::Guaranteed) {
    return newValue;
  }
  if (!singleGuaranteedUse->isLifetimeEnding()) {
    auto borrowPt = newValue->getNextInstruction()->getIterator();
    return borrowCopyOverGuaranteedUses(
        newValue, borrowPt, ArrayRef<Operand *>(singleGuaranteedUse));
  }
  // A guaranteed lifetime-ending use is always defined by a BorrowedValue.
  auto oldBorrowedVal = BorrowedValue(singleGuaranteedUse->get());
  BeginBorrowInst *newBeginBorrow =
      borrowCopyOverScope(newValue, oldBorrowedVal);

  // Cleanup the original scope, anticipating that it will lose an end-point.
  SILInstruction *usePoint = singleGuaranteedUse->getUser();
  auto *endOldBorrow = SILBuilderWithScope(usePoint).createEndBorrow(
      usePoint->getLoc(), oldBorrowedVal.value);
  ctx.callbacks.createdNewInst(endOldBorrow);

  // Cleanup the new scope since it only inherits one end-point.
  oldBorrowedVal.visitLocalScopeEndingUses([&](Operand *endScope) {
    auto borrowingOper = BorrowingOperand(endScope);
    if (borrowingOper.isReborrow())
      return true;

    auto *oldEndBorrow = endScope->getUser();
    auto *endNewBorrow =
        SILBuilderWithScope(oldEndBorrow)
            .createEndBorrow(oldEndBorrow->getLoc(), newBeginBorrow);
    ctx.callbacks.createdNewInst(endNewBorrow);
    return true;
  });
  return newBeginBorrow;
}

SILValue OwnershipLifetimeExtender::borrowOverSingleNonLifetimeEndingUser(
    SILValue newValue, SILInstruction *nonLifetimeEndingUser) {
  // Avoid borrowing guaranteed function arguments.
  if (isa<SILFunctionArgument>(newValue) &&
      newValue->getOwnershipKind() == OwnershipKind::Guaranteed) {
    return newValue;
  }
  auto borrowPt = newValue->getNextInstruction()->getIterator();
  return borrowCopyOverGuaranteedUsers(
      newValue, borrowPt, ArrayRef<SILInstruction *>(nonLifetimeEndingUser));
}

SILValue swift::makeGuaranteedValueAvailable(SILValue value,
                                             SILInstruction *user,
                                             InstModCallbacks callbacks) {
  OwnershipFixupContext ctx{callbacks};
  OwnershipLifetimeExtender extender{ctx};
  return extender.borrowOverSingleNonLifetimeEndingUser(value, user);
}

//===----------------------------------------------------------------------===//
//                OwnershipRAUWUtility - RAUW + fix ownership
//===----------------------------------------------------------------------===//

/// Given an old value and a new value, lifetime extend new value as appropriate
/// so we can RAUW new value with old value and preserve ownership
/// invariants. We leave fixing up the lifetime of old value to our caller.
namespace {

struct OwnershipRAUWPrepare {
  SILValue oldValue;
  OwnershipFixupContext &ctx;

  OwnershipLifetimeExtender getLifetimeExtender() { return {ctx}; }

  const InstModCallbacks &getCallbacks() const { return ctx.callbacks; }

  // For terminator results, the consuming point is the predecessor's
  // terminator. This avoids destroys on unused paths. It is also the
  // instruction which will be deleted, thus needs operand cleanup.
  SILInstruction *getConsumingPoint() const {
    if (auto *blockArg = dyn_cast<SILPhiArgument>(oldValue))
      return blockArg->getTerminatorForResult();

    return cast<SingleValueInstruction>(oldValue);
  }

  SILValue prepareReplacement(SILValue newValue);

private:
  SILValue prepareUnowned(SILValue newValue);
};

} // anonymous namespace

SILValue OwnershipRAUWPrepare::prepareUnowned(SILValue newValue) {
  auto &callbacks = ctx.callbacks;
  switch (newValue->getOwnershipKind()) {
  case OwnershipKind::None:
    llvm_unreachable("Should have been handled elsewhere");
  case OwnershipKind::Any:
    llvm_unreachable("Invalid for values");
  case OwnershipKind::Unowned:
    // An unowned value can always be RAUWed with another unowned value.
    return newValue;
  case OwnershipKind::Guaranteed: {
    // If we have an unowned value that we want to replace with a guaranteed
    // value, we need to ensure that the guaranteed value is live at all use
    // points of the unowned value. If so, just replace and continue.
    //
    // TODO: Implement this for more interesting cases.
    if (isa<SILFunctionArgument>(newValue))
      return newValue;

    // Otherwise, we need to lifetime extend the borrow over all of the use
    // points. To do so, we copy the value, borrow it, and insert an unchecked
    // ownership conversion to unowned at all uses that are terminator uses.
    //
    // We need to insert the conversion since if we have a non-argument
    // guaranteed value since its scope will end before the terminator so we
    // need to convert the value to unowned early.
    //
    // TODO: Do we need a separate array here?
    SmallVector<Operand *, 8> oldValueUses(oldValue->getUses());
    for (auto *use : oldValueUses) {
      if (auto *ti = dyn_cast<TermInst>(use->getUser())) {
        if (ti->isFunctionExiting()) {
          SILBuilderWithScope builder(ti);
          auto *newInst = builder.createUncheckedOwnershipConversion(
              ti->getLoc(), use->get(), OwnershipKind::Unowned);
          callbacks.createdNewInst(newInst);
          callbacks.setUseValue(use, newInst);
        }
      }
    }

    auto extender = getLifetimeExtender();
    auto borrowPt = getBorrowPoint(newValue, oldValue);
    SILValue borrow = extender.borrowCopyOverGuaranteedUses(
        newValue, borrowPt, oldValue->getUses());
    return borrow;
  }
  case OwnershipKind::Owned: {
    // If we have an unowned value that we want to replace with an owned value,
    // we first check if the owned value is live over all use points of the old
    // value. If so, just RAUW and continue.
    //
    // TODO: Implement this.

    // Otherwise, insert a copy of the owned value and lifetime extend that over
    // all uses of the value and then RAUW.
    //
    // NOTE: For terminator uses, we funnel the use through an
    // unchecked_ownership_conversion to ensure that we can end the lifetime of
    // our owned/guaranteed value before the terminator.
    SmallVector<Operand *, 8> oldValueUses(oldValue->getUses());
    for (auto *use : oldValueUses) {
      if (auto *ti = dyn_cast<TermInst>(use->getUser())) {
        if (ti->isFunctionExiting()) {
          SILBuilderWithScope builder(ti);
          auto *newInst = builder.createUncheckedOwnershipConversion(
              ti->getLoc(), use->get(), OwnershipKind::Unowned);
          callbacks.createdNewInst(newInst);
          callbacks.setUseValue(use, newInst);
        }
      }
    }
    auto extender = getLifetimeExtender();
    SILValue copy = extender.createPlusZeroCopy(newValue, oldValue->getUses());
    return copy;
  }
  }
  llvm_unreachable("covered switch isn't covered?!");
}

SILValue OwnershipRAUWPrepare::prepareReplacement(SILValue newValue) {
  assert(oldValue->getFunction()->hasOwnership());

  // A value with no uses can be "replaced" without fixup.
  // (e.g. a dead no-ownership value can be replaced by an owned value even
  // though hasValidRAUWOwnership will be false).
  if (oldValue->use_empty())
    return newValue;

  assert(
      OwnershipRAUWHelper::hasValidRAUWOwnership(oldValue, newValue,
                                                 ctx.guaranteedUsePoints) &&
      "Should have checked if can perform this operation before calling it?!");
  // If our new value is just none, we can pass anything to it so just RAUW
  // and return.
  //
  // NOTE: This handles RAUWing with undef.
  if (newValue->getOwnershipKind() == OwnershipKind::None)
    return newValue;

  assert(oldValue->getOwnershipKind() != OwnershipKind::None);

  switch (oldValue->getOwnershipKind()) {
  case OwnershipKind::None:
    // If our old value was none and our new value is not, we need to do
    // something more complex that we do not support yet, so bail. We should
    // have not called this function in such a case.
    llvm_unreachable("Should have been handled elsewhere");
  case OwnershipKind::Any:
    llvm_unreachable("Invalid for values");
  case OwnershipKind::Guaranteed: {
    return getLifetimeExtender().borrowOverValue(newValue, oldValue);
  }
  case OwnershipKind::Owned: {
    // If we have an owned value that we want to replace with a value with any
    // other non-None ownership, we need to copy the other value for a
    // lifetimeEnding RAUW, RAUW the value, and insert a destroy_value of
    // the original value.
    auto extender = getLifetimeExtender();
    auto *consumingPoint = getConsumingPoint();
    SILValue copy = extender.createPlusOneCopy(newValue, consumingPoint);

    cleanupOperandsBeforeDeletion(consumingPoint, ctx.callbacks);
    return copy;
  }
  case OwnershipKind::Unowned: {
    return prepareUnowned(newValue);
  }
  }
  llvm_unreachable("Covered switch isn't covered?!");
}

//===----------------------------------------------------------------------===//
//                     Interior Pointer Operand Rebasing
//===----------------------------------------------------------------------===//

/// Return an address equivalent to \p newValue that can be used to replace all
/// uses of \p oldValue.
///
/// Precondition: RAUW of two addresses
SILValue
OwnershipRAUWHelper::getReplacementAddress() {
  assert(oldValue->getType().isAddress() && newValue->getType().isAddress());

  // If newValue was not generated by an interior pointer, then it cannot
  // be within a borrow scope, so direct replacement works.
  if (!requiresCopyBorrowAndClone())
    return newValue;

  // newValue may be within a borrow scope, and oldValue may have uses that are
  // outside of newValue's borrow scope.
  //
  // So, we need to copy/borrow the base value of the interior pointer to
  // lifetime extend the base value over the new uses. Then we clone the
  // interior pointer instruction and change the clone to use our new borrowed
  // value. Then we RAUW as appropriate.
  OwnershipLifetimeExtender extender{*ctx};
  auto base = ctx->extraAddressFixupInfo.base;
  auto borrowPt = getBorrowPoint(newValue, oldValue);
  // FIXME: why does this use allAddressUsesFromOldValue instead of
  // guaranteedUsePoints?
  BeginBorrowInst *bbi = extender.borrowCopyOverGuaranteedUses(
      base.getReference(), borrowPt,
      llvm::makeArrayRef(
        ctx->extraAddressFixupInfo.allAddressUsesFromOldValue));
  auto bbiNext = &*std::next(bbi->getIterator());
  auto *refProjection = cast<SingleValueInstruction>(base.getBaseAddress());
  auto *newBase = refProjection->clone(bbiNext);
  ctx->callbacks.createdNewInst(newBase);
  newBase->setOperand(0, bbi);

  // Now that we have extended our lifetime as appropriate, we need to recreate
  // the access path from newValue to refProjection but upon newBase.
  //
  // This cloner invocation must match the canCloneUseDefChain check in the
  // constructor.
  auto checkBase = [&](SILValue srcAddr) {
    return (srcAddr == refProjection) ? SILValue(newBase) : SILValue();
  };
  SILValue clonedAddr = cloneUseDefChain(newValue, bbiNext, checkBase);
  assert(clonedAddr != newValue && "expect at least the base to be replaced");
  return clonedAddr;
}

//===----------------------------------------------------------------------===//
//                            OwnershipRAUWHelper
//===----------------------------------------------------------------------===//

OwnershipRAUWHelper::OwnershipRAUWHelper(OwnershipFixupContext &inputCtx,
                                         SILValue inputOldValue,
                                         SILValue inputNewValue)
    : ctx(&inputCtx), oldValue(inputOldValue), newValue(inputNewValue) {
  // If we are already not valid, just bail.
  if (!isValid())
    return;

  // If we are not in ownership, we can always RAUW successfully so just bail
  // and leave the object valid.
  if (!oldValue->getFunction()->hasOwnership())
    return;

  // This utility currently only handles erasing SingleValueInstructions and
  // terminator results.
  assert(isa<SingleValueInstruction>(inputOldValue)
         || cast<SILPhiArgument>(inputOldValue)->isTerminatorResult());

  // Precondition: If \p oldValue is a BorrowedValue that introduces a local
  // borrow scope, then \p newValue must either be defined in the same block as
  // \p oldValue, or it must dominate \p oldValue (rather than merely
  // dominating its uses).
  //
  // Handling cases where the new value does not dominate the old borrow scope
  // would require signficant complexity and such cases are currently impossible
  // to test. Consideration would be required for handling a new value within an
  // inner loop, while the old borrow scope is introduced outside that
  // loop. Since it generally makes no sense to do this kind of replacement,
  // we simply rule it out as an RAUW precondition.
  //
  // TODO: this could be converted to a bailout if we don't want the client code
  // to explicitly check this case. But then we may want DominanceInfo to be
  // available, which could cheaper in extreme cases because it caches results.
  SWIFT_ASSERT_ONLY_DECL(auto borrowedVal = BorrowedValue(inputOldValue));
  assert((!borrowedVal || !borrowedVal.isLocalScope()
          || checkDominates(inputNewValue->getParentBlock(),
                            inputOldValue->getParentBlock()))
         && "OSSA RAUW requires reachability and dominance");

  // Clear the context before populating it anew.
  ctx->clear();

  // A value with no uses can be "replaced" regardless of its uses. Bypass all
  // the use-checking logic, which assumes a non-empty use list.
  if (oldValue->use_empty()) {
    return;
  }

  // Otherwise, lets check if we can perform this RAUW operation. If we can't,
  // set ctx to nullptr to invalidate the helper and return.
  if (!canFixUpOwnershipForRAUW(oldValue, newValue, inputCtx)) {
    invalidate();
    return;
  }

  // If we have an object, at this point we are good to go so we can just
  // return.
  if (newValue->getType().isObject())
    return;

  // But if we have an address, we need to check if new value is from an
  // interior pointer or not in a way that the pass understands. What we do is:
  //
  // 1. Early exit some cases that we know can never have interior pointers.
  //
  // 2. Compute the AccessPathWithBase of newValue. If we do not get back a
  //    valid such object, invalidate and then bail.
  //
  // 3. Then we check if the base address is the result of an interior pointer
  //    instruction. If we do not find one we bail.
  //
  // 4. Then grab the base value of the interior pointer operand. We only
  //    support cases where we have a single BorrowedValue as our base. This is
  //    a safe future proof assumption since one reborrows are on
  //    structs/tuple/destructures, a guaranteed value will always be associated
  //    with a single BorrowedValue, so this will never fail (and the code will
  //    probably be DCEed).
  //
  // 5. Then we compute an AccessPathWithBase for oldValue and then find its
  //    derived uses. If we fail, we bail.
  //
  // 6. At this point, we know that we can perform this RAUW. The only question
  //    is if we need to when we RAUW copy the interior pointer base value. We
  //    perform this check by making sure all of the old value's derived uses
  //    are within our BorrowedValue's scope. If so, we clear the extra state we
  //    were tracking (the interior pointer/oldValue's transitive uses), so we
  //    perform just a normal RAUW (without inserting the copy) when we RAUW.
  //
  // We can always RAUW an address with a pointer_to_address since if there
  // were any interior pointer constraints on whatever address pointer came
  // from, the address_to_pointer producing that value erases that
  // information, so we can RAUW without worrying.
  //
  // NOTE: We also need to handle this here since a pointer_to_address is not a
  // valid base value for an access path since it doesn't refer to any storage.
  AddressOwnership addressOwnership(newValue);
  if (!addressOwnership.hasLocalOwnershipLifetime())
    return;

  ctx->extraAddressFixupInfo.base = addressOwnership.base;
  SILValue baseAddress = ctx->extraAddressFixupInfo.base.getBaseAddress();

  // For now, just gather up uses
  //
  // FIXME: get rid of allAddressUsesFromOldValue. Shouldn't this already be
  // included in guaranteedUsePoints?
  auto &oldValueUses = ctx->extraAddressFixupInfo.allAddressUsesFromOldValue;
  if (findTransitiveUsesForAddress(oldValue, &oldValueUses)
      != AddressUseKind::NonEscaping) {
    invalidate();
    return;
  }
  if (addressOwnership.areUsesWithinLifetime(oldValueUses)) {
    // We do not need to copy the base value! Clear the extra info we have.
    ctx->extraAddressFixupInfo.clear();
    return;
  }
  // This cloner check must match the later cloner invocation in
  // getReplacementAddress()
  auto *baseInst = cast<SingleValueInstruction>(baseAddress);
  auto checkBase = [&](SILValue srcAddr) {
    return (srcAddr == baseInst) ? SILValue(baseInst) : SILValue();
  };
  if (!canCloneUseDefChain(newValue, checkBase)) {
    invalidate();
    return;
  }
}

SILValue OwnershipRAUWHelper::prepareReplacement(SILValue rewrittenNewValue) {
  assert(isValid() && "OwnershipRAUWHelper invalid?!");

  if (rewrittenNewValue) {
    // Everything about \n newValue that the constructor checks should also be
    // true for rewrittenNewValue.
    // FIXME: enable these...
    // assert(rewrittenNewValue->getType() == newValue->getType());
    // assert(rewrittenNewValue->getOwnershipKind()
    //        == newValue->getOwnershipKind());
    // assert(rewrittenNewValue->getParentBlock() == newValue->getParentBlock());
    assert(!newValue->getType().isAddress() ||
           AddressOwnership(rewrittenNewValue) == AddressOwnership(newValue));

    newValue = rewrittenNewValue;
  }
  assert(newValue && "prepareReplacement can only be called once");
  SWIFT_DEFER { newValue = SILValue(); };

  if (!oldValue->getFunction()->hasOwnership())
    return newValue;

  if (oldValue->getType().isAddress()) {
    return getReplacementAddress();
  }
  OwnershipRAUWPrepare rauwPrepare{oldValue, *ctx};
  return rauwPrepare.prepareReplacement(newValue);
}

SILBasicBlock::iterator
OwnershipRAUWHelper::perform(SILValue replacementValue) {
  if (!replacementValue)
    replacementValue = prepareReplacement();

  assert(!newValue && "prepareReplacement() must be called");

  // Make sure to always clear our context after we transform.
  SWIFT_DEFER { ctx->clear(); };

  if (auto *svi = dyn_cast<SingleValueInstruction>(oldValue))
    return replaceAllUsesAndErase(svi, replacementValue, ctx->callbacks);

  // The caller must rewrite the terminator after RAUW.
  auto *term = cast<SILPhiArgument>(oldValue)->getTerminatorForResult();
  auto nextII = term->getParent()->end();
  return replaceAllUses(oldValue, replacementValue, nextII, ctx->callbacks);
}

//===----------------------------------------------------------------------===//
//                           Single Use Replacement
//===----------------------------------------------------------------------===//

namespace {

/// Given a use and a new value, lifetime extend new value as appropriate so we
/// can replace use->get() with newValue and preserve ownership invariants. We
/// assume that old value will be left alone and not deleted so we insert
/// compensating cleanups.
struct SingleUseReplacementUtility {
  Operand *use;
  SILValue newValue;
  OwnershipFixupContext &ctx;

  SILBasicBlock::iterator handleUnowned();
  SILBasicBlock::iterator handleOwned();
  SILBasicBlock::iterator handleGuaranteed();

  SILBasicBlock::iterator perform();

  OwnershipLifetimeExtender getLifetimeExtender() { return {ctx}; }

  const InstModCallbacks &getCallbacks() const { return ctx.callbacks; }
};

} // anonymous namespace

SILBasicBlock::iterator SingleUseReplacementUtility::handleUnowned() {
  auto &callbacks = ctx.callbacks;
  switch (newValue->getOwnershipKind()) {
  case OwnershipKind::None:
    llvm_unreachable("Should have been handled elsewhere");
  case OwnershipKind::Any:
    llvm_unreachable("Invalid for values");
  case OwnershipKind::Unowned:
    // An unowned value can always be RAUWed with another unowned value.
    return replaceSingleUse(use, newValue, callbacks);
  case OwnershipKind::Guaranteed: {
    // If we have an unowned value use that we want to replace with a guaranteed
    // value, we need to ensure that the guaranteed value is live at that use
    // point. If we know that is always true, just perform the replace.
    //
    // FIXME: Expand the cases here.
    if (isa<SILFunctionArgument>(newValue))
      return replaceSingleUse(use, newValue, callbacks);

    // Otherwise, we need to lifetime extend newValue to the use. If the actual
    // use is a terminator, we need to insert an unchecked_ownership_conversion
    // since our value can not be live at the terminator itself.
    if (auto *ti = dyn_cast<TermInst>(use->getUser())) {
      if (ti->isFunctionExiting()) {
        SILBuilderWithScope builder(ti);
        auto *newInst = builder.createUncheckedOwnershipConversion(
            ti->getLoc(), use->get(), OwnershipKind::Unowned);
        callbacks.createdNewInst(newInst);
        callbacks.setUseValue(use, newInst);
      }
    }

    auto extender = getLifetimeExtender();
    SILValue borrow = extender.borrowOverSingleUse(newValue, use);
    assert(!use->isLifetimeEnding()
           && "Test single-use replacement of a scope-ending instruction");

    return replaceSingleUse(use, borrow, callbacks);
  }
  case OwnershipKind::Owned: {
    // If we have an unowned value use that we want to replace with an owned
    // value use.  we first check if the owned value is live over all use points
    // of the old value. If so, just RAUW and continue.
    //
    // TODO: Implement this.

    // Otherwise, insert a copy of the owned value and lifetime extend that over
    // the use.
    //
    // NOTE: For terminator uses, we funnel the use through an
    // unchecked_ownership_conversion to ensure that we can end the lifetime of
    // our owned/guaranteed value before the terminator.
    if (auto *ti = dyn_cast<TermInst>(use->getUser())) {
      if (ti->isFunctionExiting()) {
        SILBuilderWithScope builder(ti);
        auto *newInst = builder.createUncheckedOwnershipConversion(
            ti->getLoc(), use->get(), OwnershipKind::Unowned);
        callbacks.createdNewInst(newInst);
        callbacks.setUseValue(use, newInst);
      }
    }

    auto extender = getLifetimeExtender();
    SILValue copy = extender.createPlusZeroCopy(newValue, {use});
    return replaceSingleUse(use, copy, callbacks);
  }
  }
  llvm_unreachable("covered switch isn't covered?!");
}

SILBasicBlock::iterator SingleUseReplacementUtility::handleGuaranteed() {
  // Ok, our use is guaranteed and our new value may not be guaranteed.
  auto extender = getLifetimeExtender();

  // If we don't have a lifetime ending use, just create the borrow.
  SILValue copy = extender.borrowOverSingleUse(newValue, use);

  // Then replace use->get() with this copy. We will insert compensating end
  // scope instructions on use->get() if we need to.
  return replaceSingleUse(use, copy, ctx.callbacks);
}

SILBasicBlock::iterator SingleUseReplacementUtility::handleOwned() {
  // Ok, our old value is owned and our new value may not be owned. First
  // lifetime extend newValue to use->getUser() inserting destroy_values along
  // any paths that do not go through use->getUser().
  auto extender = getLifetimeExtender();

  if (use->isLifetimeEnding()) {
    // If our use is a lifetime ending use, then create a plus one copy and
    // RAUW.
    SILValue copy = extender.createPlusOneCopy(newValue, use->getUser());
    // Then replace use->get() with this copy. We will insert compensating end
    // scope instructions on use->get() if we need to.
    return replaceSingleUse(use, copy, ctx.callbacks);
  }

  // If we don't have a lifetime ending use, just create a +0 copy and set the
  // use. All destroys will be placed for us.
  SILValue copy =
      extender.createPlusZeroCopy<ArrayRef<Operand *>>(newValue, {use});

  // Then replace use->get() with this copy. We will insert compensating end
  // scope instructions on use->get() if we need to.
  return replaceSingleUse(use, copy, ctx.callbacks);
}

SILBasicBlock::iterator SingleUseReplacementUtility::perform() {
  auto oldValue = use->get();
  assert(oldValue->getFunction()->hasOwnership());

  // If our new value is just none, we can pass anything to do it so just RAUW
  // and return.
  //
  // NOTE: This handles RAUWing with undef.
  if (newValue->getOwnershipKind() == OwnershipKind::None)
    return replaceSingleUse(use, newValue, ctx.callbacks);

  assert(SILValue(oldValue)->getOwnershipKind() != OwnershipKind::None);

  switch (SILValue(oldValue)->getOwnershipKind()) {
  case OwnershipKind::None:
    // If our old value was none and our new value is not, we need to do
    // something more complex that we do not support yet, so bail. We should
    // have not called this function in such a case.
    llvm_unreachable("Should have been handled elsewhere");
  case OwnershipKind::Any:
    llvm_unreachable("Invalid for values");
  case OwnershipKind::Guaranteed:
    return handleGuaranteed();
  case OwnershipKind::Owned:
    return handleOwned();
  case OwnershipKind::Unowned:
    return handleUnowned();
  }
  llvm_unreachable("Covered switch isn't covered?!");
}

//===----------------------------------------------------------------------===//
//                      OwnershipReplaceSingleUseHelper
//===----------------------------------------------------------------------===//

OwnershipReplaceSingleUseHelper::OwnershipReplaceSingleUseHelper(
    OwnershipFixupContext &inputCtx, Operand *inputUse, SILValue inputNewValue)
    : ctx(&inputCtx), use(inputUse), newValue(inputNewValue) {
  // If we are already not valid, just bail.
  if (!isValid())
    return;

  // If we do not have ownership, we are already done.
  if (!inputUse->getUser()->getFunction()->hasOwnership())
    return;

  // If we have an address, bail. We don't support this.
  if (newValue->getType().isAddress()) {
    invalidate();
    return;
  }

  // Otherwise, lets check if we can perform this RAUW operation. If we can't,
  // set ctx to nullptr to invalidate the helper and return.
  SmallVector<Operand *, 1> oldUses;
  oldUses.push_back(use);
  if (!OwnershipRAUWHelper::hasValidRAUWOwnership(use->get(), newValue,
                                                  oldUses)) {
    invalidate();
    return;
  }

  if (inputUse->getOperandOwnership() == OperandOwnership::Reborrow) {
    // FIXME: Use GuaranteedPhiBorrowFixup to handle this case during perform().
    invalidate();
    return;
  }
}

SILBasicBlock::iterator OwnershipReplaceSingleUseHelper::perform() {
  assert(isValid() && "OwnershipReplaceSingleUseHelper invalid?!");

  if (!use->getUser()->getFunction()->hasOwnership())
    return replaceSingleUse(use, newValue, ctx->callbacks);

  // Make sure to always clear our context after we transform.
  SWIFT_DEFER { ctx->clear(); };
  SingleUseReplacementUtility utility{use, newValue, *ctx};
  return utility.perform();
}

//===----------------------------------------------------------------------===//
//                      createBorrowScopeForPhiOperands
//===----------------------------------------------------------------------===//

/// Given a phi that has been newly created or converted from terminator
/// results, check if any of the phi's operands are inner guaranteed values.
/// This is invalid OSSA because the phi is a reborrow. Like all
/// borrow-scope-ending instructions a phi must directly use the BorrowedValue
/// that introduces the scope.
///
/// Create nested borrow scopes for its operands.
///
/// Transitively follow its phi uses.
///
/// Create end_borrows at all points that cover the inner uses.
///
/// The client must check canCloneTerminator() first to make sure that the
/// search for transitive uses does not encounter a PointerEscape.
class GuaranteedPhiBorrowFixup {
  // A phi in mustConvertPhis has already been determined to be part of this
  // new nested borrow scope.
  SmallSetVector<SILPhiArgument *, 8> mustConvertPhis;

  // Phi operands that are already within the new nested borrow scope.
  llvm::SmallDenseSet<PhiOperand, 8> nestedPhiOperands;

public:
  /// Return true if an extended nested borrow scope was created.
  bool createExtendedNestedBorrowScope(SILPhiArgument *newPhi);

protected:
  bool phiOperandNeedsBorrow(Operand *operand) {
    SILValue inVal = operand->get();
    if (inVal->getOwnershipKind() != OwnershipKind::Guaranteed) {
      assert(inVal->getOwnershipKind() == OwnershipKind::None);
      return false;
    }
    // This operand needs a nested borrow if inVal is not a BorrowedValue.
    return !bool(BorrowedValue(inVal));
  }

  void borrowPhiOperand(Operand *oper) {
    // Begin the borrow just before the branch.
    SILInstruction *borrowPoint = oper->getUser();
    auto loc = RegularLocation::getAutoGeneratedLocation(borrowPoint->getLoc());
    auto *borrow =
        SILBuilderWithScope(borrowPoint).createBeginBorrow(loc, oper->get());
    oper->set(borrow);
  }

  EndBorrowInst *createEndBorrow(SILValue guaranteedValue,
                                 SILBasicBlock::iterator borrowPoint) {
    auto loc = borrowPoint->getLoc();
    return SILBuilderWithScope(borrowPoint)
        .createEndBorrow(loc, guaranteedValue);
  }

  void insertEndBorrowsAndFindPhis(SILPhiArgument *phi);
};

void GuaranteedPhiBorrowFixup::insertEndBorrowsAndFindPhis(
    SILPhiArgument *phi) {
  // Scope ending instructions are only needed for nontrivial results.
  if (phi->getOwnershipKind() != OwnershipKind::Guaranteed) {
    assert(phi->getOwnershipKind() == OwnershipKind::None);
    return;
  }
  SmallVector<Operand *, 16> usePoints;
  bool result = findInnerTransitiveGuaranteedUses(phi, &usePoints);
  assert(result && "should be checked by canCloneTerminator");
  (void)result;

  // Add usePoints to a set for phi membership checking.
  //
  // FIXME: consider integrating with ValueLifetimeBoundary instead.
  SmallPtrSet<Operand *, 16> useSet(usePoints.begin(), usePoints.end());

  auto phiUsers = llvm::map_range(usePoints, ValueBase::UseToUser());
  ValueLifetimeAnalysis lifetimeAnalysis(phi, phiUsers);
  ValueLifetimeBoundary boundary;
  lifetimeAnalysis.computeLifetimeBoundary(boundary);

  for (auto *boundaryEdge : boundary.boundaryEdges) {
    createEndBorrow(phi, boundaryEdge->begin());
  }

  for (SILInstruction *lastUser : boundary.lastUsers) {
    // If the last use is a branch, transitively process the phi.
    if (isa<BranchInst>(lastUser)) {
      for (Operand &oper : lastUser->getAllOperands()) {
        if (!useSet.count(&oper))
          continue;

        PhiOperand phiOper(&oper);
        nestedPhiOperands.insert(phiOper);
        mustConvertPhis.insert(phiOper.getValue());
        continue;
      }
    }
    // If the last user is a terminator, add the successors as boundary edges.
    if (isa<TermInst>(lastUser)) {
      for (auto *succBB : lastUser->getParent()->getSuccessorBlocks()) {
        // succBB cannot already be in boundaryEdges. It has a
        // single predecessor with liveness ending at the terminator, which
        // means it was not live into any successor blocks.
        createEndBorrow(phi, succBB->begin());
      }
      continue;
    }
    // Otherwise, just plop down an end_borrow after the last use.
    createEndBorrow(phi, std::next(lastUser->getIterator()));
  }
}

// For each phi that transitively uses an inner guaranteed value, create nested
// borrow scopes so that it is a well-formed reborrow.
bool GuaranteedPhiBorrowFixup::
createExtendedNestedBorrowScope(SILPhiArgument *newPhi) {
  // Determine if this new phi needs a nested borrow scope. If so, seed the
  // Visit phi operands, returning false as soon as one needs a borrow.
  if (!newPhi->visitIncomingPhiOperands(
        [&](Operand *op) { return !phiOperandNeedsBorrow(op); })) {
    mustConvertPhis.insert(newPhi);
  }
  if (mustConvertPhis.empty())
    return false;

  // mustConvertPhis grows in this loop.
  for (unsigned mustConvertIdx = 0; mustConvertIdx < mustConvertPhis.size();
         ++mustConvertIdx) {
    SILPhiArgument *phi = mustConvertPhis[mustConvertIdx];
    insertEndBorrowsAndFindPhis(phi);
  }
  // To handle recursive phis, first discover all phis before attempting to
  // borrow any phi operands.
  for (SILPhiArgument *phi : mustConvertPhis) {
    phi->visitIncomingPhiOperands([&](Operand *op) {
      if (!nestedPhiOperands.count(op))
        borrowPhiOperand(op);
      return true;
    });
  }
  return true;
}

// Note: \p newPhi itself might not have Guaranteed ownership. A phi that
// converts Guaranteed to None ownership still needs nested borrows.
//
// Note: This may be called on partially invalid OSSA form, where multiple
// newly created phis do not yet have a borrow scope. The implementation
// assumes that this API will eventually be called for all such new phis until
// OSSA is fully valid.
bool swift::createBorrowScopeForPhiOperands(SILPhiArgument *newPhi) {
  if (newPhi->getOwnershipKind() != OwnershipKind::Guaranteed
      && newPhi->getOwnershipKind() != OwnershipKind::None) {
      return false;
  }
  return GuaranteedPhiBorrowFixup().createExtendedNestedBorrowScope(newPhi);
}

bool swift::extendStoreBorrow(StoreBorrowInst *sbi,
                              SmallVectorImpl<Operand *> &newUses,
                              DeadEndBlocks *deadEndBlocks,
                              InstModCallbacks callbacks) {
  ScopedAddressValue scopedAddress(sbi);

  SmallVector<SILBasicBlock *, 4> discoveredBlocks;
  SSAPrunedLiveness storeBorrowLiveness(&discoveredBlocks);

  // FIXME: if OSSA lifetimes are complete, then we don't need transitive
  // liveness here.
  AddressUseKind useKind =
      scopedAddress.computeTransitiveLiveness(storeBorrowLiveness);

  // If all new uses are within store_borrow boundary, no need for extension.
  if (storeBorrowLiveness.areUsesWithinBoundary(newUses, deadEndBlocks)) {
    return true;
  }

  if (useKind != AddressUseKind::NonEscaping) {
    return false;
  }

  // store_borrow extension is possible only when there are no other
  // store_borrows to the same destination within the store_borrow's lifetime
  // built from newUsers.
  if (hasOtherStoreBorrowsInLifetime(sbi, &storeBorrowLiveness,
                                     deadEndBlocks)) {
    return false;
  }

  InstModCallbacks tempCallbacks = callbacks;
  InstructionDeleter deleter(std::move(tempCallbacks));
  GuaranteedOwnershipExtension borrowExtension(deleter, *deadEndBlocks,
                                               sbi->getFunction());
  auto status = borrowExtension.checkBorrowExtension(
      BorrowedValue(sbi->getSrc()), newUses);
  if (status == GuaranteedOwnershipExtension::Invalid) {
    return false;
  }

  borrowExtension.transform(status);

  SmallVector<Operand *, 4> endBorrowUses;
  // Collect old scope-ending instructions.
  scopedAddress.visitScopeEndingUses([&](Operand *op) {
    endBorrowUses.push_back(op);
    return true;
  });

  for (auto *use : newUses) {
    // Update newUsers as non-lifetime ending.
    storeBorrowLiveness.updateForUse(use->getUser(),
                                     /* lifetimeEnding */ false);
  }

  // Add new scope-ending instructions.
  scopedAddress.endScopeAtLivenessBoundary(&storeBorrowLiveness);

  // Remove old scope-ending instructions.
  for (auto *endBorrowUse : endBorrowUses) {
    callbacks.deleteInst(endBorrowUse->getUser());
  }

  return true;
}
