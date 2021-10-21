//===--- OwnershipOptUtils.h ----------------------------------------------===//
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

#ifndef SWIFT_SILOPTIMIZER_UTILS_OWNERSHIPOPTUTILS_H
#define SWIFT_SILOPTIMIZER_UTILS_OWNERSHIPOPTUTILS_H

#include "swift/Basic/Defer.h"
#include "swift/SIL/BasicBlockUtils.h"
#include "swift/SIL/NodeDatastructures.h"
#include "swift/SIL/OwnershipUtils.h"
#include "swift/SIL/PrunedLiveness.h"
#include "swift/SIL/SILModule.h"
#include "swift/SILOptimizer/Utils/InstructionDeleter.h"

namespace swift {

/// Returns true if this value requires OSSA cleanups.
inline bool requiresOSSACleanup(SILValue v) {
  return v->getFunction()->hasOwnership() &&
         v->getOwnershipKind() != OwnershipKind::None &&
         v->getOwnershipKind() != OwnershipKind::Unowned;
}

//===----------------------------------------------------------------------===//
//                            Lifetime Completion
//===----------------------------------------------------------------------===//

class OSSALifetimeCompletion {
  // Cache intructions already handled by the recursive algorithm to avoid
  // recomputing their lifetimes.
  ValueSet completedValues;

public:
  OSSALifetimeCompletion(SILFunction *function) : completedValues(function) {}

  /// Insert a lifetime-ending instruction on every path to complete the OSSA
  /// lifetime of \p value.
  ///
  /// \p value must be a non-lexical owned value or a non-lexical borrow
  /// introducer.
  ///
  /// Returns true if any new instructions were created to complete the
  /// lifetime.
  bool completeNonLexicalLifetime(SILValue value);

  /// End the lifetime of \p value in dead-end blocks.
  ///
  /// \p value must be a lexical owned value or a lexical borrow
  /// introducer.
  bool completeLexicalLifetime(SILValue value);

  enum class LifetimeCompletion { NoLifetime, AlreadyComplete, WasCompleted };

  inline LifetimeCompletion completeOSSALifetime(SILValue value) {
    if (value->getOwnershipKind() == OwnershipKind::None)
      return LifetimeCompletion::NoLifetime;

    if (value->getOwnershipKind() != OwnershipKind::Owned) {
      BorrowedValue borrowedValue(value);
      if (!borrowedValue)
        return LifetimeCompletion::NoLifetime;

      if (!borrowedValue.isLocalScope())
        return LifetimeCompletion::AlreadyComplete;
    }
    if (!completedValues.insert(borrowedValue.value))
      return LifetimeCompletion::AlreadyComplete;

    if (value->isLexical())
      return completeLexicalLifetime(value)
                 ? LifetimeCompletion::WasCompleted
                 : LifetimeCompletion::AlreadyComplete;

    return completeNonLexicalLifetime(value)
               ? LifetimeCompletion::WasCompleted
               : LifetimeCompletion::AlreadyComplete;
  }
}

//===----------------------------------------------------------------------===//
//                   Basic scope and lifetime extension API
//===----------------------------------------------------------------------===//

/// Rewrite the lifetime of \p ownedValue to match \p lifetimeBoundary. This may
/// insert copies at forwarding consumes, including phis.
///
/// Precondition: lifetimeBoundary is dominated by ownedValue.
///
/// Precondition: lifetimeBoundary is a superset of ownedValue's current
/// lifetime (therefore, none of the safety checks done during
/// CanonicalizeOSSALifetime are needed here).
void extendOwnedLifetime(SILValue ownedValue,
                         PrunedLivenessBoundary &lifetimeBoundary,
                         InstructionDeleter &deleter);

/// Rewrite the local borrow scope introduced by \p beginBorrow to match \p
/// guaranteedBoundary.
///
/// Precondition: guaranteedBoundary is dominated by beginBorrow which has no
/// reborrows.
///
/// Precondition: guaranteedBoundary is a superset of beginBorrow's current
/// scope (therefore, none of the safety checks done during
/// CanonicalizeBorrowScope are needed here).
void extendLocalBorrow(BeginBorrowInst *beginBorrow,
                       PrunedLivenessBoundary &guaranteedBoundary,
                       InstructionDeleter &deleter);

/// Given a new phi that may use a guaranteed value, create nested borrow scopes
/// for its incoming operands and end_borrows that cover the phi's extended
/// borrow scope, which transitively includes any phis that use this phi.
///
/// Returns true if any changes were made.
///
/// Note: \p newPhi itself might not have Guaranteed ownership. A phi that
/// converts Guaranteed to None ownership still needs nested borrows.
///
/// Note: This may be called on partially invalid OSSA form, where multiple
/// newly created phis do not yet have a borrow scope.
bool createBorrowScopeForPhiOperands(SILPhiArgument *newPhi);

SILValue
makeGuaranteedValueAvailable(SILValue value, SILInstruction *user,
                             InstModCallbacks callbacks = InstModCallbacks());

/// Compute the liveness boundary for a guaranteed value. Returns true if no
/// uses are pointer escapes. If pointer escapes are present, the liveness
/// boundary is still valid for all known uses.
///
/// Precondition: \p value has guaranteed ownership and no reborrows. It is
/// either an "inner" guaranteed value or a simple borrow introducer whose
/// end_borrows have not yet been inserted.
bool computeGuaranteedBoundary(SILValue value,
                               PrunedLivenessBoundary &boundary);

//===----------------------------------------------------------------------===//
//                        GuaranteedOwnershipExtension
//===----------------------------------------------------------------------===//

/// Extend existing guaranteed ownership to cover new guaranteed uses that are
/// dominated by the borrow introducer.
class GuaranteedOwnershipExtension {
  // --- context
  InstructionDeleter &deleter;

  // --- analysis state
  MultiDefPrunedLiveness guaranteedLiveness;
  SSAPrunedLiveness ownedLifetime;
  SmallVector<SILBasicBlock *, 4> ownedConsumeBlocks;
  BeginBorrowInst *beginBorrow = nullptr;

public:
  GuaranteedOwnershipExtension(InstructionDeleter &deleter, SILFunction *function)
    : deleter(deleter), guaranteedLiveness(function) {}

  void clear() {
    guaranteedLiveness.clear();
    ownedLifetime.clear();
    ownedConsumeBlocks.clear();
    beginBorrow = nullptr;
  }

  /// Invalid indicates that the current guaranteed scope is insufficient, and
  /// it does not meet the precondition for scope extension.
  ///
  /// Valid indicates that the current guaranteed scope is sufficient with no
  /// transformation required.
  ///
  /// ExtendBorrow indicates that the local borrow scope can be extended without
  /// affecting the owned lifetime or introducing copies.
  ///
  /// ExtendLifetime indicates that the owned lifetime can be extended possibly
  /// requiring additional copies.
  enum Status { Invalid, Valid, ExtendBorrow, ExtendLifetime };

  /// Can the OSSA ownership of the \p parentAddress cover all uses of the \p
  /// childAddress?
  ///
  /// Precondition: \p parentAddress dominates \p childAddress
  Status checkAddressOwnership(SILValue parentAddress, SILValue childAddress);

  /// Can the OSSA scope of \p borrow cover all \p newUses?
  ///
  /// Precondition: \p borrow dominates \p newUses
  Status checkBorrowExtension(BorrowedValue borrow,
                              ArrayRef<Operand *> newUses);

  /// Can the OSSA scope of \p ownedValue cover all the guaranteed \p newUses?
  ///
  /// Precondition: \p ownedValue dominates \p newUses
  Status checkLifetimeExtension(SILValue ownedValue,
                                ArrayRef<Operand *> newUses);

  void transform(Status status);
};

//===----------------------------------------------------------------------===//
// UnreachableLifetimeCompletion
//===----------------------------------------------------------------------===//

/// Fixup OSSA before deleting an unreachable code path.
///
/// Only needed when a code path reaches a no-return function, making the
/// path now partially unreachable. Conditional branch folding requires no fixup
/// because it causes the entire path to become unreachable.
class UnreachableLifetimeCompletion {
  BasicBlockSetVector unreachableBlocks;
  InstructionSet unreachableInsts; // not including those in unreachableBlocks
  ValueSetVector incompleteValues;
  bool updatingLifetimes = false;

public:
  UnreachableLifetimeCompletion(SILFunction *function)
      : unreachableBlocks(function), unreachableInsts(function),
        incompleteValues(function) {}

  /// Record information about this unreachable instruction and return true if
  /// ends any simple OSSA lifetimes.
  ///
  /// Note: this must be called in forward order so that lifetime completion
  /// runs from the inside out.
  void visitUnreachableInst(SILInstruction *instruction);

  void visitUnreachableBlock(SILBasicBlock *block) {
    unreachableBlocks.insert(block);
  }

  /// Complete the lifetime of any value defined outside of the unreachable
  /// region that was previously destroyed in the unreachable region.
  bool completeLifetimes();
};

//===----------------------------------------------------------------------===//
//                      RAUW - Replace All Uses With...
//===----------------------------------------------------------------------===//

/// A struct that contains context shared in between different operation +
/// "ownership fixup" utilities. Please do not put actual methods on this, it is
/// meant to be composed with.
struct OwnershipFixupContext {
  Optional<InstModCallbacks> inlineCallbacks;
  InstModCallbacks &callbacks;

  // Cache the use-points for the lifetime of an inner guaranteed value (which
  // does not introduce a borrow scope) after checking validity. These will be
  // used again to extend the lifetime of the replacement value.
  SmallVector<Operand *, 8> guaranteedUsePoints;

  /// Extra state initialized by OwnershipRAUWFixupHelper::get() that we use
  /// when RAUWing addresses. This ensures we do not need to recompute this
  /// state when we perform the actual RAUW.
  struct AddressFixupContext {
    /// When determining if we need to perform an address pointer fixup, we
    /// compute all transitive address uses of oldValue. If we find that we do
    /// need this fixed up, then we will copy our interior pointer base value
    /// and use this to seed that new lifetime.
    ///
    /// FIXME: shouldn't these already be covered by guaranteedUsePoints?
    SmallVector<Operand *, 8> allAddressUsesFromOldValue;

    /// This is the interior pointer (e.g. ref_element_addr)
    /// that the new value we want to RAUW is transitively derived from and
    /// enables us to know the underlying borrowed base value that we need to
    /// lifetime extend.
    ///
    /// This is only initialized when the interior pointer has uses that must be
    /// replaced.
    AccessBase base;

    void clear() {
      allAddressUsesFromOldValue.clear();
      base = AccessBase();
    }
  };
  AddressFixupContext extraAddressFixupInfo;

  OwnershipFixupContext(InstModCallbacks &callbacks)
      : callbacks(callbacks) {}

  void clear() {
    guaranteedUsePoints.clear();
    extraAddressFixupInfo.allAddressUsesFromOldValue.clear();
    extraAddressFixupInfo.base = AccessBase();
  }

private:
  /// Helper method called to determine if we discovered we needed interior
  /// pointer fixups while simplifying.
  bool needsInteriorPointerFixups() const {
    return bool(extraAddressFixupInfo.base);
  }
};

/// A utility composed on top of OwnershipFixupContext that knows how to RAUW a
/// value or a single value instruction with a new value and then fixup
/// ownership invariants afterwards.
class OwnershipRAUWHelper {
public:
  /// Return true if \p oldValue can be replaced with \p newValue in terms of
  /// their value ownership. This checks current uses of \p oldValue for
  /// satisfying lexical lifetimes. To completely determine whether \p oldValue
  /// can be replaced as-is with it's existing uses, create an instance of
  /// OwnershipRAUWHelper and check its validity.
  static bool hasValidRAUWOwnership(SILValue oldValue, SILValue newValue,
                                    ArrayRef<Operand *> oldUses);

  static bool hasValidNonLexicalRAUWOwnership(SILValue oldValue,
                                              SILValue newValue) {
    if (oldValue->isLexical() || newValue->isLexical())
      return false;

    // Pretend that we have no uses since they are only used to check lexical
    // lifetimes.
    return hasValidRAUWOwnership(oldValue, newValue, {});
  }

private:
  OwnershipFixupContext *ctx;
  SILValue oldValue;
  // newValue is the aspirational replacement. It might not be the actual
  // replacement after SILCombine fixups (like type casting) and OSSA fixups.
  SILValue newValue;

public:
  OwnershipRAUWHelper() : ctx(nullptr) {}

  ~OwnershipRAUWHelper() { if (ctx) ctx->clear(); }

  /// Return an instance of this class if we can perform the specific RAUW
  /// operation ignoring if the types line up. Returns None otherwise.
  ///
  /// \p oldValue may be either a SingleValueInstruction or a terminator result.
  ///
  /// Precondition: If \p oldValue is a BorrowedValue that introduces a local
  /// borrow scope, then \p newValue must either be defined in the same block as
  /// \p oldValue, or it must dominate \p oldValue (rather than merely
  /// dominating its uses).
  ///
  /// DISCUSSION: We do not check that the types line up here so that we can
  /// allow for our users to transform our new value in ways that preserve
  /// ownership at \p oldValue before we perform the actual RAUW. If \p newValue
  /// is an object, any instructions in the chain of transforming instructions
  /// from \p newValue at \p oldValue's must be forwarding. If \p newValue is an
  /// address, then these transforms can only transform the address into a
  /// derived address.
  OwnershipRAUWHelper(OwnershipFixupContext &ctx, SILValue oldValue,
                      SILValue newValue);

  /// Returns true if this helper was initialized into a valid state.
  operator bool() const { return isValid(); }
  bool isValid() const { return bool(ctx) && bool(oldValue) && bool(newValue); }

  /// True if replacement requires copying the original instruction's source
  /// operand, creating a new borrow scope for that copy, then cloning the
  /// original.
  bool requiresCopyBorrowAndClone() const {
    return ctx->extraAddressFixupInfo.base;
  }

  /// Perform OSSA fixup on newValue and return a fixed-up value based that can
  /// be used to replace all uses of oldValue.
  ///
  /// This is only used by clients that must transform \p newValue, such as
  /// adding type casts, before it can be used to replace \p oldValue.
  ///
  /// \p rewrittenNewValue is only passed when the client needs to regenerate
  /// newValue after checking its RAUW validity, but before performing OSSA
  /// fixup on it.
  SILValue prepareReplacement(SILValue rewrittenNewValue = SILValue());

  /// Perform the actual RAUW--replace all uses if \p oldValue.
  ///
  /// Precondition: \p replacementValue is either invalid or has the same type
  /// as \p oldValue and is a valid OSSA replacement.
  SILBasicBlock::iterator
  perform(SILValue replacementValue = SILValue());

private:
  void invalidate() {
    ctx = nullptr;
  }

  SILValue getReplacementAddress();
};

/// Whether the provided uses lie within the current liveness of the
/// specified lexical value.
bool areUsesWithinLexicalValueLifetime(SILValue, ArrayRef<Operand *>);

/// A utility composed ontop of OwnershipFixupContext that knows how to replace
/// a single use of a value with another value with a different ownership. We
/// allow for the values to have different types.
///
/// Precondition: if \p use ends a borrow scope, then \p newValue dominates the
/// BorrowedValue that begins the scope.
///
/// NOTE: When not in OSSA, this just performs a normal set use, so this code is
/// safe to use with all code.
class OwnershipReplaceSingleUseHelper {
  OwnershipFixupContext *ctx;
  Operand *use;
  SILValue newValue;

public:
  OwnershipReplaceSingleUseHelper()
      : ctx(nullptr), use(nullptr), newValue(nullptr) {}

  /// Return an instance of this class if we support replacing \p use->get()
  /// with \p newValue.
  ///
  /// NOTE: For now we only support objects, not addresses so addresses will
  /// always yield an invalid helper.
  OwnershipReplaceSingleUseHelper(OwnershipFixupContext &ctx, Operand *use,
                                  SILValue newValue);

  ~OwnershipReplaceSingleUseHelper() { if (ctx) ctx->clear(); }

  /// Returns true if this helper was initialized into a valid state.
  operator bool() const { return isValid(); }
  bool isValid() const { return bool(ctx) && bool(use) && bool(newValue); }

  /// Perform the actual RAUW.
  SILBasicBlock::iterator perform();

private:
  void invalidate() {
    ctx->clear();
    ctx = nullptr;
  }
};

/// An abstraction over LoadInst/LoadBorrowInst so one can handle both types of
/// load using common code.
struct LoadOperation {
  llvm::PointerUnion<LoadInst *, LoadBorrowInst *> value;

  LoadOperation() : value() {}
  LoadOperation(SILInstruction *input) : value(nullptr) {
    if (auto *li = dyn_cast<LoadInst>(input)) {
      value = li;
      return;
    }

    if (auto *lbi = dyn_cast<LoadBorrowInst>(input)) {
      value = lbi;
      return;
    }
  }

  explicit operator bool() const { return !value.isNull(); }

  SingleValueInstruction *getLoadInst() const {
    if (value.is<LoadInst *>())
      return value.get<LoadInst *>();
    return value.get<LoadBorrowInst *>();
  }

  SingleValueInstruction *operator*() const { return getLoadInst(); }

  const SingleValueInstruction *operator->() const { return getLoadInst(); }

  SingleValueInstruction *operator->() { return getLoadInst(); }

  SILValue getOperand() const {
    if (value.is<LoadInst *>())
      return value.get<LoadInst *>()->getOperand();
    return value.get<LoadBorrowInst *>()->getOperand();
  }

  /// Return the ownership qualifier of the underlying load if we have a load or
  /// None if we have a load_borrow.
  ///
  /// TODO: Rather than use an optional here, we should include an invalid
  /// representation in LoadOwnershipQualifier.
  Optional<LoadOwnershipQualifier> getOwnershipQualifier() const {
    if (auto *lbi = value.dyn_cast<LoadBorrowInst *>()) {
      return None;
    }

    return value.get<LoadInst *>()->getOwnershipQualifier();
  }
};

/// Extend the store_borrow \p sbi's scope such that it encloses \p newUsers.
bool extendStoreBorrow(StoreBorrowInst *sbi,
                       SmallVectorImpl<Operand *> &newUses,
                       DeadEndBlocks *deadEndBlocks,
                       InstModCallbacks callbacks = InstModCallbacks());

} // namespace swift

#endif
