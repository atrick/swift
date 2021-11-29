//===--- AddressLowering.h - Lower SIL address-only types. ----------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2018 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

#include "swift/SIL/SILArgument.h"
#include "swift/SIL/SILValue.h"
#include "llvm/ADT/DenseMap.h"

namespace swift {

/// Track a value's storage. After allocation, ValueStorage either has a valid
/// storage address, or indicates that it is the projection of another value's
/// storage. A projection may either be from the value's use (a use projection),
/// or from an operand of the value's defining instruction (a def projection).
///
/// After rewriting, all ValueStorage entries have a valid storage address.
///
/// To express projections, ValueStorage refers to the storage of other
/// values. Consequently, values that have storage cannot be removed from SIL or
/// from the storage map until rewriting is complete. However, since we don't
/// directly store references to any SIL entities, such as Operands or
/// SILValues, mapped values can be replaced as long as the original value had
/// no arguments (e.g. block arguments).
struct ValueStorage {
  enum : uint32_t { InvalidID = uint32_t(~0) };
  enum : uint16_t { InvalidOper = uint16_t(~0) };

  /// The final address of this storage unit after rewriting the SIL.
  /// For values linked to their own storage, this is set during storage
  /// allocation. For projections, it is only set after instruction rewriting.
  SILValue storageAddress;

  /// Refer to another storage projection for isUseProjection ||
  /// isDefProjection.
  uint32_t projectedStorageID;

  /// Identifies the operand index of a composed aggregate. Invalid for
  /// non-projections, def projections, and branch use projections.
  uint16_t projectedOperandNum;

  /// Flags.
  unsigned isUseProjection : 1;
  unsigned isDefProjection : 1;
  unsigned isRewritten : 1;

  ValueStorage() { clear(); }

  void clear() {
    storageAddress = SILValue();
    projectedStorageID = InvalidID;
    projectedOperandNum = InvalidOper;
    isUseProjection = false;
    isDefProjection = false;
    isRewritten = false;
  }

  bool isAllocated() const {
    return storageAddress || isUseProjection || isDefProjection;
  }

  bool isProjection() const { return isUseProjection || isDefProjection; }

  bool isBranchUseProjection() const {
    return isUseProjection && projectedOperandNum == InvalidOper;
  }

  void markRewritten() {
    assert(storageAddress);
    isRewritten = true;
  }
};

/// Map each opaque/resilient SILValue to its abstract storage.
/// O(1) membership test.
/// O(n) iteration, and guaranteed RPO order.
///
/// Mapped values are expected to be created in a single RPO pass. "erase" is
/// unsupported. Values must be replaced using 'replaceValue()'.
class ValueStorageMap {
  struct ValueStoragePair {
    SILValue value;
    ValueStorage storage;
    ValueStoragePair(SILValue v, ValueStorage s) : value(v), storage(s) {}
  };
  typedef std::vector<ValueStoragePair> ValueVector;
  // Hash of values to ValueVector indices.
  typedef llvm::DenseMap<SILValue, unsigned> ValueHashMap;

  ValueVector valueVector;
  ValueHashMap valueHashMap;

public:
  bool empty() const { return valueVector.empty(); }

  void clear() {
    valueVector.clear();
    valueHashMap.clear();
  }

  /// Iterate over value storage in RPO order. Once we begin erasing
  /// instructions, some entries could become invalid. ValueStorage validity can
  /// be checked with valueStorageMap.contains(value).
  ValueVector::iterator begin() { return valueVector.begin(); }

  ValueVector::iterator end() { return valueVector.end(); }

  ValueVector::reverse_iterator rbegin() { return valueVector.rbegin(); }

  ValueVector::reverse_iterator rend() { return valueVector.rend(); }

  bool contains(SILValue value) const {
    return valueHashMap.find(value) != valueHashMap.end();
  }

  unsigned getOrdinal(SILValue value) const {
    auto hashIter = valueHashMap.find(value);
    assert(hashIter != valueHashMap.end() && "Missing SILValue");
    return hashIter->second;
  }

  ValueStorage &getStorage(SILValue value) {
    return valueVector[getOrdinal(value)].storage;
  }
  const ValueStorage &getStorage(SILValue value) const {
    return valueVector[getOrdinal(value)].storage;
  }

  /// Insert a value in the map, creating a ValueStorage object for it. This
  /// must be called in RPO order.
  ValueStorage &insertValue(SILValue value);

  /// Replace a value that is mapped to storage with another value. This allows
  /// limited rewritting of original address-only values. For example, block
  /// arguments can be replaced with fake loads in order to rewrite their
  /// corresponding terminator.
  void replaceValue(SILValue oldValue, SILValue newValue);

  /// Given storage for a projection, return the projected storage by following
  /// one level of storage projection. The returned storage may also be a
  /// projection.
  ValueStoragePair &getProjectedStorage(ValueStorage &storage) {
    assert(storage.isUseProjection || storage.isDefProjection);
    return valueVector[storage.projectedStorageID];
  }

  /// Return the non-projection storage that the given storage ultimately refers
  /// to by following all projections.
  ValueStorage &getNonProjectionStorage(ValueStorage &storage) {
    if (storage.isDefProjection || storage.isUseProjection)
      return getNonProjectionStorage(getProjectedStorage(storage).storage);

    return storage;
  }

  /// Return the non-projection storage that the given storage ultimately refers
  /// to by following all projections.
  ValueStorage &getNonProjectionStorage(SILValue value) {
    return getNonProjectionStorage(getStorage(value));
  }

  /// Record a storage projection from the source of the given operand into its
  /// use (e.g. struct_extract, tuple_extract project storage from their
  /// source).
  void setExtractedDefOperand(Operand *oper) {
    auto *extractInst = cast<SingleValueInstruction>(oper->getUser());
    auto &storage = getStorage(extractInst);
    storage.projectedStorageID = getOrdinal(oper->get());
    storage.isDefProjection = true;
  }

  /// Record a storage projection from the source of the given operand into its
  /// borrowing use. e.g. copy->store. Because the copy only has one immediate
  /// use, it can copy directly into memory.
  void setBorrowedDefOperand(Operand *oper) {
    setExtractedDefOperand(oper);
  }

  // Record a storage projection from a terminator (switch_enum) to a block
  // argument that is a subobject of the given operand of the terminator.
  void setExtractedBlockArg(SILValue singlePredVal, SILPhiArgument *bbArg) {
    auto &storage = getStorage(bbArg);
    storage.projectedStorageID = getOrdinal(singlePredVal);
    storage.isDefProjection = true;
  }

  /// Record a storage projection from the use of the given operand into the
  /// operand's source. (e.g. Any value used by a struct, tuple, or enum may
  /// project storage from its use).
  void setComposingUseProjection(Operand *oper);

  // The terminator argument can be deduced from the block argument that we
  // project from.
  void setBranchUseProjection(Operand *oper, SILPhiArgument *bbArg);

  /// Return true if the given operand projects storage from its use into its
  /// source.
  bool isNonBranchUseProjection(Operand *oper) const;

#ifndef NDEBUG
  void dump();
#endif
};

} // namespace swift
