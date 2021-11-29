//===--- BlockArgumentStorage.h - Block Argument Storage Optimizer --------===//
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
///
/// This file defines BlockArgumentStorageOptimizer, a utility for use with the
/// mandatory AddressLowering pass.
///
//===----------------------------------------------------------------------===//

#include "AddressLowering.h"
#include "swift/SIL/SILArgument.h"
#include "swift/SIL/SILValue.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace swift {
using llvm::SmallVector;

/// An analysis used by AddressLowering to reuse storage across block arguments.
///
/// Populates Result::projectedBBArgs with all inputs to bbArg that can reuse
/// the argument's storage.
class BlockArgumentStorageOptimizer {
  /// The result is simply an array of phi operands whose storage can be
  /// coalesced with the BlockArgument storage.
  class Result {
    friend class BlockArgumentStorageOptimizer;
    SmallVector<Operand *, 4> projectedBBArgs;

    struct GetOper {
      SILValue operator()(Operand *oper) const { return oper->get(); }
    };

    Result(const Result &) = delete;
    Result &operator=(const Result &) = delete;

  public:
    Result() = default;
    Result(Result &&) = default;
    Result &operator=(Result &&) = default;

    ArrayRef<Operand *> getArgumentProjections() const {
      return projectedBBArgs;
    }

    // TODO: LLVM needs a map_range.
    iterator_range<
        llvm::mapped_iterator<ArrayRef<Operand *>::iterator, GetOper>>
    getIncomingValueRange() const {
      return make_range(
          llvm::map_iterator(getArgumentProjections().begin(), GetOper()),
          llvm::map_iterator(getArgumentProjections().end(), GetOper()));
    }

    void clear() { projectedBBArgs.clear(); }
  };

  const ValueStorageMap &valueStorageMap;

  SILPhiArgument *bbArg;
  Result result;

  // Working state for this bbArg.
  //
  // TODO: These are possible candidates for bitsets since we're reusing storage
  // across multiple uses and want to perform a fast union.
  SmallPtrSet<SILBasicBlock *, 16> blackBlocks;
  SmallPtrSet<SILBasicBlock *, 16> greyBlocks;

  // Working state per-incoming-value.
  SmallVector<SILBasicBlock *, 16> liveBBWorklist;

public:
  BlockArgumentStorageOptimizer(const ValueStorageMap &valueStorageMap,
                                SILPhiArgument *bbArg)
      : valueStorageMap(valueStorageMap), bbArg(bbArg) {}

  Result &&computeArgumentProjections() &&;

protected:
  bool computeIncomingLiveness(Operand *useOper, SILBasicBlock *defBB);
  bool hasCoalescedOperand(SILInstruction *defInst);
  void findNonInterferingBlockArguments();
};

} // namespace swift
