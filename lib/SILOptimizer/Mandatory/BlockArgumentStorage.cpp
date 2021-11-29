//===--- BlockArgumentStorage.cpp Block Argument Storage Optimizer --------===//
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
/// BlockArgumentStorageOptimizer implements an analysis used by AddressLowering
/// to reuse storage across block arguments.
///
/// TODO: This approach uses on-the-fly liveness discovery for all incoming
/// values at once. It requires no storage for liveness. Hopefully this is
/// sufficient for -Onone. At -O, we should explore implementing strong phi
/// elimination. However, that depends the ability to perform interference
/// checks between arbitrary storage locations, which requires computing and
/// storing liveness per-storage location.
///
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "address-lowering"
#include "BlockArgumentStorage.h"

using namespace swift;

static Operand *getBranchOperand(SILBasicBlock *predBB, unsigned argIdx) {
  return &cast<BranchInst>(predBB->getTerminator())->getAllOperands()[argIdx];
};

// Process an incoming value.
//
// Fully compute liveness from this use operand. Return true if no interference
// was detected along the way.
bool BlockArgumentStorageOptimizer::computeIncomingLiveness(
    Operand *useOper, SILBasicBlock *defBB) {
  bool noInterference = true;

  auto visitLiveBlock = [&](SILBasicBlock *liveBB) {
    if (blackBlocks.count(liveBB))
      noInterference = false;
    else if (greyBlocks.insert(liveBB).second && liveBB != defBB)
      liveBBWorklist.push_back(liveBB);
  };

  assert(liveBBWorklist.empty());

  visitLiveBlock(useOper->getUser()->getParent());

  while (!liveBBWorklist.empty()) {
    auto *succBB = liveBBWorklist.pop_back_val();
    for (auto *predBB : succBB->getPredecessorBlocks())
      visitLiveBlock(predBB);
  }
  return noInterference;
}

bool BlockArgumentStorageOptimizer::
hasCoalescedOperand(SILInstruction *defInst) {
  for (Operand &oper : defInst->getAllOperands()) {
    if (valueStorageMap.isNonBranchUseProjection(&oper))
      return true;
  }
  return false;
}

/// Finds all non-interfering block arguments and adds them to the result's
/// projectedBBArgs. The algorithm can be described in the abstract as follows
/// (assuming no critical edges):
///
/// Blocks are marked white, grey, or black.
///
/// All blocks start white.
/// Set all predecessor blocks black.
///
/// For each incoming value:
///
///   Mark the current predecessor white (from black). If any other source is
///   live out of that predecessor, then this predecessor block will be marked
///   black when we process that other incoming value.
///
///   For all uses of the current incoming value:
///     Scan the CFG backward following predecessors.
///     If the current block is:
///       White: mark it grey and continue scanning.
///       Grey: stop scanning and continue with the next use.
///       Black: record interference, stop scanning, continue with the next use.
///
///   If no black blocks were reached, record this incoming value as a valid
///   projection.
///
///   Mark all grey blocks black. This will mark the incoming predecessor black
///   again, along with any other blocks in which the incoming value is
///   live-out.
///
/// In the end, we have a set of non-interfering incoming values that can reuse
/// the bbArg's storage.
///
/// Note: This simple description of the algorithm assumes that none of the
/// incoming values nor their uses have been coalesced with storage via
/// projections. If they have, then liveness would need to consider all values
/// associated with that storage. For now, we simply deal with scalar values.
void BlockArgumentStorageOptimizer::findNonInterferingBlockArguments() {
  SILBasicBlock *succBB = bbArg->getParent();
  for (auto *predBB : succBB->getPredecessorBlocks()) {
    // Block arguments on critical edges are disallowed.
    assert(predBB->getSingleSuccessorBlock() == succBB);
    blackBlocks.insert(predBB);
  }
  SmallVector<SILValue, 2> storageValues;
  unsigned argIdx = bbArg->getIndex();
  for (auto *incomingPred : succBB->getPredecessorBlocks()) {
    Operand *incomingOper = getBranchOperand(incomingPred, argIdx);
    SILValue incomingVal = incomingOper->get();
    auto &incomingStorage = valueStorageMap.getStorage(incomingVal);

    // If the incoming use is pre-allocated it can't be coalesced.
    // This also handles incoming values that are already coalesced with
    // another use.
    if (incomingStorage.isProjection())
      continue;

    // Make sure that the incomingVal is not coalesced with any of its operands.
    // 
    // TODO: handle incomingValues that project onto their operands by
    // recursively finding the set of value definitions and their dominating
    // defBB instead of incomingVal->getParentBlock().
    if (auto *defInst = incomingVal->getDefiningInstruction()) {
      assert(incomingStorage.isAllocated());
      // Don't coalesce an incoming value unless it's storage is from a stack
      // allocation, which can be replaced.
      if (!isa<AllocStackInst>(incomingStorage.storageAddress))
        continue;

      if (hasCoalescedOperand(defInst))
        continue;
    } else {
      // For now, don't attempt to coalesce other block arguments. Indirect
      // function arguments were replaced by loads.
      assert(!isa<SILFunctionArgument>(incomingVal));
      continue;
    }
    // For now, just stop liveness traversal at defBB.
    SILBasicBlock *defBB = incomingVal->getParentBlock();

    bool erased = blackBlocks.erase(incomingPred);
    (void)erased;
    assert(erased);

    bool noInterference = true;
    // Continue marking live blocks even after detecting an interference so that
    // the live set is complete when evaluating subsequent incoming values.
    for (auto *use : incomingVal->getUses()) {
      // TODO: recursively check liveness by following uses across use
      // projections instead of just the immediate use.
      noInterference &= computeIncomingLiveness(use, defBB);
    }
    if (noInterference)
      result.projectedBBArgs.push_back(incomingOper);

    // FIXME: If the incoming def is a copy, transitively consider coalescing
    // the source of the copy by continuing the same liveness traversal,
    // treating the copy as a new use.

    blackBlocks.insert(greyBlocks.begin(), greyBlocks.end());
    assert(blackBlocks.count(incomingPred));
    greyBlocks.clear();
  }
}

// Process this bbArg, recording in the Result which incoming values can reuse
// storage with the argument itself.
BlockArgumentStorageOptimizer::Result &&
BlockArgumentStorageOptimizer::computeArgumentProjections() && {
  assert(!valueStorageMap.getStorage(bbArg).isDefProjection);

  // If this block argument is already "coalesced", don't attempt to merge its
  // live range with its incoming values.
  //
  // TODO: recursively check liveness of use projections to allow use
  // projections across block boundaries.
  if (valueStorageMap.getStorage(bbArg).isUseProjection)
    return std::move(result);

  // FIXME: remove the following check once criticalEdges are verified and
  // CondBranch never has args.
  if (llvm::any_of(bbArg->getParent()->getPredecessorBlocks(),
                   [](SILBasicBlock *predBB) {
                     return !isa<BranchInst>(predBB->getTerminator());
                   })) {
    return std::move(result);
  }
  // The single incoming value case always projects storage.
  if (auto *predBB = bbArg->getParent()->getSinglePredecessorBlock()) {
    result.projectedBBArgs.push_back(
        getBranchOperand(predBB, bbArg->getIndex()));
    return std::move(result);
  }
  findNonInterferingBlockArguments();
  return std::move(result);
}
