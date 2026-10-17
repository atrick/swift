//===--- BoxAccessSummary.h - Accesses to a box during a call ---*- C++ -*-===//
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
///
/// Summarize the accesses to the storage of a box that may happen, and that
/// must happen, during a call. The box may be passed to the call directly, as
/// a local function's capture, or captured by a closure that is passed to the
/// call or is the callee.
///
/// A box is used for a variable captured by a closure that may escape. Its
/// accesses are dynamically enforced unless AccessEnforcementSelection can
/// prove that static enforcement is equivalent. Static diagnostics must not
/// reject code that dynamic enforcement would accept, so a conflict between an
/// access in progress at the call and an access during the call can only be
/// diagnosed statically if the access during the call happens on every path
/// through the call. These are the "must" accesses.
///
//===----------------------------------------------------------------------===//

#ifndef SWIFT_SILOPTIMIZER_UTILS_BOXACCESSSUMMARY_H
#define SWIFT_SILOPTIMIZER_UTILS_BOXACCESSSUMMARY_H

#include "swift/SIL/ApplySite.h"
#include "swift/SIL/SILFunction.h"
#include "llvm/ADT/DenseMap.h"
#include <optional>

namespace swift {

class AccessSummaryAnalysis;

/// The accesses to a box's storage during a call. A modify includes any
/// access other than a read.
struct BoxAccesses {
  /// The strongest access that may happen during the call.
  std::optional<SILAccessKind> mayAccess;

  /// The strongest access that happens on every path through the call, and
  /// the location of that access. Only accesses to the whole variable are
  /// included.
  std::optional<SILAccessKind> mustAccess;
  SILLocation mustLoc = SILLocation::invalid();

  bool mayConflictWith(SILAccessKind kind) const {
    return mayAccess && conflicts(*mayAccess, kind);
  }

  bool mustConflictWith(SILAccessKind kind) const {
    return mustAccess && conflicts(*mustAccess, kind);
  }

  static bool conflicts(SILAccessKind a, SILAccessKind b) {
    return a != SILAccessKind::Read || b != SILAccessKind::Read;
  }

  /// Merge accesses that may happen.
  void mergeMay(const BoxAccesses &other);

  /// Merge accesses that happen on every path.
  void mergeMust(const BoxAccesses &other);

  void addMay(SILAccessKind kind);
  void addMust(SILAccessKind kind, SILLocation loc);
};

/// Whether a closure argument may be invoked during a call, and whether it is
/// invoked on every path through the call.
struct ClosureInvocation {
  bool mayInvoke = false;
  bool mustInvoke = false;
};

class BoxAccessSummary {
  AccessSummaryAnalysis *ASA;

  llvm::DenseMap<SILFunctionArgument *, std::optional<BoxAccesses>>
      boxArgumentCache;
  llvm::DenseMap<SILFunctionArgument *, std::optional<ClosureInvocation>>
      closureArgumentCache;

public:
  BoxAccessSummary(AccessSummaryAnalysis *ASA);
  ~BoxAccessSummary();

  /// Return the accesses to the storage of the box whose reference root is
  /// \p boxRoot during \p apply, through each argument or capture that passes
  /// the box itself. Addresses projected from the box are not included.
  BoxAccesses getAccessesDuringCall(FullApplySite apply, SILValue boxRoot);

  /// The accesses to box argument \p arg during a call of its function.
  BoxAccesses getBoxArgumentAccesses(SILFunctionArgument *arg);

  /// Whether closure argument \p arg is invoked during a call of its
  /// function.
  ClosureInvocation getClosureArgumentInvocation(SILFunctionArgument *arg);

  /// Return true if \p inst executes on every path through its function: no
  /// path from the entry exits the function, including by trapping, or loops
  /// forever without executing \p inst. A call that may not return is assumed
  /// to return unless it is known not to.
  bool executesOnAllPaths(SILInstruction *inst);

private:
  BoxAccesses computeBoxArgumentAccesses(SILFunctionArgument *arg);
  ClosureInvocation computeClosureArgumentInvocation(SILFunctionArgument *arg);

  /// Add the accesses to the box whose reference root is \p boxRoot by the
  /// closures that \p closureValue may refer to, given how they are invoked.
  void addClosureAccesses(SILValue closureValue, SILValue boxRoot,
                          ClosureInvocation invocation, BoxAccesses &result);
};

} // end namespace swift

#endif
