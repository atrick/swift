//===--- BorrowUtils.swift - Utilities for borrow scopes ------------------===//
//
// This source file is part of the Swift.org open source project
//
// Copyright (c) 2014 - 2023 Apple Inc. and the Swift project authors
// Licensed under Apache License v2.0 with Runtime Library Exception
//
// See https://swift.org/LICENSE.txt for license information
// See https://swift.org/CONTRIBUTORS.txt for the list of Swift project authors
//
//===----------------------------------------------------------------------===//

import SIL

/// A scoped instruction that borrows one or more operands.
///
/// If this instruction produces a borrowed value, then
/// BeginBorrowValue(resultOf: self) != nil.
///
/// This does not include instructions like `apply` and `try_apply` that
/// instantaneously borrow a value from the caller.
///
/// This does not include `load_borrow` because it borrows a memory
/// location, not the value of its operand.
///
/// Note: This must handle all instructions with a .borrow operand ownership.
///
/// TODO: replace BorrowIntroducingInstruction
///
/// TODO: Add non-escaping MarkDependence.
enum BorrowingInstruction : CustomStringConvertible, Hashable {
  case beginBorrow(BeginBorrowInst)
  case storeBorrow(StoreBorrowInst)
  case beginApply(BeginApplyInst)
  case partialApply(PartialApplyInst)
  case startAsyncLet(BuiltinInst)

  init?(_ inst: Instruction) {
    switch inst {
    case let bbi as BeginBorrowInst: self = .beginBorrow(bbi)
    case let sbi as StoreBorrowInst: self = .storeBorrow(sbi)
    case let bai as BeginApplyInst: self = .beginApply(bai)
    case let pai as PartialApplyInst where pai.isOnStack:
      self = .partialApply(pai)
    case let bi as BuiltinInst
           where bi.id == .StartAsyncLetWithLocalBuffer:
      self = .startAsyncLet(bi)
    default:
      return nil
    }
  }
  
  var instruction: Instruction {
    switch self {
    case .beginBorrow(let bbi): return bbi
    case .storeBorrow(let sbi): return sbi
    case .beginApply(let bai): return bai
    case .partialApply(let pai): return pai
    case .startAsyncLet(let bi): return bi
    }
  }

  /// Visit the operands that end the local borrow scope.
  ///
  /// Note: When this instruction's result is BeginBorrowValue the
  /// scopeEndingOperand may include reborrows. To find all uses that
  /// contribute to liveness, the caller needs to determine whether an
  /// incoming value dominates or is consumed by an outer adjacent
  /// phi. See InteriorLiveness.
  ///
  /// TODO: to hande reborrow-extended uses migrate ExtendedLiveness
  /// to SwiftCompilerSources.
  ///
  /// TODO: For instructions that are not a BeginBorrowValue, verify
  /// that scope ending instructions exist on all paths. These
  /// instructions should be complete after SILGen and never cloned to
  /// produce phis.
  func visitScopeEndingOperands(_ context: Context,
                                visitor: @escaping (Operand) -> WalkResult)
  -> WalkResult {
    switch self {
    case .beginBorrow, .storeBorrow:
      let svi = instruction as! SingleValueInstruction
      return svi.uses.walk {
        if $0.instruction is EndBorrowInst { return visitor($0) }
        return .continueWalk
      }
    case .beginApply(let bai):
      return bai.token.uses.walk { return visitor($0) }
    case .partialApply(let pai):
      return visitForwardedUses(introducer: pai, context) {
        switch $0 {
        case let .operand(operand):
          if operand.endsLifetime {
            return visitor(operand)
          }
          return .continueWalk
        case let .deadValue(_, operand):
          if let operand = operand {
            assert(!operand.endsLifetime,
                   "a dead forwarding instruction cannot end a lifetime")
          }
          return .continueWalk
        }
      }
    case .startAsyncLet(let builtin):
      return builtin.uses.walk {
        if let builtinUser = $0.instruction as? BuiltinInst,
          builtinUser.id == .EndAsyncLetLifetime {
          return visitor($0)
        }
        return .continueWalk
      }
    }
  }

  var description: String { instruction.description }
}

/// A value that introduces a borrow scope:
/// begin_borrow, load_borrow, reborrow, guaranteed function argument.
///
/// If the value introduces a local scope, then that scope is
/// terminated by scope ending operands. Function arguments do not
/// introduce a local scope because the caller owns the scope.
///
/// If the value is a begin_apply result, then it may be the token or
/// one of the yielded values. In any case, the scope ending operands
/// are on the end_apply or abort_apply intructions that use the
/// token.
///
/// Note: equivalent to C++ BorrowedValue, but also handles begin_apply.
enum BeginBorrowValue {
  case beginBorrow(BeginBorrowInst)
  case loadBorrow(LoadBorrowInst)
  case beginApply(Value)
  case functionArgument(FunctionArgument)
  case reborrow(Phi)

  init?(_ value: Value) {
    switch value {
    case let bbi as BeginBorrowInst: self = .beginBorrow(bbi)
    case let lbi as LoadBorrowInst: self = .loadBorrow(lbi)
    case let arg as FunctionArgument: self = .functionArgument(arg)
    case let arg as Argument where arg.isReborrow:
      self = .reborrow(Phi(arg)!)
    default:
      if value.definingInstruction is BeginApplyInst {
        self = .beginApply(value)
        break
      }
      return nil
    }
  }
  
  var value: Value {
    switch self {
    case .beginBorrow(let bbi): return bbi
    case .loadBorrow(let lbi): return lbi
    case .beginApply(let v): return v
    case .functionArgument(let arg): return arg
    case .reborrow(let phi): return phi.value
    }
  }

  init?(using operand: Operand) {
    switch operand.instruction {
    case is BeginBorrowInst, is LoadBorrowInst:
      let inst = operand.instruction as! SingleValueInstruction
      self = BeginBorrowValue(inst)!
    case is BranchInst:
      guard let phi = Phi(using: operand) else { return nil }
      guard phi.isReborrow else { return nil }
      self = .reborrow(phi)
    default:
      return nil
    }
  }

  init?(resultOf borrowInstruction: BorrowingInstruction) {
    switch borrowInstruction {
    case let .beginBorrow(beginBorrow):
      self = BeginBorrowValue(beginBorrow)!
    case let .beginApply(beginApply):
      self = BeginBorrowValue(beginApply.token)!
    case .storeBorrow, .partialApply, .startAsyncLet:
      return nil
    }
  }

  var hasLocalScope: Bool {
    switch self {
    case .beginBorrow, .loadBorrow, .beginApply, .reborrow:
      return true
    case .functionArgument:
      return false
    }
  }

  // Return the value borrowed by begin_borrow or address borrowed by
  // load_borrow.
  //
  // Return nil for begin_apply and reborrow, which need special handling.
  var baseOperand: Operand? {
    switch self {
    case let .beginBorrow(beginBorrow):
    return beginBorrow.operand
    case let .loadBorrow(loadBorrow):
      return loadBorrow.operand
    case .beginApply, .functionArgument, .reborrow:
      return nil
    }
  }

  /// The EndBorrows, reborrows (phis), and consumes (of closures)
  /// that end the local borrow scope. Empty if hasLocalScope is false.
  var scopeEndingOperands: LazyFilterSequence<UseList> {
    switch self {
    case let .beginApply(value):
      (value.definingInstruction as! BeginApplyInst).token.uses.endingLifetime
    default:
      value.uses.endingLifetime
    }
  }
}

/// Find the borrow introducers for `value`. This gives you a set of
/// OSSA lifetimes that directly include `value`. If `value` is owned,
/// or introduces a borrow scope, then `value` is the single
/// introducer for itself.
///
/// Example:                                       // introducers:
///                                                // ~~~~~~~~~~~~
///   bb0(%0 : @owned $Class,                      // %0
///       %1 : @guaranteed $Class):                // %1
///     %borrow0 = begin_borrow %0                 // %borrow0
///     %pair = struct $Pair(%borrow0, %1)         // %borrow0, %1
///     %first = struct_extract %pair              // %borrow0, %1
///     %field = ref_element_addr %first           // (none)
///     %load = load_borrow %field : $*C           // %load
func gatherBorrowIntroducers(for value: Value,
                             in borrowIntroducers: inout Stack<Value>,
                             _ context: Context) {

  // Cache introducers across multiple instances of BorrowIntroducers.
  var cache = BorrowIntroducers.Cache(context)
  defer { cache.deinitialize() }
  BorrowIntroducers.gather(for: value, in: &borrowIntroducers,
                           &cache, context)
}

private struct BorrowIntroducers {
  typealias CachedIntroducers = SingleInlineArray<Value>
  struct Cache {
    // Cache the introducers already found for each SILValue.
    var valueIntroducers: Dictionary<HashableValue, CachedIntroducers>
    // Record recursively followed phis to avoid infinite cycles.
    // Phis are removed from this set when they are cached.
    var pendingPhis: ValueSet

    init(_ context: Context) {
      valueIntroducers = Dictionary<HashableValue, CachedIntroducers>()
      pendingPhis = ValueSet(context)
    }

    mutating func deinitialize() {
      pendingPhis.deinitialize()
    }
  }
  
  let context: Context
  // BorrowIntroducers instances are recursively nested in order to
  // find outer adjacent phis. Each instance populates a separate
  // 'introducers' set. The same value may occur in 'introducers' at
  // multiple levels. Each instance, therefore, needs a separate
  // introducer set to avoid adding duplicates.
  var visitedIntroducers: Set<HashableValue> = Set()

  static func gather(for value: Value, in introducers: inout Stack<Value>,
                     _ cache: inout Cache, _ context: Context) {
    var borrowIntroducers = BorrowIntroducers(context: context)
    borrowIntroducers.gather(for: value, in: &introducers, &cache)
  }

  private mutating func push(_ introducer: Value,
    introducers: inout Stack<Value>) {
    if visitedIntroducers.insert(introducer.hashable).inserted {
      introducers.push(introducer)
    }
  }

  private mutating func push<S: Sequence>(contentsOf other: S,
    introducers: inout Stack<Value>) where S.Element == Value {
    for elem in other {
      push(elem, introducers: &introducers)
    }
  }

  // This is the identity function (i.e. just adds `value` to `introducers`)
  // when:
  // - `value` is owned
  // - `value` introduces a borrow scope (begin_borrow, load_borrow, reborrow)
  //
  // Otherwise recurse up the use-def chain to find all introducers.
  private mutating func gather(for value: Value,
                               in introducers: inout Stack<Value>,
                               _ cache: inout Cache) {
    // Check if this value's introducers have already been added to
    // 'introducers' to avoid duplicates and avoid exponential
    // recursion on aggregates.
    if let cachedIntroducers = cache.valueIntroducers[value.hashable] {
      cachedIntroducers.forEach { push($0, introducers: &introducers) }
      return
    }
    introducers.withMarker(
      pushElements: { introducers in
        gatherUncached(for: value, in: &introducers, &cache)
      },
      withNewElements: { newIntroducers in
        { cachedIntroducers in
          newIntroducers.forEach { cachedIntroducers.push($0) }
        }(&cache.valueIntroducers[value.hashable, default: CachedIntroducers()])
      })
  }

  private mutating func gatherUncached(for value: Value,
                                       in introducers: inout Stack<Value>,
                                       _ cache: inout Cache) {
    switch value.ownership {
    case .none, .unowned:
      return

    case .owned:
      push(value, introducers: &introducers);
      return

    case .guaranteed:
      break
    }
    // BeginBorrowedValue handles the initial scope introducers: begin_borrow,
    // load_borrow, & reborrow.
    if BeginBorrowValue(value) != nil {
      push(value, introducers: &introducers)
      return
    }
    // Handle guaranteed forwarding phis
    if let phi = Phi(value) {
      gather(forPhi: phi, in: &introducers, &cache)
      return
    }
    // Recurse through guaranteed forwarding non-phi instructions.
    guard let forwardingInst = value.forwardingInstruction else {
      fatalError("guaranteed value must be forwarding")
    }
    for operand in forwardingInst.forwardedOperands {
      if operand.value.ownership == .guaranteed {
        gather(for: operand.value, in: &introducers, &cache);
      }
    }
  }

  // Find the introducers of a guaranteed forwarding phi's borrow
  // scope. The introducers are either dominating values or reborrows
  // in the same block as the forwarding phi.
  //
  // Recurse along the use-def phi web until a begin_borrow is reached. At each
  // level, find the outer-adjacent phi, if one exists, otherwise return the
  // dominating definition.
  //
  // Example:
  //
  //     bb1(%reborrow_1 : @guaranteed)
  //         %field = struct_extract %reborrow_1
  //         br bb2(%reborrow_1, %field)
  //     bb2(%reborrow_2 : @guaranteed, %forward_2 : @guaranteed)
  //         end_borrow %reborrow_2
  //
  // Calling `gather(forPhi: %forward_2)`
  // recursively computes these introducers:
  //
  //    %field is the only value incoming to %forward_2.
  //
  //    %field is introduced by %reborrow_1 via
  //    gather(for: %field).
  //
  //    %reborrow_1 is remapped to %reborrow_2 in bb2 via
  //    mapToPhi(bb1, %reborrow_1)).
  //
  //    %reborrow_2 is returned.
  //
  private mutating func gather(forPhi phi: Phi,
                               in introducers: inout Stack<Value>,
                               _ cache: inout Cache) {
    // Phi cycles are skipped. They cannot contribute any new introducer.
    if !cache.pendingPhis.insert(phi.value) {
      return
    }
    for (pred, value) in zip(phi.predecessors, phi.incomingValues) {
      // Each phi operand requires a new introducer list and visited
      // values set. These values will be remapped to successor phis
      // before adding them to the caller's introducer list. It may be
      // necessary to revisit a value that was already visited by the
      // caller before remapping to phis.
      var incomingIntroducers = Stack<Value>(context)
      defer {
        incomingIntroducers.deinitialize()
      }
      BorrowIntroducers.gather(for: value, in: &incomingIntroducers,
                               &cache, context)
      // Map the incoming introducers to an outer-adjacent phi if one exists.
      push(contentsOf: mapToPhi(predecessor: pred,
                                incomingValues: incomingIntroducers),
           introducers: &introducers)
    }
    // Remove this phi from the pending set. This phi may be visited
    // again at a different level of phi recursion. In that case, we
    // should return the cached introducers so that they can be
    // remapped.
    cache.pendingPhis.erase(phi.value)
  }
}

// Given incoming values on a predecessor path, return the
// corresponding values on the successor block. Each incoming value is
// either used by a phi in the successor block, or it must dominate
// the successor block.
private func mapToPhi<PredecessorSequence: Sequence<Value>> (
  predecessor: BasicBlock, incomingValues: PredecessorSequence)
-> LazyMapSequence<PredecessorSequence, Value> {

  let branch = predecessor.terminator as! BranchInst
  // Gather the new introducers for the successor block.
  return incomingValues.lazy.map { incomingValue in
    // Find an outer adjacent phi in the successor block.
    if let incomingOp =
         branch.operands.first(where: { $0.value == incomingValue }) {
      return branch.getArgument(for: incomingOp)
    }
    // No candidates phi are outer-adjacent phis. The incoming
    // `predDef` must dominate the current guaranteed phi.
    return incomingValue
  }
}

/// Find each "enclosing value" whose OSSA lifetime immediately
/// encloses a guaranteed value. The guaranteed `value` being enclosed
/// effectively keeps these enclosing values alive. This lets you walk
/// up the levels of nested OSSA lifetimes to determine all the
/// lifetimes that are kept alive by a given SILValue. In particular,
/// it discovers "outer-adjacent phis": phis that are kept alive by
/// uses of another phi in the same block.
///
/// If `value` is a forwarded guaranteed value, then this finds the
/// introducers of the current borrow scope, which is never an empty
/// set.
///
/// If `value` introduces a borrow scope, then this finds the
/// introducers of the outer enclosing borrow scope that contains this
/// inner scope.
///
/// If `value` is a `begin_borrow`, then this returns its operand.
///
/// If `value` is an owned value, a function argument, or a
/// load_borrow, then this is an empty set.
///
/// If `value` is a reborrow, then this either returns a dominating
/// enclosing value or an outer adjacent phi.
///
/// Example:                                       // enclosing value:
///                                                // ~~~~~~~~~~~~
///   bb0(%0 : @owned $Class,                      // (none)
///       %1 : @guaranteed $Class):                // (none)
///     %borrow0 = begin_borrow %0                 // %0
///     %pair = struct $Pair(%borrow0, %1)         // %borrow0, %1
///     %first = struct_extract %pair              // %borrow0, %1
///     %field = ref_element_addr %first           // (none)
///     %load = load_borrow %field : $*C           // %load
///
/// Example:                                       // enclosing value:
///                                                // ~~~~~~~~~~~~
///     %outerBorrow = begin_borrow %0             // %0
///     %innerBorrow = begin_borrow %outerBorrow   // %outerBorrow
///     br bb1(%outerBorrow, %innerBorrow)
///   bb1(%outerReborrow : @guaranteed,            // %0
///       %innerReborrow : @guaranteed)            // %outerReborrow
///
func gatherEnclosingValues(for value: Value,
                           in enclosingValues: inout Stack<Value>,
                           _ context: some Context) {

  var gatherValues = EnclosingValues(context)
  defer { gatherValues.deinitialize() }
  var cache = BorrowIntroducers.Cache(context)
  defer { cache.deinitialize() }
  gatherValues.gather(for: value, in: &enclosingValues, &cache)
}

/// Find inner adjacent phis in the same block as `enclosingPhi`.
/// These keep the enclosing (outer adjacent) phi alive.
func gatherInnerAdjacentPhis(for enclosingPhi: Phi,
                             in innerAdjacentPhis: inout Stack<Phi>,
                             _ context: Context) {
  for candidatePhi in enclosingPhi.successor.arguments {
    var enclosingValues = Stack<Value>(context)
    defer { enclosingValues.deinitialize() }
    gatherEnclosingValues(for: candidatePhi, in: &enclosingValues, context)
    if enclosingValues.contains(where: { $0 == enclosingPhi.value}) {
      innerAdjacentPhis.push(Phi(candidatePhi)!)
    }
  }
}

// Find the enclosing values for any value, including reborrows.
private struct EnclosingValues {
  var context: Context
  var visitedReborrows : ValueSet

  init(_ context: Context) {
    self.context = context
    self.visitedReborrows = ValueSet(context)
  }

  mutating func deinitialize() {
    visitedReborrows.deinitialize()
  }

  mutating func gather(for value: Value,
                       in enclosingValues: inout Stack<Value>,
                       _ cache: inout BorrowIntroducers.Cache) {
    if value is Undef || value.ownership != .guaranteed {
      return
    }
    if let beginBorrow = BeginBorrowValue(value) {
      switch beginBorrow {
      case let .beginBorrow(bbi):
        // Gather the outer enclosing borrow scope.
        BorrowIntroducers.gather(for: bbi.operand.value, in: &enclosingValues,
                                 &cache, context)
      case .loadBorrow, .beginApply, .functionArgument:
        // There is no enclosing value on this path.
        break
      case let .reborrow(reborrow):
        gather(forReborrow: reborrow, in: &enclosingValues, &cache)
      }
    } else {
      // Handle forwarded guaranteed values.
      BorrowIntroducers.gather(for: value, in: &enclosingValues,
                               &cache, context)
    }
  }
  
  // Given a reborrow, find the enclosing values. Each enclosing value
  // is represented by one of the following cases, which refer to the
  // example below:
  //
  // dominating owned value -> %value encloses %reborrow_1
  // owned outer-adjacent phi -> %phi_3 encloses %reborrow_3
  // dominating outer borrow introducer -> %outerBorrowB encloses %reborrow
  // outer-adjacent reborrow -> %outerReborrow encloses %reborrow
  //
  // Recurse along the use-def phi web until a begin_borrow is
  // reached. Then find all introducers of the begin_borrow's
  // operand. At each level, find the outer adjacent phi, if one
  // exists, otherwise return the most recently found dominating
  // definition.
  //
  // If `reborrow` was already encountered because of a phi cycle,
  // then no enclosingDefs are added.
  //
  // Example:
  //
  //         %value = ...
  //         %borrow = begin_borrow %value
  //         br one(%borrow)
  //     one(%reborrow_1 : @guaranteed)
  //         br two(%value, %reborrow_1)
  //     two(%phi_2 : @owned, %reborrow_2 : @guaranteed)
  //         br three(%value, %reborrow_2)
  //     three(%phi_3 : @owned, %reborrow_3 : @guaranteed)
  //         end_borrow %reborrow_3
  //         destroy_value %phi_3
  //
  // gather(forReborrow: %reborrow_3) finds %phi_3 by computing
  // enclosing defs in this order
  //     (inner -> outer):
  //
  //     %reborrow_1 -> %value
  //     %reborrow_2 -> %phi_2
  //     %reborrow_3 -> %phi_3
  //
  // Example:
  //
  //         %outerBorrowA = begin_borrow
  //         %outerBorrowB = begin_borrow
  //         %struct = struct (%outerBorrowA, outerBorrowB)
  //         %borrow = begin_borrow %struct
  //         br one(%outerBorrowA, %borrow)
  //     one(%outerReborrow : @guaranteed, %reborrow : @guaranteed)
  //
  // gather(forReborrow: %reborrow) finds (%outerReborrow, %outerBorrowB).
  //
  private mutating func gather(forReborrow reborrow: Phi,
                               in enclosingValues: inout Stack<Value>,
                               _ cache: inout BorrowIntroducers.Cache) {

    guard visitedReborrows.insert(reborrow.value) else { return }

    // avoid duplicates in the enclosingValues set.
    var pushedEnclosingValues = ValueSet(context)
    defer { pushedEnclosingValues.deinitialize() }

    // Find the enclosing introducer for each reborrow operand, and
    // remap it to the enclosing introducer for the successor block.
    for (pred, incomingValue)
    in zip(reborrow.predecessors, reborrow.incomingValues) {
      var incomingEnclosingValues = Stack<Value>(context)
      defer {
        incomingEnclosingValues.deinitialize()
      }
      gather(for: incomingValue, in: &incomingEnclosingValues, &cache)
      mapToPhi(predecessor: pred,
               incomingValues: incomingEnclosingValues).forEach {
        if pushedEnclosingValues.insert($0) {
          enclosingValues.append($0)
        }
      }
    }
  }
}

let borrowIntroducersTest = FunctionTest("borrow_introducers") {
  function, arguments, context in
  let value = arguments.takeValue()
  print(function)
  print("Borrow introducers for: \(value)")
  var introducers = Stack<Value>(context)
  defer {
    introducers.deinitialize()
  }
  gatherBorrowIntroducers(for: value, in: &introducers, context)
  introducers.forEach { print($0) }
}

let enclosingValuesTest = FunctionTest("enclosing_values") {
  function, arguments, context in
  let value = arguments.takeValue()
  print(function)
  print("Enclosing values for: \(value)")
  var enclosing = Stack<Value>(context)
  defer {
    enclosing.deinitialize()
  }
  gatherEnclosingValues(for: value, in: &enclosing, context)
  enclosing.forEach { print($0) }
}
