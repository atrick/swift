//===--- BoxAccessSummary.swift - Accesses to a box during a call --------===//
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

import SIL

/// Whether something happens during a call of a function.
enum PathSummary {
  /// It never happens.
  case noPaths
  /// It happens on every path that returns, throws or unwinds. Paths that trap or loop forever
  /// don't count: the call never completes on them.
  case allPaths
  /// It happens on some paths, or it can't be determined.
  case undecided
}

/// The accesses to a box argument during a call of its function.
struct BoxArgumentSummary {
  /// The strongest kind of access during the call: `.modify` if any access is not a read. Nil if
  /// there are no accesses.
  let kind: BeginAccessInst.AccessKind?

  /// Where the accesses of `kind` happen.
  ///
  /// Accesses of a weaker kind are ignored. With `.read`, an access in progress in the caller
  /// either never conflicts (a read) or conflicts with every access (a modify). With `.modify`, a
  /// modify on every path conflicts with any access in progress. A read on every path combined
  /// with a modify on some paths is undecided, because a read in progress only conflicts on some
  /// paths.
  let paths: PathSummary

  static let none = BoxArgumentSummary(kind: nil, paths: .noPaths)
  static let unknown = BoxArgumentSummary(kind: .modify, paths: .undecided)
}

/// Summarizes the accesses to a box argument, and the invocations of a closure argument, during a
/// call of the argument's function.
///
/// The accesses to a box argument are those through its projections, through direct callees that
/// the box is passed to, and conservatively through any other use, such as a closure that captures
/// it. An access to a stored property or tuple element counts as an access to the entire boxed
/// value.
final class BoxAccessSummary {
  /// A nil entry marks a summary in progress, for recursion.
  private var boxArgumentCache = [FunctionArgument: BoxArgumentSummary?]()
  private var closureArgumentCache = [FunctionArgument: PathSummary?]()

  /// The accesses to box argument `arg` during a call of its function.
  func boxArgument(_ arg: FunctionArgument) -> BoxArgumentSummary {
    if let cached = boxArgumentCache[arg] {
      // A recursive call can't be summarized.
      return cached ?? .unknown
    }
    boxArgumentCache[arg] = .some(nil)
    let summary = computeBoxArgument(arg)
    boxArgumentCache[arg] = summary
    return summary
  }

  /// Whether closure argument `arg` is invoked during a call of its function.
  func closureArgumentInvocation(_ arg: FunctionArgument) -> PathSummary {
    if let cached = closureArgumentCache[arg] {
      // A recursive call can't be summarized.
      return cached ?? .undecided
    }
    closureArgumentCache[arg] = .some(nil)
    let invocation = computeClosureArgumentInvocation(arg)
    closureArgumentCache[arg] = invocation
    return invocation
  }

  private func computeBoxArgument(_ arg: FunctionArgument) -> BoxArgumentSummary {
    var reads = PathCollector()
    var modifies = PathCollector()
    func add(_ kind: BeginAccessInst.AccessKind, known block: BasicBlock) {
      if kind == .read {
        reads.add(known: block)
      } else {
        modifies.add(known: block)
      }
    }
    func addPotential(_ kind: BeginAccessInst.AccessKind) {
      if kind == .read {
        reads.addPotential()
      } else {
        modifies.addPotential()
      }
    }

    var addresses = [Value]()
    var boxes: [Value] = [arg]
    var visited = Set<HashableValue>([arg.hashable])
    while let box = boxes.popLast() {
      for use in box.uses {
        let user = use.instruction
        switch user {
        case is BeginBorrowInst, is CopyValueInst, is MoveValueInst, is MarkUninitializedInst:
          let result = user as! SingleValueInstruction
          if visited.insert(result.hashable).inserted {
            boxes.append(result)
          }
          continue
        case let projectBox as ProjectBoxInst:
          addresses.append(projectBox)
          continue
        case is DestroyValueInst, is EndBorrowInst, is DebugValueInst, is DeallocBoxInst:
          continue
        default:
          break
        }
        if let apply = user as? FullApplySite, !apply.isCallee(operand: use),
           let callee = apply.referencedFunction, callee.isDefinition,
           let calleeArg = apply.calleeArgument(of: use, in: callee), calleeArg.type.isBox
        {
          // The box is forwarded to a direct callee.
          let calleeSummary = boxArgument(calleeArg)
          if let kind = calleeSummary.kind {
            switch calleeSummary.paths {
            case .noPaths:
              break
            case .allPaths:
              add(kind, known: user.parentBlock)
            case .undecided:
              addPotential(kind)
            }
          }
          continue
        }
        // The box may be accessed through an unknown use, such as a closure that captures it.
        addPotential(.modify)
      }
    }
    while let address = addresses.popLast() {
      for use in address.uses {
        switch use.instruction {
        case let access as BeginAccessInst:
          if access.enforcement != .unsafe {
            add(access.accessKind, known: access.parentBlock)
          }
        case let projection as SingleValueInstruction
            where projection is MarkUnresolvedNonCopyableValueInst ||
                  projection is StructElementAddrInst || projection is TupleElementAddrInst:
          addresses.append(projection)
        case is ApplySite:
          // An address passed to a call, such as a non-escaping closure that captures it, may be
          // accessed during the call.
          addPotential(.modify)
        default:
          // Other uses, such as initialization, are not formal accesses.
          break
        }
      }
    }
    if modifies.hasAccesses {
      return BoxArgumentSummary(kind: .modify, paths: modifies.paths(in: arg.parentFunction))
    }
    if reads.hasAccesses {
      return BoxArgumentSummary(kind: .read, paths: reads.paths(in: arg.parentFunction))
    }
    return .none
  }

  private func computeClosureArgumentInvocation(_ arg: FunctionArgument) -> PathSummary {
    var invocations = PathCollector()
    var worklist: [Value] = [arg]
    var visited = Set<HashableValue>([arg.hashable])
    while let value = worklist.popLast() {
      for use in value.uses {
        let user = use.instruction
        if let forwarded = use.forwardedClosure {
          if visited.insert(forwarded.hashable).inserted {
            worklist.append(forwarded)
          }
          continue
        }
        switch user {
        case is DestroyValueInst, is EndBorrowInst, is DebugValueInst, is DeallocStackInst,
             is DestroyNotEscapedClosureInst:
          continue
        default:
          break
        }
        if let apply = user as? FullApplySite {
          if apply.isCallee(operand: use) {
            invocations.add(known: user.parentBlock)
            continue
          }
          if let callee = apply.referencedFunction, callee.isDefinition,
             let calleeArg = apply.calleeArgument(of: use, in: callee), calleeArg.type.isLoweredFunction
          {
            switch closureArgumentInvocation(calleeArg) {
            case .noPaths:
              break
            case .allPaths:
              invocations.add(known: user.parentBlock)
            case .undecided:
              invocations.addPotential()
            }
            continue
          }
        }
        // An unknown use may invoke the closure.
        invocations.addPotential()
      }
    }
    return invocations.paths(in: arg.parentFunction)
  }
}

/// Collects where something happens in a function: the blocks where it is known to happen, and
/// whether it may also happen somewhere that isn't known.
private struct PathCollector {
  private var knownBlocks = Set<BasicBlock>()
  private var hasPotential = false

  var hasAccesses: Bool { hasPotential || !knownBlocks.isEmpty }

  mutating func add(known block: BasicBlock) {
    knownBlocks.insert(block)
  }

  mutating func addPotential() {
    hasPotential = true
  }

  func paths(in function: Function) -> PathSummary {
    if hasPotential {
      return .undecided
    }
    if knownBlocks.isEmpty {
      return .noPaths
    }
    return allPathsReach(knownBlocks, in: function) ? .allPaths : .undecided
  }
}

/// Return true if no path from the entry returns, throws or unwinds without passing through a block
/// in `blocks`. A path that traps or loops forever doesn't count. This uses a Swift set rather than
/// a pass context's block set, so it works for any function.
private func allPathsReach(_ blocks: Set<BasicBlock>, in function: Function) -> Bool {
  var worklist = [function.entryBlock]
  var visited = Set<BasicBlock>(worklist)
  while let block = worklist.popLast() {
    if blocks.contains(block) || block.hasNoReturnCall {
      continue
    }
    if block.terminator.isFunctionExiting {
      return false
    }
    for succ in block.successors where visited.insert(succ).inserted {
      worklist.append(succ)
    }
  }
  return true
}

private extension BasicBlock {
  /// Return true if a call that never returns ends the path through this block. Before
  /// NoReturnFolding, such a call need not be followed by `unreachable`.
  var hasNoReturnCall: Bool {
    instructions.contains { ($0 as? FullApplySite)?.isCalleeNoReturn ?? false }
  }
}

extension PartialApplyInst {
  /// The calls that this closure is forwarded to, through copies, borrows and conversions, either as
  /// the callee or as an argument.
  var callsOfClosure: [Instruction] {
    var calls = [Instruction]()
    var values: [Value] = [self]
    var visited = Set<HashableValue>([self.hashable])
    while let value = values.popLast() {
      for use in value.uses {
        if let forwarded = use.forwardedClosure {
          if visited.insert(forwarded.hashable).inserted {
            values.append(forwarded)
          }
        } else if use.instruction is FullApplySite {
          calls.append(use.instruction)
        }
      }
    }
    return calls
  }
}

private extension Operand {
  /// The result of the instruction that forwards a closure from this operand, if any.
  var forwardedClosure: Value? {
    switch instruction {
    case is BeginBorrowInst, is CopyValueInst, is MoveValueInst, is ConvertFunctionInst,
         is ConvertEscapeToNoEscapeInst:
      return (instruction as! SingleValueInstruction)
    case let markDependence as MarkDependenceInst:
      return self == markDependence.valueOperand ? markDependence : nil
    default:
      return nil
    }
  }
}
