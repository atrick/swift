// RUN: %empty-directory(%t)
// RUN: %target-build-swift %s -o %t/a.out_Onone -enforce-exclusivity=checked -Onone
// RUN: %target-build-swift %s -o %t/a.out_O -enforce-exclusivity=checked -O
// RUN: %target-codesign %t/a.out_Onone
// RUN: %target-codesign %t/a.out_O
// RUN: %target-run %t/a.out_Onone
// RUN: %target-run %t/a.out_O
// REQUIRES: executable_test

// Exclusivity of variables captured by box must be enforced the same way at
// every optimization level, whether or not AllocBoxToStack promotes the box.
// Static enforcement is only used where it is equivalent to dynamic
// enforcement, so these conflicts, which only happen on some paths, must be
// caught at run time. rdar://179406916

import StdlibUnittest

var tests = TestSuite("ExclusivityCapturedBox")

let yes = CommandLine.arguments.count > 0

// The callee calls the escaping closure only on some paths.
@inline(never)
func modifyAndCallIf(_ x: inout Int, _ flag: Bool, _ body: @escaping () -> ()) {
  if flag {
    body()
  }
}

@inline(never)
func escapingClosureMayBeCalledDuringAccess(_ flag: Bool) {
  var count = 0
  modifyAndCallIf(&count, flag) {
    count += 1
  }
  _blackHole(count)
}

tests.test("EscapingClosureNotCalledDuringAccess") {
  escapingClosureMayBeCalledDuringAccess(!yes)
}

tests.test("EscapingClosureCalledDuringAccess") {
  expectCrashLater(withMessage: "Simultaneous accesses")
  escapingClosureMayBeCalledDuringAccess(yes)
}

// A local function that captures the variable by box modifies it only on some
// paths.
@inline(never)
func localFunctionMayModifyDuringAccess(_ flag: Bool) {
  var x = 0
  func inner() {
    if flag {
      x += 1
    }
  }
  func outer(_ v: inout Int) {
    inner()
  }
  outer(&x)
  _blackHole(x)
}

tests.test("LocalFunctionDoesNotModifyDuringAccess") {
  localFunctionMayModifyDuringAccess(!yes)
}

tests.test("LocalFunctionModifiesDuringAccess") {
  expectCrashLater(withMessage: "Simultaneous accesses")
  localFunctionMayModifyDuringAccess(yes)
}

// An escaped closure is called during an access.
var saved: (() -> ())?

@inline(never)
func callSaved(_ x: inout Int) {
  saved!()
}

@inline(never)
func escapedClosureCalledDuringAccess() {
  var count = 0
  saved = { count += 1 }
  callSaved(&count)
  _blackHole(count)
}

tests.test("EscapedClosureCalledDuringAccess") {
  expectCrashLater(withMessage: "Simultaneous accesses")
  escapedClosureCalledDuringAccess()
}

// `modify` captures `x` by box, and it is passed a non-escaping closure that
// captures `x` by address. The closure runs while `modify` accesses `x`
// through its box.
@inline(never)
func bump(_ v: inout Int, _ body: () -> ()) {
  body()
  v += 1
}

@inline(never)
func boxPassedWithNonEscapingClosure() {
  var x = 0
  func modify(_ body: () -> ()) {
    bump(&x, body)
  }
  modify { x += 1 }
  _blackHole(x)
}

tests.test("BoxPassedWithNonEscapingClosure") {
  expectCrashLater(withMessage: "Simultaneous accesses")
  boxPassedWithNonEscapingClosure()
}

// Local functions that only access their captured box, called directly
// (rdar://157458037). No conflict.
@inline(never)
func localFunctionsWithoutConflict() -> Int {
  var a1 = 1
  var a2 = 2

  func innerFunction() {
    a1 += 1
    a2 += 2
  }

  innerFunction()
  innerFunction()
  return a1 + a2
}

tests.test("LocalFunctionsWithoutConflict") {
  expectEqual(9, localFunctionsWithoutConflict())
}

runAllTests()
