// RUN: %target-swift-frontend -enforce-exclusivity=checked -Onone -Xllvm -sil-print-types -emit-sil -parse-as-library %s -Xllvm -debug-only=access-enforcement-selection 2>&1 | %FileCheck %s
// RUN: %target-swift-frontend -enforce-exclusivity=checked -Onone -Xllvm -sil-print-types -emit-sil -parse-as-library %s -Xllvm -debug-only=access-enforcement-selection 2>&1 | %FileCheck %s --check-prefix=BOX
// REQUIRES: asserts

// This is a source-level test because it helps bring up the entire -Onone pipeline with the access markers.

public func takesInout(_ i: inout Int) {
  i = 42
}
// CHECK-LABEL: Access Enforcement Selection in $s28access_enforcement_selection10takesInoutyySizF
// CHECK: Static Access: %{{.*}} = begin_access [modify] [static] %{{.*}} : $*Int

// Helper taking a basic, no-escape closure.
func takeClosure(_: ()->Int) {}

var escapedClosure: (()->Int)?

// Helper taking an escaping closure.
func takeClosureAndInout(_: inout Int, _ c: @escaping ()->Int) {
  escapedClosure = c
}

// Helper taking an escaping closure.
func takeEscapingClosure(_ c: @escaping ()->Int) {
  escapedClosure = c
}

// Helper taking an escaping closure that it only calls.
func callEscapingClosure(_ c: @escaping ()->Int) -> Int {
  return c()
}

// Generate an alloc_stack that escapes into a closure.
public func captureStack() -> Int {
  // Use a `var` so `x` isn't treated as a literal.
  var x = 3
  takeClosure { return x }
  return x
}
// BOX-LABEL: Box Enforcement Selection in $s28access_enforcement_selection12captureStackSiyF
// Static access for `return x`.
// BOX: Static Access: %{{.*}} = begin_access [read] [static] %{{.*}} : $*Int

// CHECK-LABEL: Access Enforcement Selection in $s28access_enforcement_selection12captureStackSiyFSiyXEfU_
// CHECK: Static Access: %{{.*}} = begin_access [read] [static] %{{.*}} : $*Int


// Generate an alloc_stack that does not escape into a closure.
public func nocaptureStack() -> Int {
  var x = 3
  takeClosure { return 5 }
  return x
}
// BOX-LABEL: Box Enforcement Selection in $s28access_enforcement_selection14nocaptureStackSiyF
// Static access for `return x`.
// BOX: Static Access: %{{.*}} = begin_access [read] [static] %{{.*}} : $*Int
//
// CHECK-LABEL: Access Enforcement Selection in $s28access_enforcement_selection14nocaptureStackSiyFSiyXEfU_

// Generate an alloc_stack that escapes into a closure while an access is
// in progress.
public func captureStackWithInoutInProgress() -> Int {
  // Use a `var` so `x` isn't treated as a literal.
  var x = 3
  takeClosureAndInout(&x) { return x }
  return x
}
// BOX-LABEL: Box Enforcement Selection in $s28access_enforcement_selection31captureStackWithInoutInProgressSiyF
// Access for `return x`.
// BOX-DAG: Dynamic Access: %{{.*}} = begin_access [read] [dynamic] %{{.*}} : $*Int
// Access for `&x`.
// BOX-DAG: Dynamic Access: %{{.*}} = begin_access [modify] [dynamic] %{{.*}} : $*Int
//
// CHECK-LABEL: Access Enforcement Selection in $s28access_enforcement_selection31captureStackWithInoutInProgressSiyFSiycfU_
// CHECK: Dynamic Access: %{{.*}} = begin_access [read] [dynamic] %{{.*}} : $*Int

// Generate an alloc_box that escapes into a closure.
public func captureBox() -> Int {
  var x = 3
  takeEscapingClosure { x = 4; return x }
  return x
}
// BOX-LABEL: Box Enforcement Selection in $s28access_enforcement_selection10captureBoxSiyF
// Dynamic access for `return x`.
// BOX: Dynamic Access: %{{.*}} = begin_access [read] [dynamic] %{{.*}} : $*Int
// CHECK-LABEL: $s28access_enforcement_selection10captureBoxSiyFSiycfU_

// Generate an alloc_box that is captured by an escaping closure that does not
// escape. Static enforcement is equivalent, because no access is in progress
// when the closure is called.
public func captureBoxNonEscaping() -> Int {
  var x = 3
  _ = callEscapingClosure { x = 4; return x }
  return x
}
// BOX-LABEL: Box Enforcement Selection in $s28access_enforcement_selection21captureBoxNonEscapingSiyF
// Static access for `return x`.
// BOX: Static Access: %{{.*}} = begin_access [read] [static] %{{.*}} : $*Int
// CHECK-LABEL: Access Enforcement Selection in $s28access_enforcement_selection21captureBoxNonEscapingSiyFSiycfU_
// CHECK: Static Access: %{{.*}} = begin_access [modify] [static] %{{.*}} : $*Int
// CHECK: Static Access: %{{.*}} = begin_access [read] [static] %{{.*}} : $*Int

// Generate a closure in which the @inout_aliasing argument
// escapes to an @inout function `bar`.
public func recaptureStack() -> Int {
  var x = 3
  takeClosure { takesInout(&x); return x }
  return x
}
// BOX-LABEL: Box Enforcement Selection in $s28access_enforcement_selection14recaptureStackSiyF
//
// Static access for `return x`.
// BOX: Static Access:   %{{.*}} = begin_access [read] [static] %{{.*}} : $*Int

// CHECK-LABEL: Access Enforcement Selection in $s28access_enforcement_selection14recaptureStackSiyFSiyXEfU_
//
// The first [modify] access inside the closure is static. It enforces the
// @inout argument.
// CHECK: Static Access: %{{.*}} = begin_access [modify] [static] %{{.*}} : $*Int
//
// The second [read] access is static. Same as `captureStack` above.
//
// CHECK: Static Access: %{{.*}} = begin_access [read] [static] %{{.*}} : $*Int


// -----------------------------------------------------------------------------
// Inout access in a recursive non-escaping closure currently has
// static enforcement.

public protocol Apply {
    func apply(_ body: (Apply) -> ())
}

// CHECK-LABEL: sil private @$s28access_enforcement_selection20testRecursiveClosure1a1xyAA5Apply_p_SiztF9localFuncL_1byAaE_p_tF : $@convention(thin) (@in_guaranteed any Apply, @inout_aliasable Int) -> () {
// CHECK: bb0(%0 : $*any Apply, %1 : @closureCapture $*Int):
// CHECK:   begin_access [modify] [static] %1 : $*Int
// CHECK-LABEL: } // end sil function '$s28access_enforcement_selection20testRecursiveClosure1a1xyAA5Apply_p_SiztF9localFuncL_1byAaE_p_tF'
public func testRecursiveClosure(a: Apply, x: inout Int) {
    func localFunc(b: Apply) {
        x += 1
        let closure = { (c: Apply) in
            c.apply(localFunc)
        }
        b.apply(closure)
    }
    a.apply(localFunc)
}

// CHECK-LABEL: sil private @$s28access_enforcement_selection25testRecursiveLocalCapture1iys5Int64Vz_tF6local2L_yyF : $@convention(thin) (@inout_aliasable Int64) -> () {
// CHECK: begin_access [modify] [static] %0 : $*Int64
// CHECK-LABEL: } // end sil function '$s28access_enforcement_selection25testRecursiveLocalCapture1iys5Int64Vz_tF6local2L_yyF'
func testRecursiveLocalCapture(i: inout Int64) {
  func local1() {
    if i < 1 {
      local2()
    }
  }
  func local2() {
    i = i + 1
    local1()
  }
  local1()
}
