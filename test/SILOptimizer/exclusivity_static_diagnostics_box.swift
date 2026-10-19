// RUN: %target-swift-frontend -emit-sil -verify -O -target %target-swift-6.2-abi-triple %s -o /dev/null

// Static diagnostics for variables captured by box: by a closure that may
// escape, or by a local function. Their accesses are dynamically enforced
// unless MandatoryAllocBoxToStack promotes the box, which it only does if a
// conflict either can't happen or happens on every path through the call. So
// a conflict is only diagnosed statically if dynamic enforcement would fail
// every time the call executes. rdar://179406916

func use(_: Span<Int>) {}
func use(_: Int) {}

func callNow(_ body: @escaping () -> ()) {
  body()
}

func modifyAndCall(_ x: inout Int, _ body: @escaping () -> ()) {
  body()
}

// The callee calls the closure on every path.

func escapingClosureModifiesBorrowedArray() {
  var array = [1, 2, 3]
  let span = array.span // expected-note {{conflicting access is here}}
  callNow {
    array = [] // expected-error {{overlapping accesses to 'array', but modification requires exclusive access; consider copying to a local variable}}
  }
  use(span)
}

func escapingClosureModifiesInoutArgument() {
  var count = 0
  modifyAndCall(&count) { // expected-error {{overlapping accesses to 'count', but modification requires exclusive access; consider copying to a local variable}}
    count += 1 // expected-note {{conflicting access is here}}
  }
}

func callForwarded(_ body: @escaping () -> ()) {
  callNow(body)
}

func escapingClosureForwarded() {
  var array = [1, 2, 3]
  let span = array.span // expected-note {{conflicting access is here}}
  callForwarded {
    array = [] // expected-error {{overlapping accesses to 'array', but modification requires exclusive access; consider copying to a local variable}}
  }
  use(span)
}

// Reads don't conflict with reads.

func escapingClosureReadsBorrowedArray() {
  var array = [1, 2, 3]
  array.append(4)
  let span = array.span
  callNow {
    use(array.count)
  }
  use(span)
}

// The callee may not call the closure. Dynamic enforcement accepts the
// program if it doesn't.

func callIf(_ flag: Bool, _ body: @escaping () -> ()) {
  if flag {
    body()
  }
}

func callInLoop(_ n: Int, _ body: @escaping () -> ()) {
  for _ in 0..<n {
    body()
  }
}

func callUnlessTrapping(_ flag: Bool, _ body: @escaping () -> ()) {
  if flag {
    fatalError()
  }
  body()
}

func ignore(_ body: @escaping () -> ()) {}

var savedClosure: (() -> ())?

func save(_ body: @escaping () -> ()) {
  savedClosure = body
}

func escapingClosureMayBeCalled(_ flag: Bool, _ n: Int) {
  var array = [1, 2, 3]
  let span = array.span
  callIf(flag) {
    array = []
  }
  callInLoop(n) {
    array = []
  }
  callUnlessTrapping(flag) {
    array = []
  }
  ignore {
    array = []
  }
  save {
    array = []
  }
  use(span)
}

// The closure may not access the variable.

func escapingClosureMayModify(_ flag: Bool) {
  var array = [1, 2, 3]
  let span = array.span
  callNow {
    if flag {
      array = []
    }
  }
  use(span)
}

// A local function that captures the variable by box.

func localFunctionModifies() {
  var array = [1, 2, 3]
  func modify() {
    array = [] // expected-error {{overlapping accesses to 'array', but modification requires exclusive access; consider copying to a local variable}}
  }
  let span = array.span // expected-note {{conflicting access is here}}
  modify()
  use(span)
}

func localFunctionMayModify(_ flag: Bool) {
  var array = [1, 2, 3]
  func modify() {
    if flag {
      array = []
    }
  }
  let span = array.span
  modify()
  use(span)
}

// A non-escaping argument that is a closure capturing the variable by box.

func readAndPerform(_ x: UnsafePointer<Int>, closure: () -> ()) {
  closure()
}

func modifyAndPerform(_ x: UnsafeMutablePointer<Int>, closure: () -> ()) {
  closure()
}

// The closure is converted to a non-escaping closure, which
// MandatoryAllocBoxToStack doesn't promote, so the conflict is caught at run
// time.
func noescapeArgumentCapturesBox() {
  var x = 3
  let c = { x = 42 }
  readAndPerform(&x, closure: c)
  _ = x
}

func noescapeArgumentMayAccessBox(_ flag: Bool) {
  var x = 3
  let c = { if flag { x = 42 } }
  modifyAndPerform(&x, closure: c)
  _ = x
}

// rdar://102056143: a violation through recursive captures. local1 only
// modifies flag (through local2) on some paths, so flag stays a box and the
// conflict traps at run time (Interpreter/enforce_exclusive_access.swift).
func recursiveCaptureViolation() -> Bool {
  var flag = false
  func local1(_ x: inout Bool) {
    if !flag {
      x = true
      local2()
    }
  }
  func local2() {
    local1(&flag)
  }
  local2()
  return flag
}
