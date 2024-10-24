// RUN: %target-swift-frontend -module-name Swift -emit-sil %s -o - | FileCheck %s

@_silgen_name("makeUpAPointer")
func makeUpAPointer<T: ~Copyable>() -> UnsafePointer<T>
@_silgen_name("makeUpAMutablePointer")
func makeUpAPointer<T: ~Copyable>() -> UnsafeMutablePointer<T>
@_silgen_name("makeUpAnInt")
func makeUpAnInt() -> Int

class X {}

struct NC: ~Copyable {
  var x: Any = X()
  deinit {}
}

struct S {
  var data: NC {
    unsafeAddress { return makeUpAPointer() }
  }

  var mutableData: NC {
    unsafeAddress { return makeUpAPointer() }
    unsafeMutableAddress { return makeUpAPointer() }
  }
}

struct SNC: ~Copyable {
  var data: NC {
    unsafeAddress { return makeUpAPointer() }
  }

  var mutableData: NC {
    unsafeAddress { return makeUpAPointer() }
    unsafeMutableAddress { return makeUpAPointer() }
  }
}

class C {
  final var data: NC {
    unsafeAddress { return makeUpAPointer() }
  }

  final var mutableData: NC {
    unsafeAddress { return makeUpAPointer() }
    unsafeMutableAddress { return makeUpAPointer() }
  }
}

func borrow(_ nc: borrowing NC) {}
func mod(_ nc: inout NC) {}

// CHECK-LABEL: sil hidden @$s4main11testCBorrow1cyAA1CC_tF : $@convention(thin) (@guaranteed C) -> () {
// CHECK: [[ADR:%.*]] = pointer_to_address %4 : $Builtin.RawPointer to [strict] $*NC
// CHECK: [[MD:%.*]] = mark_dependence [nonescaping] [[ADR]] : $*NC on %0 : $C
// CHECK: begin_access [read] [unsafe] [[MD]] : $*NC
// CHECK: apply
// CHECK: end_access
// CHECK-LABEL: } // end sil function '$s4main11testCBorrow1cyAA1CC_tF'
func testCBorrow(c: C) {
  borrow(c.data)
}

// CHECK-LABEL: sil hidden @$s4main8testCMod1cyAA1CC_tF : $@convention(thin) (@guaranteed C) -> () {
// CHECK-LABEL: } // end sil function '$s4main8testCMod1cyAA1CC_tF'
func testCMod(c: C) {
  mod(&c.mutableData)
}

func testSBorrow(s: S) {
  borrow(s.data)
}

func testSMod(s: inout S) {
  mod(&s.mutableData)
}

func testSInoutBorrow(mut_s s: inout S) {
  borrow(s.data)
}

func testSInoutMutBorrow(mut_s s: inout S) {
  borrow(s.mutableData)
}

func testSInoutMod(mut_s s: inout S) {
  mod(&s.mutableData)
}

func testSNCBorrow(snc: borrowing SNC) {
  borrow(snc.data)
}

func testSNCMutBorrow(snc: borrowing SNC) {
  borrow(snc.mutableData)
}

func testSNCMod(mut_snc snc: inout SNC) {
  mod(&snc.mutableData)
}
