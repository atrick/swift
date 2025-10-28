// RUN: %target-swift-frontend -O -wmo -Xllvm -enable-existential-specializer -Xllvm -sil-disable-pass=GenericSpecializer -Xllvm -sil-disable-pass=FunctionSignatureOpts -Xllvm -sil-disable-pass=SILCombine -emit-sil -sil-verify-all %s | %FileCheck %s

internal protocol SPP {
  func bar()  -> Int32
}
internal class SCC: SPP {
  @inline(never) func bar() -> Int32 {
   return 5
  }
}

@inline(never) internal func test_00(b:SPP) -> Int32{
 return b.bar()
}

@inline(never) internal func test_01(b:SPP) -> Int32{
 return test_00(b:b)
}

// CHECK-LABEL: sil hidden [noinline] @$s30existential_transform_soletype7test_011bs5Int32VAA3SPP_p_tF : $@convention(thin) (@in_guaranteed any SPP) -> Int32 {
// CHECK: bb0(%0 : $*any SPP):
// CHECK:   debug_value {{.*}} expr op_deref
// CHECK:   function_ref @$s30existential_transform_soletype7test_001bs5Int32VAA3SPP_p_tFTf4e_n : $@convention(thin) <τ_0_0 where τ_0_0 : SPP> (@in_guaranteed τ_0_0) -> Int32
// CHECK:   open_existential_addr
// CHECK:   apply
// CHECK:   return
// CHECK-LABEL: } // end sil function '$s30existential_transform_soletype7test_011bs5Int32VAA3SPP_p_tF'

// CHECK-LABEL: sil hidden [Onone] @$s30existential_transform_soletype7test_026numbers5Int32VAE_tF : $@convention(thin) (Int32) -> Int32 {
// CHECK: bb0(%0 : $*τ_0_0):
// CHECK:   alloc_stack
// CHECK:   init_existential_addr
// CHECK:   copy_addr
// CHECK:   debug_value {{.*}} expr op_deref
// CHECK:   open_existential_addr
// CHECK:   witness_method
// CHECK:   apply
// CHECK:   dealloc_stack
// CHECK:   return
// CHECK-LABEL: } // end sil function '$s30existential_transform_soletype7test_026numbers5Int32VAE_tF{'


@_optimize(none) func test_02(number:Int32)->Int32 {
  var b:SPP
  if number < 5 {
    b = SCC()
  } else {
    b = SCC()
  }
  let x = test_01(b:b)
  return x
}

// -----------------------------------------------------------------------------
// rdar://163199428 (Compiler assertion when trying to box Int into any ~Copyable)
//
// Generate a thunk specialized existential argument that consumes a ~Copyable.

// CHECK-LABEL: sil [signature_optimized_thunk] [heuristic_always_inline] @$s30existential_transform_soletype7test_03ySiSgypRi_s_XPnF : $@convention(thin) (@in any ~Copyable) -> Optional<Int> {
// CHECK: bb0(%0 : $*any ~Copyable):
// CHECK: function_ref @$s30existential_transform_soletype7test_03ySiSgypRi_s_XPnFTf4e_n : $@convention(thin) <τ_0_0 where τ_0_0 : ~Copyable> (@in τ_0_0) -> Optional<Int>
// CHECK: [[OPEN:%[0-9]+]] = open_existential_addr mutable_access %0 to $*@opened("{{.*}}", any ~Copyable) Self
// CHECK-NEXT: apply %{{.*}}<@opened("{{.*}}", any ~Copyable) Self>([[OPEN]]) : $@convention(thin) <τ_0_0 where τ_0_0 : ~Copyable> (@in τ_0_0) -> Optional<Int>
// CHECK-NEXT: return

// specialized test_03(_:)
// CHECK-LABEL: sil shared @$s30existential_transform_soletype7test_03ySiSgypRi_s_XPnFTf4e_n : $@convention(thin) <τ_0_0 where τ_0_0 : ~Copyable> (@in τ_0_0) -> Optional<Int> {
public func test_03(_ t: consuming any ~Copyable) -> Int? {
  fatalError()
}

public func test_04(_ x: Int) -> Int? {
  test_03(x)
}
