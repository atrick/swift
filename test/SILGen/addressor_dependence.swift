// RUN: %target-swift-emit-silgen %s | %FileCheck %s

public class C {}

public struct BoxC {
  let storage: AnyObject?
  let pointer: UnsafePointer<C>

  subscript() -> C {
    unsafeAddress {
      pointer
    }
  }
}

// The addressor result must explicitly dependend on the apply's 'self' arg.
//
// CHECK-LABEL: sil [ossa] @$s20addressor_dependence21testAddressorLifetime3boxAA1CCAA4BoxCVn_tF : $@convention(thin) (@owned BoxC) -> @owned C {
// CHECK: bb0(%0 : @noImplicitCopy @_eagerMove @owned $BoxC):
// CHECK: [[MV:%.*]] = moveonlywrapper_to_copyable [guaranteed]
// CHECK: [[APPLY:%.*]] = apply %{{.*}}([[MV]]) : $@convention(method) (@guaranteed BoxC) -> UnsafePointer<C>
// CHECK: [[MD:%.*]] = mark_dependence [[APPLY]] : $UnsafePointer<C> on %10 : $BoxC
// CHECK: struct_extract [[MD]] : $UnsafePointer<C>, #UnsafePointer._rawValue
// CHECK-LABEL: } // end sil function '$s20addressor_dependence21testAddressorLifetime3boxAA1CCAA4BoxCVn_tF'
public func testAddressorLifetime(box: consuming BoxC) -> C {
  box[]
}
