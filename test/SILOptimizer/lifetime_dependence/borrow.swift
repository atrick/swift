// RUN: %target-swift-frontend %s -emit-sil \
// RUN:   -o /dev/null \
// RUN:   -verify \
// RUN:   -sil-verify-all \
// RUN:   -enable-builtin-module \
// RUN:   -module-name test \
// RUN:   -enable-experimental-feature Lifetimes \
// RUN:   -enable-experimental-feature AddressableTypes \
// RUN:   -enable-experimental-feature AddressableParameters

// REQUIRES: swift_in_compiler
// REQUIRES: swift_feature_Lifetimes
// REQUIRES: swift_feature_AddressableTypes
// REQUIRES: swift_feature_AddressableParameters

// Test the lifetime dependency semantics of the proposed Borrow<T> type.

@frozen
@safe
public struct _Borrow<Value: ~Copyable>: Copyable, ~Escapable {
  @usableFromInline
  let _pointer: UnsafePointer<Value>

  @lifetime(borrow value)
  @_alwaysEmitIntoClient
  @_transparent
  public init(_ value: borrowing @_addressable Value) {
    unsafe _pointer = UnsafePointer(Builtin.unprotectedAddressOfBorrow(value))
  }

  @lifetime(borrow owner)
  @_alwaysEmitIntoClient
  @_transparent
  public init<Owner: ~Copyable & ~Escapable>(
    unsafeAddress: UnsafePointer<Value>,
    borrowing owner: borrowing Owner
  ) {
    unsafe _pointer = unsafeAddress
  }

  @lifetime(copy owner)
  @_alwaysEmitIntoClient
  @_transparent
  public init<Owner: ~Copyable & ~Escapable>(
    unsafeAddress: UnsafePointer<Value>,
    copying owner: borrowing Owner
  ) {
    unsafe _pointer = unsafeAddress
  }

  @_alwaysEmitIntoClient
  public subscript() -> T {
    @_transparent
    unsafeAddress {
      unsafe _pointer
    }
  }
}

func testReborrow<T: ~Copyable>(borrowed: borrowing Borrow<T>) -> Borrow<T> {
  Borrow(borrowed[])
}
