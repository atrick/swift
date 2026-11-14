// RUN: %target-typecheck-verify-swift -enable-experimental-feature Lifetimes \
// RUN:   -enable-experimental-feature CoroutineAccessors

// REQUIRES: swift_feature_Lifetimes
// REQUIRES: swift_feature_CoroutineAccessors

// rdar://192737567: the lifetime dependencies of an overriding coroutine
// accessor must match the overridden accessor. With CoroutineAccessors, '_read'
// is represented as 'yielding borrow'. See also lifetime_depend_override.swift.

struct NE: ~Escapable {
  @_lifetime(immortal) init() {}
}

class Base {
  // The dependence target of a coroutine accessor is its yielded value.
  subscript(read ne: NE) -> NE {
    @_lifetime(borrow self) _read { yield NE() } // expected-note {{overridden declaration is here}}
  }

  subscript(yielding ne: NE) -> NE {
    @_lifetime(borrow self) yielding borrow { yield NE() } // expected-note {{overridden declaration is here}}
  }

  var readFewer: NE {
    @_lifetime(borrow self) _read { yield NE() } // expected-note {{overridden declaration is here}}
  }

  var yieldingFewer: NE {
    @_lifetime(borrow self) yielding borrow { yield NE() } // expected-note {{overridden declaration is here}}
  }

  var readMatching: NE {
    @_lifetime(borrow self) _read { yield NE() }
  }

  var yieldingMatching: NE {
    @_lifetime(borrow self) yielding borrow { yield NE() }
  }
}

class Derived: Base {
  override subscript(read ne: NE) -> NE {
    @_lifetime(borrow self, copy ne) _read { yield ne } // expected-error {{_read accessor for subscript 'subscript(read:)' has a lifetime dependence that does not match the overridden declaration}}
  }

  override subscript(yielding ne: NE) -> NE {
    @_lifetime(borrow self, copy ne) yielding borrow { yield ne } // expected-error {{accessor for subscript 'subscript(yielding:)' has a lifetime dependence that does not match the overridden declaration}}
  }

  override var readFewer: NE {
    @_lifetime(immortal) _read { yield NE() } // expected-error {{_read accessor for property 'readFewer' has a lifetime dependence that does not match the overridden declaration}}
  }

  override var yieldingFewer: NE {
    @_lifetime(immortal) yielding borrow { yield NE() } // expected-error {{accessor for property 'yieldingFewer' has a lifetime dependence that does not match the overridden declaration}}
  }

  override var readMatching: NE {
    @_lifetime(borrow self) _read { yield NE() }
  }

  override var yieldingMatching: NE {
    @_lifetime(borrow self) yielding borrow { yield NE() }
  }
}
