// RUN: %target-typecheck-verify-swift -enable-experimental-feature Lifetimes

// REQUIRES: swift_feature_Lifetimes

// See also lifetime_depend_override_coroutine_accessors.swift.

// rdar://192737567: class overrides must have the same lifetime dependencies
// as the overridden declaration.

struct NE: ~Escapable {
  @_lifetime(immortal) init() {}
}

class Base {
  @_lifetime(borrow ne) func borrowing(_ ne: NE) -> NE { ne } // expected-note {{overridden declaration is here}}
  @_lifetime(copy ne) func copying(_ ne: NE) -> NE { ne } // expected-note {{overridden declaration is here}}
  @_lifetime(copy ne) func matching(_ ne: NE) -> NE { ne }
  @_lifetime(copy a) func twoParams(_ a: NE, _ b: NE) -> NE { a } // expected-note {{overridden declaration is here}}
  @_lifetime(immortal) func immortal(_ ne: NE) -> NE { NE() } // expected-note {{overridden declaration is here}}
  @_lifetime(copy ne) class func classMethod(_ ne: NE) -> NE { ne } // expected-note {{potential overridden class method 'classMethod' here}}
  @_lifetime(a: copy b) func inoutTarget(_ a: inout NE, _ b: NE) { a = b } // expected-note {{overridden declaration is here}}
  @_lifetime(a: copy b) func inoutMatching(_ a: inout NE, _ b: NE) { a = b }
  func inferred(_ ne: NE) -> NE { ne }

  // An override with fewer dependencies is safe, but is rejected.
  @_lifetime(copy ne) func fewerImmortal(_ ne: NE) -> NE { ne } // expected-note {{overridden declaration is here}}
  @_lifetime(copy a, copy b) func fewerParams(_ a: NE, _ b: NE) -> NE { a } // expected-note {{overridden declaration is here}}

  @_lifetime(a: copy b) init(_ a: inout NE, _ b: NE) { a = b } // expected-note {{overridden declaration is here}}
  @_lifetime(a: copy b) init(matching a: inout NE, _ b: NE) { a = b }

  var property: NE {
    @_lifetime(immortal) get { NE() } // expected-note {{overridden declaration is here}}
  }

  // The dependence target of a '_read' accessor is its yielded value.
  subscript(read ne: NE) -> NE {
    @_lifetime(borrow self) _read { yield NE() } // expected-note {{overridden declaration is here}}
  }

  var readFewer: NE {
    @_lifetime(borrow self) _read { yield NE() } // expected-note {{overridden declaration is here}}
  }

  var readMatching: NE {
    @_lifetime(borrow self) _read { yield NE() }
  }

  subscript(ne: NE) -> NE {
    @_lifetime(copy ne) get { ne } // expected-note {{overridden declaration is here}}
  }
}

class Derived: Base {
  @_lifetime(copy ne) override func borrowing(_ ne: NE) -> NE { ne } // expected-error {{instance method 'borrowing' has a lifetime dependence that does not match the overridden declaration}}
  @_lifetime(borrow ne) override func copying(_ ne: NE) -> NE { ne } // expected-error {{instance method 'copying' has a lifetime dependence that does not match the overridden declaration}}
  @_lifetime(copy ne) override func matching(_ ne: NE) -> NE { ne }
  @_lifetime(copy a, copy b) override func twoParams(_ a: NE, _ b: NE) -> NE { a } // expected-error {{instance method 'twoParams' has a lifetime dependence that does not match the overridden declaration}}
  @_lifetime(copy ne) override func immortal(_ ne: NE) -> NE { ne } // expected-error {{instance method 'immortal' has a lifetime dependence that does not match the overridden declaration}}
  // Lifetime dependencies of static methods are part of the compared type.
  @_lifetime(borrow ne) override class func classMethod(_ ne: NE) -> NE { ne } // expected-error {{method does not override any method from its superclass}}
  @_lifetime(a: borrow b) override func inoutTarget(_ a: inout NE, _ b: NE) {} // expected-error {{instance method 'inoutTarget' has a lifetime dependence that does not match the overridden declaration}}
  @_lifetime(a: copy b) override func inoutMatching(_ a: inout NE, _ b: NE) { a = b }
  @_lifetime(copy ne) override func inferred(_ ne: NE) -> NE { ne }

  @_lifetime(immortal) override func fewerImmortal(_ ne: NE) -> NE { NE() } // expected-error {{instance method 'fewerImmortal' has a lifetime dependence that does not match the overridden declaration}}
  @_lifetime(copy a) override func fewerParams(_ a: NE, _ b: NE) -> NE { a } // expected-error {{instance method 'fewerParams' has a lifetime dependence that does not match the overridden declaration}}

  @_lifetime(a: borrow b) override init(_ a: inout NE, _ b: NE) { a = b } // expected-error {{initializer 'init(_:_:)' has a lifetime dependence that does not match the overridden declaration}}
  @_lifetime(a: copy b) override init(matching a: inout NE, _ b: NE) { a = b }

  override var property: NE {
    @_lifetime(borrow self) get { NE() } // expected-error {{getter for property 'property' has a lifetime dependence that does not match the overridden declaration}}
  }

  override subscript(read ne: NE) -> NE {
    @_lifetime(borrow self, copy ne) _read { yield ne } // expected-error {{_read accessor for subscript 'subscript(read:)' has a lifetime dependence that does not match the overridden declaration}}
  }

  override var readFewer: NE {
    @_lifetime(immortal) _read { yield NE() } // expected-error {{_read accessor for property 'readFewer' has a lifetime dependence that does not match the overridden declaration}}
  }

  override var readMatching: NE {
    @_lifetime(borrow self) _read { yield NE() }
  }

  override subscript(ne: NE) -> NE {
    @_lifetime(borrow ne) get { ne } // expected-error {{getter for subscript 'subscript(_:)' has a lifetime dependence that does not match the overridden declaration}}
  }
}

// The base's type is substituted with the subclass's generic arguments. A
// dependency whose target becomes Escapable is ignored.
class GenericBase<T: ~Escapable> {
  @_lifetime(copy t) func method(_ t: T) -> T { t } // expected-note 3 {{overridden declaration is here}}
  @_lifetime(copy t) class func classMethod(_ t: T) -> T { t }
  @_lifetime(copy u) func genericMethod<U: ~Escapable>(_ t: T, _ u: U) -> U { u } // expected-note {{overridden declaration is here}}
}

class EscapableArgument: GenericBase<Int> {
  override func method(_ t: Int) -> Int { t }
  override class func classMethod(_ t: Int) -> Int { t }
  // 'U' is still ~Escapable.
  @_lifetime(borrow u) override func genericMethod<U: ~Escapable>(_ t: Int, _ u: U) -> U { u } // expected-error {{instance method 'genericMethod' has a lifetime dependence that does not match the overridden declaration}}
}

class NonEscapableArgument: GenericBase<NE> {
  @_lifetime(borrow t) override func method(_ t: NE) -> NE { t } // expected-error {{instance method 'method' has a lifetime dependence that does not match the overridden declaration}}
}

class NonEscapableArgumentMatching: GenericBase<NE> {
  @_lifetime(copy t) override func method(_ t: NE) -> NE { t }
}

class GenericEscapableArgument<V>: GenericBase<V> {
  override func method(_ t: V) -> V { t }
}

class GenericNonEscapableArgument<V: ~Escapable>: GenericBase<V> {
  @_lifetime(borrow t) override func method(_ t: V) -> V { t } // expected-error {{instance method 'method' has a lifetime dependence that does not match the overridden declaration}}
}

class GenericEscapableSubclass<V>: GenericBase<NE> {
  @_lifetime(borrow t) override func method(_ t: NE) -> NE { t } // expected-error {{instance method 'method' has a lifetime dependence that does not match the overridden declaration}}
}

// An override may be implemented with a different kind of accessor. A
// synthesized accessor has the dependence of the accessor that it wraps.
class AccessorKindBase {
  var immortalRead: NE { @_lifetime(immortal) _read { yield NE() } }
  var immortalGet: NE { @_lifetime(immortal) get { NE() } }
  subscript(copyRead ne: NE) -> NE { @_lifetime(copy ne) _read { yield ne } }
  subscript(copyGet ne: NE) -> NE { @_lifetime(copy ne) get { ne } }
}

class AccessorKindDerived: AccessorKindBase {
  override var immortalRead: NE { @_lifetime(immortal) get { NE() } }
  override var immortalGet: NE { @_lifetime(immortal) _read { yield NE() } }
  override subscript(copyRead ne: NE) -> NE { @_lifetime(copy ne) get { ne } }
  override subscript(copyGet ne: NE) -> NE { @_lifetime(copy ne) _read { yield ne } }
}

// A restated protocol requirement must have the same lifetime dependencies.
protocol P1 {
  @_lifetime(copy ne) func method(_ ne: NE) -> NE // expected-note {{overridden declaration is here}}
  @_lifetime(copy ne) func matching(_ ne: NE) -> NE
}

protocol P2: P1 {
  @_lifetime(borrow ne) func method(_ ne: NE) -> NE // expected-error {{instance method 'method' has a lifetime dependence that does not match the overridden declaration}}
  @_lifetime(copy ne) func matching(_ ne: NE) -> NE
}
