// RUN: %target-swift-frontend -primary-file %s -parse-as-library -emit-sil \
// RUN:   -o /dev/null \
// RUN:   -verify \
// RUN:   -sil-verify-all \
// RUN:   -module-name test \
// RUN:   -define-availability "Span 0.1:macOS 9999, iOS 9999, watchOS 9999, tvOS 9999, visionOS 9999" \
// RUN:   -strict-memory-safety \
// RUN:   -enable-experimental-feature Lifetimes

// REQUIRES: swift_in_compiler
// REQUIRES: swift_feature_Lifetimes

@safe
@_silgen_name("getRawPointer")
func getRawPointer() -> UnsafeRawPointer

@safe
@_silgen_name("getMutRawPointer")
func getMutRawPointer() -> UnsafeMutableRawPointer {
  UnsafeMutableRawPointer(UnsafeMutablePointer<RawSpan>.allocate(capacity: 1))
}

//===----------------------------------------------------------------------===//
// raw pointer .load()
//
// TODO: test non-BitwiseCopyable loadUnaligned
//===----------------------------------------------------------------------===//

@available(Span 0.1, *)
@_lifetime(immortal)
func badLoad() -> RawSpan {
  let p = getRawPointer() // expected-error{{lifetime-dependent variable 'p' escapes its scope}}
    // expected-note@-1{{it depends on the lifetime of variable 'p'}}
  return unsafe p.load(as: RawSpan.self) // expected-note{{this use causes the lifetime-dependent value to escape}}
}

@available(Span 0.1, *)
@_lifetime(borrow p)
func goodLoad(p: UnsafeRawPointer<RawSpan>) -> RawSpan {
  unsafe useSpan(p.load(as: RawSpan.self))
  return unsafe p.load(as: RawSpan.self)
}

@available(Span 0.1, *)
@_lifetime(immortal)
func badLoadUnalignedBitwise() -> RawSpan {
  let p = getRawPointer() // expected-error{{lifetime-dependent variable 'p' escapes its scope}}
    // expected-note@-1{{it depends on the lifetime of variable 'p'}}
  return unsafe p.loadUnaligned(as: RawSpan.self) // expected-note{{this use causes the lifetime-dependent value to escape}}
}

@available(Span 0.1, *)
@_lifetime(borrow p)
func goodLoad(p: UnsafeRawPointer<RawSpan>) -> RawSpan {
  unsafe useSpan(p.loadUnaligned(as: RawSpan.self))
  return unsafe p.loadUnaligned(as: RawSpan.self)
}
