// RUN: %empty-directory(%t)
// RUN: not %target-swift-frontend -emit-module -o %t/FortyTwo.swiftmodule -I %S/Inputs -enable-cxx-interop %s 2>&1 | %FileCheck %s

@_implementationOnly import DeclA
import DeclB

@_inlineable
public func callFortySomething() -> CInt {
  return getFortySomething()
}

// CHECK: 'getFortySomething' has different definitions in different modules
