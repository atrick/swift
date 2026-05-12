// RUN: %target-swift-emit-silgen -module-name partial_apply_lifetime -enable-experimental-feature Lifetimes %s | %FileCheck %s

// These tests exercise the lifetime dependencies computed for partial_apply
// result types by LifetimeDependenceInfo::partialApply. Each case pins down
// exactly what partialApply must produce by declaring an explicit
// @_lifetime(...) on the callee's closure parameter and then checking the
// convert_escape_to_noescape target type (which is the partial_apply's result
// type, minus the @noescape attribute). A convert_function between the
// partial_apply and the consuming apply would mean partialApply disagreed with
// the callee's expected type, so `CHECK-NOT: convert_function` is added as an
// extra guard.

struct NE: ~Escapable {
  @_lifetime(immortal)
  init() {}
}

// -----------------------------------------------------------------------------
// Baseline: a single-capture closure with a borrow-on-capture result.
// Lifted closure: (NE) -> @lifetime(borrow 0) NE
// Bind 1 captured NE -> () -> @lifetime(captures) NE
// Verifies: target=result is remapped, all-bound scope source collapses to
// nullptr, and the captures flag is set.
// -----------------------------------------------------------------------------

@_lifetime(copy f)
func copyNE(f: () -> NE) -> NE {
  f()
}

// CHECK-LABEL: sil hidden [ossa] @$s22partial_apply_lifetime9callGetNE3ne1AA0F0VAE_tF : $@convention(thin) (@guaranteed NE) -> @lifetime(copy 0) @owned NE {
// CHECK: [[FNREF:%[0-9]+]] = function_ref @$s22partial_apply_lifetime9callGetNE3ne1AA0F0VAE_tFAEyXEfU_ : $@convention(thin) (@guaranteed NE) -> @lifetime(borrow 0) @owned NE
// CHECK: [[CLOSURE:%[0-9]+]] = partial_apply [callee_guaranteed] [[FNREF]]
// CHECK-NOT: convert_function
// CHECK: [[NECLOSURE:%[0-9]+]] = convert_escape_to_noescape [not_guaranteed] [[CLOSURE]] to $@noescape @callee_guaranteed () -> @lifetime(captures) @owned NE
// CHECK: [[GETNE:%[0-9]+]] = function_ref @$s22partial_apply_lifetime6copyNE1fAA0E0VAEyXE_tF
// CHECK: apply [[GETNE]]([[NECLOSURE]])
// CHECK-LABEL: } // end sil function '$s22partial_apply_lifetime9callGetNE3ne1AA0F0VAE_tF'
func callGetNE(ne1: NE) -> NE {
  copyNE { ne1 }
}

// -----------------------------------------------------------------------------
// Dependency on an unbound formal parameter: the source index is kept and no
// captures flag is added.
//
// Lifted closure: (NE, Bool) -> @lifetime(borrow 0) NE
// Bind 1 (the captured Bool) -> (NE) -> @lifetime(borrow 0) NE
// -----------------------------------------------------------------------------

@_lifetime(copy f)
func eatOneBorrow(f: @_lifetime(borrow ne) (_ ne: NE) -> NE) -> NE {
  let local = NE()
  return f(local)
}

// CHECK-LABEL: sil hidden [ossa] @$s22partial_apply_lifetime19callBorrowOnUnbound4condAA2NEVSb_tF :
// CHECK: [[FNREF:%[0-9]+]] = function_ref @$s22partial_apply_lifetime19callBorrowOnUnbound4condAA2NEVSb_tFA2EXEfU_ : $@convention(thin) (@guaranteed NE, Bool) -> @lifetime(borrow 0) @owned NE
// CHECK: [[CLOSURE:%[0-9]+]] = partial_apply [callee_guaranteed] [[FNREF]]
// CHECK-NOT: convert_function
// CHECK: convert_escape_to_noescape [not_guaranteed] [[CLOSURE]] to $@noescape @callee_guaranteed (@guaranteed NE) -> @lifetime(borrow 0) @owned NE
// CHECK-LABEL: } // end sil function '$s22partial_apply_lifetime19callBorrowOnUnbound4condAA2NEVSb_tF'
func callBorrowOnUnbound(cond: Bool) -> NE {
  eatOneBorrow { n in if cond { return n } else { return n } }
}

// -----------------------------------------------------------------------------
// Dependency on a bound-only source: the index list collapses to nullptr and
// the captures flag is set.
//
// Lifted closure: (NE, NE) -> @lifetime(borrow 1) NE
// Bind 1 (the captured NE) -> (NE) -> @lifetime(captures) NE
// -----------------------------------------------------------------------------

@_lifetime(copy f)
func eatOneCaptures(f: @_lifetime(captures) (NE) -> NE) -> NE {
  let local = NE()
  return f(local)
}

// CHECK-LABEL: sil hidden [ossa] @$s22partial_apply_lifetime14callDepOnBound5boundAA2NEVAE_tF :
// CHECK: [[FNREF:%[0-9]+]] = function_ref @$s22partial_apply_lifetime14callDepOnBound5boundAA2NEVAE_tFA2EXEfU_ : $@convention(thin) (@guaranteed NE, @guaranteed NE) -> @lifetime(borrow 1) @owned NE
// CHECK: [[CLOSURE:%[0-9]+]] = partial_apply [callee_guaranteed] [[FNREF]]
// CHECK-NOT: convert_function
// CHECK: convert_escape_to_noescape [not_guaranteed] [[CLOSURE]] to $@noescape @callee_guaranteed (@guaranteed NE) -> @lifetime(captures) @owned NE
// CHECK-LABEL: } // end sil function '$s22partial_apply_lifetime14callDepOnBound5boundAA2NEVAE_tF'
@_lifetime(copy bound)
func callDepOnBound(bound: NE) -> NE {
  eatOneCaptures { _ in bound }
}

// -----------------------------------------------------------------------------
// Dependency on a mix of bound and unbound sources: the bound bits are trimmed
// off and the captures flag is set, while the unbound bits remain.
//
// Lifted closure: (NE, Bool, NE) -> @lifetime(borrow 0, borrow 1, borrow 2) NE
// Bind 2 (the captured Bool and NE) ->
//   (NE) -> @lifetime(captures, borrow 0) NE
// -----------------------------------------------------------------------------

@_lifetime(copy f)
func eatOneCapturesAndBorrow(f: @_lifetime(captures, borrow ne) (_ ne: NE) -> NE) -> NE {
  let local = NE()
  return f(local)
}

// CHECK-LABEL: sil hidden [ossa] @$s22partial_apply_lifetime9callMixed5bound4condAA2NEVAF_SbtF :
// CHECK: [[FNREF:%[0-9]+]] = function_ref @$s22partial_apply_lifetime9callMixed5bound4condAA2NEVAF_SbtFA2FXEfU_ : $@convention(thin) (@guaranteed NE, Bool, @guaranteed NE) -> @lifetime(borrow 0, borrow 1, borrow 2) @owned NE
// CHECK: [[CLOSURE:%[0-9]+]] = partial_apply [callee_guaranteed] [[FNREF]]
// CHECK-NOT: convert_function
// CHECK: convert_escape_to_noescape [not_guaranteed] [[CLOSURE]] to $@noescape @callee_guaranteed (@guaranteed NE) -> @lifetime(captures, borrow 0) @owned NE
// CHECK-LABEL: } // end sil function '$s22partial_apply_lifetime9callMixed5bound4condAA2NEVAF_SbtF'
@_lifetime(copy bound)
func callMixed(bound: NE, cond: Bool) -> NE {
  eatOneCapturesAndBorrow { n in cond ? n : bound }
}

// -----------------------------------------------------------------------------
// Multiple bound parameters with every source bound: exercises
// numBoundParams > 1 in combination with the all-bound collapse path. Every
// captured parameter contributes a source, so the result ends up as a pure
// captures-only dependency.
//
// Lifted closure: (Bool, NE, NE) -> @lifetime(borrow 0, borrow 1, borrow 2) NE
// Bind 3 (the three captures) -> () -> @lifetime(captures) NE
// -----------------------------------------------------------------------------

@_lifetime(copy f)
func eatZeroCaptures(f: @_lifetime(captures) () -> NE) -> NE {
  f()
}

// CHECK-LABEL: sil hidden [ossa] @$s22partial_apply_lifetime18callBindMultiBound1a1b4condAA2NEVAG_AGSbtF :
// CHECK: [[FNREF:%[0-9]+]] = function_ref @$s22partial_apply_lifetime18callBindMultiBound1a1b4condAA2NEVAG_AGSbtFAGyXEfU_ : $@convention(thin) (Bool, @guaranteed NE, @guaranteed NE) -> @lifetime(borrow 0, borrow 1, borrow 2) @owned NE
// CHECK: [[CLOSURE:%[0-9]+]] = partial_apply [callee_guaranteed] [[FNREF]]
// CHECK-NOT: convert_function
// CHECK: convert_escape_to_noescape [not_guaranteed] [[CLOSURE]] to $@noescape @callee_guaranteed () -> @lifetime(captures) @owned NE
// CHECK-LABEL: } // end sil function '$s22partial_apply_lifetime18callBindMultiBound1a1b4condAA2NEVAG_AGSbtF'
@_lifetime(copy a, copy b)
func callBindMultiBound(a: NE, b: NE, cond: Bool) -> NE {
  eatZeroCaptures { cond ? a : b }
}

// -----------------------------------------------------------------------------
// Multiple bound parameters with the dependency on the unbound formal: the
// partialApply transform should leave the source index untouched and not set
// captures.
//
// Lifted closure: (NE, Bool, Int) -> @lifetime(borrow 0) NE
// Bind 2 (cond and tag) -> (NE) -> @lifetime(borrow 0) NE
// -----------------------------------------------------------------------------

// CHECK-LABEL: sil hidden [ossa] @$s22partial_apply_lifetime20callBindMultiUnbound4cond3tagAA2NEVSb_SitF :
// CHECK: [[FNREF:%[0-9]+]] = function_ref @$s22partial_apply_lifetime20callBindMultiUnbound4cond3tagAA2NEVSb_SitFA2FXEfU_ : $@convention(thin) (@guaranteed NE, Bool, Int) -> @lifetime(borrow 0) @owned NE
// CHECK: [[CLOSURE:%[0-9]+]] = partial_apply [callee_guaranteed] [[FNREF]]
// CHECK-NOT: convert_function
// CHECK: convert_escape_to_noescape [not_guaranteed] [[CLOSURE]] to $@noescape @callee_guaranteed (@guaranteed NE) -> @lifetime(borrow 0) @owned NE
// CHECK-LABEL: } // end sil function '$s22partial_apply_lifetime20callBindMultiUnbound4cond3tagAA2NEVSb_SitF'
@_lifetime(immortal)
func callBindMultiUnbound(cond: Bool, tag: Int) -> NE {
  eatOneBorrow { n in
    if cond && tag > 0 { return n } else { return n }
  }
}

// -----------------------------------------------------------------------------
// @_lifetime(immortal) should round-trip through partialApply unchanged. The
// immortal entry has no source index lists (all nullptr), so the captures
// flag must not be set.
//
// Lifted closure: (Int) -> @lifetime(immortal) NE
// Bind 1 -> () -> @lifetime(immortal) NE
// -----------------------------------------------------------------------------

@_lifetime(copy f)
func eatImmortal(f: @_lifetime(immortal) () -> NE) -> NE {
  f()
}

// CHECK-LABEL: sil hidden [ossa] @$s22partial_apply_lifetime12callImmortal5extraAA2NEVSi_tF :
// CHECK: [[FNREF:%[0-9]+]] = function_ref @$s22partial_apply_lifetime12callImmortal5extraAA2NEVSi_tFAEyXEfU_ : $@convention(thin) (Int) -> @lifetime(immortal) @owned NE
// CHECK: [[CLOSURE:%[0-9]+]] = partial_apply [callee_guaranteed] [[FNREF]]
// CHECK-NOT: convert_function
// CHECK: convert_escape_to_noescape [not_guaranteed] [[CLOSURE]] to $@noescape @callee_guaranteed () -> @lifetime(immortal) @owned NE
// CHECK-NOT: captures
// CHECK-LABEL: } // end sil function '$s22partial_apply_lifetime12callImmortal5extraAA2NEVSi_tF'
@_lifetime(immortal)
func callImmortal(extra: Int) -> NE {
  eatImmortal { let _ = extra; return NE() }
}

// -----------------------------------------------------------------------------
// Mixed dependency kinds (inherit + scope) within a single entry: exercises
// that captureBoundParams is applied independently to each index list so that
// an unbound inherit index is preserved while the corresponding scope list
// collapses and sets the captures flag.
//
// Lifted closure: (NE, Bool, NE) -> @lifetime(copy 0, borrow 1, borrow 2) NE
// Bind 2 (cond and the captured NE) ->
//   (NE) -> @lifetime(captures, copy 0) NE
// -----------------------------------------------------------------------------

@_lifetime(copy f)
func eatOneCapturesCopy(f: @_lifetime(captures, copy ne) (_ ne: NE) -> NE) -> NE {
  let local = NE()
  return f(local)
}

// CHECK-LABEL: sil hidden [ossa] @$s22partial_apply_lifetime14callMixedKinds5bound4condAA2NEVAF_SbtF :
// CHECK: [[FNREF:%[0-9]+]] = function_ref @$s22partial_apply_lifetime14callMixedKinds5bound4condAA2NEVAF_SbtFA2FXEfU_ : $@convention(thin) (@guaranteed NE, Bool, @guaranteed NE) -> @lifetime(copy 0, borrow 1, borrow 2) @owned NE
// CHECK: [[CLOSURE:%[0-9]+]] = partial_apply [callee_guaranteed] [[FNREF]]
// CHECK-NOT: convert_function
// CHECK: convert_escape_to_noescape [not_guaranteed] [[CLOSURE]] to $@noescape @callee_guaranteed (@guaranteed NE) -> @lifetime(captures, copy 0) @owned NE
// CHECK-LABEL: } // end sil function '$s22partial_apply_lifetime14callMixedKinds5bound4condAA2NEVAF_SbtF'
@_lifetime(copy bound)
func callMixedKinds(bound: NE, cond: Bool) -> NE {
  eatOneCapturesCopy { n in cond ? n : bound }
}
