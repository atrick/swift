// RUN: %empty-directory(%t)

// RUN: %target-swift-frontend                           \
// RUN:     %s                                           \
// RUN:     -emit-silgen                                 \
// RUN:     -debug-diagnostic-names                      \
// RUN:     -I %t                                        \
// RUN: |                                                \
// RUN: %FileCheck %s

public struct FA<T> {
  public subscript(_ int: Int) -> T {
// CHECK-LABEL: sil [ossa] @read : {{.*}} {
                  // function_ref UnsafeMutablePointer.subscript.unsafeAddressor
// CHECK:         [[ADDRESSOR:%[^,]+]] = function_ref @$sSpsRi_zrlEyxSicilu
// CHECK:         [[POINTER:%[^,]+]] = apply [[ADDRESSOR]]
// CHECK:         [[MD:%.*]] = mark_dependence [[POINTER]] : $UnsafePointer<T>
// CHECK:         [[RAW_POINTER:%[^,]+]] = struct_extract [[MD]]
// CHECK:         [[ADDR:%[^,]+]] = pointer_to_address [[RAW_POINTER]]
// CHECK:         [[ACCESS:%[^,]+]] = begin_access [read] [unsafe] [[ADDR]]
// Verify that no spurious temporary is emitted.
// CHECK-NOT:     alloc_stack
// CHECK:         yield [[ACCESS]] : $*T, resume [[SUCCESS:bb[0-9]+]], unwind [[FAILURE:bb[0-9]+]]
// CHECK:       [[SUCCESS]]:
// CHECK:         end_access [[ACCESS]]
// CHECK:       [[FAILURE]]:
// CHECK:         end_access [[ACCESS]]
// CHECK:         unwind
// CHECK-LABEL: } // end sil function 'read'
    @_silgen_name("read")
    _read {
      yield ump[int]
    }
  }
  var ump: UnsafeMutablePointer<T>
}

