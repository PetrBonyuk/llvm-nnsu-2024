// RUN: split-file %s %t
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/dephCounter%shlibext --pass-pipeline="builtin.module(func.func(VasilevDepthCounter))" %t/one.mlir | FileCheck %t/one.mlir

//--- one.mlir
// CHECK: func.func @one() attributes {max_region_depth = 1 : i32}
func.func @one() {
  func.return
}

//--- two.mlir
// CHECK: func.func @two() attributes {max_region_depth = 2 : i32}
func.func @two() {
    %cond = arith.constant 1 : i1
    %0 = scf.if %cond -> (i1) {
        scf.yield %cond : i1
    } else {
        scf.yield %cond : i1
    }
    func.return
}

//--- three.mlir
// CHECK: func.func @three() attributes {max_region_depth = 3 : i32}
func.func @three() {
    %cond = arith.constant 1 : i1
    %0 = scf.if %cond -> (i1) {
        %1 = scf.if %cond -> (i1) {
            scf.yield %cond : i1
        } else {
            scf.yield %cond : i1
        }
        scf.yield %cond : i1
    } else {
        scf.yield %cond : i1
    }
    func.return
}
