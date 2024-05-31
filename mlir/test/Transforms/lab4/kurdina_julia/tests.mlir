// RUN: split-file %s %t
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/KurdinaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(KurdinaMaxDepth))" %t/func1.mlir | FileCheck %t/func1.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/KurdinaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(KurdinaMaxDepth))" %t/func2.mlir | FileCheck %t/func2.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/KurdinaMaxDepth%shlibext --pass-pipeline="builtin.module(func.func(KurdinaMaxDepth))" %t/func3.mlir | FileCheck %t/func3.mlir

//--- func1.mlir
func.func @func1() {
// CHECK: func.func @func1() attributes {maxDepth = 1 : i32}
  func.return
}

//--- func2.mlir
func.func @func2() {
// CHECK: func.func @func2() attributes {maxDepth = 2 : i32}
    %cond = arith.constant 1 : i1
    %0 = scf.if %cond -> (i1) {
        scf.yield %cond : i1
    } else {
        scf.yield %cond : i1
    }
    func.return
}

//--- func3.mlir
func.func @func3() {
// CHECK: func.func @func3() attributes {maxDepth = 3 : i32}
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