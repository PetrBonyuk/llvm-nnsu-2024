// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/myArithCeilDiv%shlibext --pass-pipeline="builtin.module(func.func(BendArithCeilDiv))" %s | FileCheck %s

func.func @mrr(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) {
  // CHECK: func.func @mrr(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32)
  // CHECK-NEXT: %c-1_i32 = arith.constant -1 : i32
  // CHECK-NEXT: %0 = arith.addi %arg0, %arg1 : i32
  // CHECK-NEXT: %1 = arith.addi %0, %c-1_i32 : i32
  // CHECK-NEXT: %2 = arith.divsi %1, %arg1 : i32
  // CHECK-NOT: %si = arith.ceildivsi %arg0, %arg1 : i32
  %si = arith.ceildivsi %arg0, %arg1 : i32
  // CHECK: %c-1_i32_0 = arith.constant -1 : i32
  // CHECK-NEXT: %3 = arith.addi %arg2, %arg3 : i32
  // CHECK-NEXT: %4 = arith.addi %3, %c-1_i32_0 : i32
  // CHECK-NEXT: %5 = arith.divui %4, %arg3 : i32
  // CHECK-NOT: %ui = arith.ceildivui %arg2, %arg3 : i32
  // CHECK-NEXT: llvm.return
  %ui = arith.ceildivui %arg2, %arg3 : i32
  llvm.return
}
