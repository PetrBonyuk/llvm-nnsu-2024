// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/AkopyanDiv%shlibext --pass-pipeline="builtin.module(akopyan_divpass)" %s | FileCheck %s

module {
  llvm.func @ceildivsi(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK: llvm.func @ceildivsi(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
    // CHECK-NEXT: %[[add:.*]] = arith.addi %arg0, %arg1 : i32
    // CHECK-NEXT: %[[sub:.*]] = arith.subi %[[add:.*]], %c1_i32 : i32
    // CHECK-NEXT: %[[div:.*]] = arith.divsi %[[sub]], %arg1 : i32
    // CHECK-NOT: %0 = arith.ceildivsi %arg0, %arg1 : i32
    %0 = arith.ceildivsi %arg0, %arg1 : i32
    llvm.return %0 : i32
  }
  llvm.func @ceildivui(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK: llvm.func @ceildivui(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
    // CHECK-NEXT: %[[add:.*]] = arith.addi %arg0, %arg1 : i32
    // CHECK-NEXT: %[[sub:.*]] = arith.subi %[[add]], %c1_i32 : i32
    // CHECK-NEXT: %[[div:.*]] = arith.divui %[[sub]], %arg1 : i32
    // CHECK-NOT: %0 = arith.ceildivui %arg0, %arg1 : i32
    %0 = arith.ceildivui %arg0, %arg1 : i32
    llvm.return %0 : i32
  }
  llvm.func @example(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %0 = llvm.call @ceildivsi(%arg0, %arg1) : (i32, i32) -> i32
    %1 = llvm.call @ceildivui(%arg2, %arg3) : (i32, i32) -> i32
    %2 = arith.addi %0, %1 : i32
    llvm.return %2 : i32
  }
}