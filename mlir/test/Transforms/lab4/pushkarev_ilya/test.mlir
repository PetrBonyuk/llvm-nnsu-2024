// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/PushkarevCeilDiv%shlibext --pass-pipeline="builtin.module(llvm.func(pushkarev_ceildiv))" %s | FileCheck %s

module {
llvm.func @foo1(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK: llvm.func @foo1(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
    // CHECK-NEXT: %0 = arith.addi %arg0, %arg1 : i32
    // CHECK-NEXT: %1 = arith.subi %0, %c1_i32 : i32
    // CHECK-NEXT: %2 = arith.divsi %1, %arg1 : i32
    // CHECK-NOT: %0 = arith.ceildivsi %arg0, %arg1 : i32
    %0 = arith.ceildivsi %arg0, %arg1 : i32
    llvm.return %0 : i32
}
llvm.func @foo2(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK: llvm.func @foo2(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    // CHECK-NEXT: %c1_i32 = arith.constant 1 : i32
    // CHECK-NEXT: %0 = arith.addi %arg0, %arg1 : i32
    // CHECK-NEXT: %1 = arith.subi %0, %c1_i32 : i32
    // CHECK-NEXT: %2 = arith.divui %1, %arg1 : i32
    // CHECK-NOT: %0 = arith.ceildivui %arg0, %arg1 : i32
    %0 = arith.ceildivui %arg0, %arg1 : i32
    llvm.return %0 : i32
}
}