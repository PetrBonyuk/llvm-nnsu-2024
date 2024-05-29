// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/MortinaCilDivConvert%shlibext --pass-pipeline="builtin.module(llvm.func(mortina-cildiv-conv))" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func local_unnamed_addr @_Z5func1ii(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (i32 {llvm.noundef}) {
    // CHECK: %c1_i32 = arith.constant 1 : i32
    // CHECK-NEXT: %0 = arith.addi %arg0, %arg1  : i32
    // CHECK-NEXT: %1 = arith.subi %0, %c1_i32 : i32
    // CHECK-NEXT: %2 = arith.divsi %1, %arg1  : i32
    // CHECK-NOT: %3 = arith.ceildivsi %arg0, %arg1 : i32
    %0 = arith.ceildivsi %arg0, %arg1 : i32
    llvm.return %0 : i32
  }
  llvm.func local_unnamed_addr @_Z5func2jj(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (i32 {llvm.noundef}) {
    // CHECK: %c1_i32 = arith.constant 1 : i32
    // CHECK-NEXT: %0 = arith.addi %arg0, %arg1  : i32
    // CHECK-NEXT: %1 = arith.subi %0, %c1_i32 : i32
    // CHECK-NEXT: %2 = arith.divui %1, %arg1  : i32
    // CHECK-NOT: %3 = arith.ceildivui %arg0, %arg1 : i32
    %0 = arith.ceildivui %arg0, %arg1 : i32
    llvm.return %0 : i32
  }
}
