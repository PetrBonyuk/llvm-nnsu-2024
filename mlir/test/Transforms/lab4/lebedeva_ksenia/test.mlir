// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/LebedevaCallFuncCounter%shlibext --pass-pipeline="builtin.module(lebedeva-call-func-counter)" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func @_Z9function1i(%arg0: i32 {llvm.noundef}) -> (i1 {llvm.noundef, llvm.zeroext}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z9function1i
    // CHECK-SAME: attributes {"call-count" = 4 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2 : i32) : i32
    %2 = llvm.mlir.constant(0 : i32) : i32
    %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %4 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %5 = llvm.srem %4, %1  : i32
    %6 = llvm.icmp "eq" %5, %2 : i32
    llvm.return %6 : i1
  }
  llvm.func @_Z9function2ii(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) -> (i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z9function2ii
    // CHECK-SAME: attributes {"call-count" = 2 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x i8 {alignment = 1 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %4 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %5 = llvm.call @_Z9function1i(%4) : (i32) -> i1
    %6 = llvm.zext %5 : i1 to i8
    llvm.store %6, %3 {alignment = 1 : i64} : i8, !llvm.ptr
    %7 = llvm.load %3 {alignment = 1 : i64} : !llvm.ptr -> i8
    %8 = llvm.trunc %7 : i8 to i1
    %9 = llvm.zext %8 : i1 to i32
    %10 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %11 = llvm.add %9, %10  : i32
    llvm.return %11 : i32
  }
  llvm.func @_Z9function3ii(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z9function3ii
    // CHECK-SAME: attributes {"call-count" = 1 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %5 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %6 = llvm.call @_Z9function1i(%5) : (i32) -> i1
    %7 = llvm.zext %6 : i1 to i32
    llvm.store %7, %3 {alignment = 4 : i64} : i32, !llvm.ptr
    %8 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %9 = llvm.load %3 {alignment = 4 : i64} : !llvm.ptr -> i32
    %10 = llvm.call @_Z9function2ii(%8, %9) : (i32, i32) -> i32
    llvm.store %10, %4 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.return
  }
  llvm.func @extern_func(i32) -> i32
  // CHECK: llvm.func @extern_func
  // CHECK-SAME: attributes {"call-count" = 1 : i32
  llvm.func @_Z9function4ii(%arg0: i32 {llvm.noundef}, %arg1: i32 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    // CHECK: llvm.func @_Z9function4ii
    // CHECK-SAME: attributes {"call-count" = 0 : i32
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x i8 {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %4 = llvm.alloca %0 x i8 {alignment = 1 : i64} : (i32) -> !llvm.ptr
    %5 = llvm.alloca %0 x i32 {alignment = 4 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 4 : i64} : i32, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 4 : i64} : i32, !llvm.ptr
    %6 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %7 = llvm.call @_Z9function1i(%6) : (i32) -> i1
    %8 = llvm.zext %7 : i1 to i8
    %9 = llvm.call @extern_func(%6) : (i32) -> i32
    llvm.store %8, %3 {alignment = 1 : i64} : i8, !llvm.ptr
    %10 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %11 = llvm.call @_Z9function1i(%9) : (i32) -> i1
    %12 = llvm.zext %11 : i1 to i8
    llvm.store %12, %4 {alignment = 1 : i64} : i8, !llvm.ptr
    %13 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %14 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    %15 = llvm.call @_Z9function2ii(%13, %14) : (i32, i32) -> i32
    llvm.store %14, %5 {alignment = 4 : i64} : i32, !llvm.ptr
    %16 = llvm.load %1 {alignment = 4 : i64} : !llvm.ptr -> i32
    %17 = llvm.load %2 {alignment = 4 : i64} : !llvm.ptr -> i32
    llvm.call @_Z9function3ii(%16, %17) : (i32, i32) -> ()
    llvm.return
  }
}
