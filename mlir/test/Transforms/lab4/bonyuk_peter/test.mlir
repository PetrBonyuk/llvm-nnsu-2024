// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/BonyukFusedMultiplyAddPass%shlibext --pass-pipeline="builtin.module(llvm.func(bonyuk_fused_multiply_add))" %s | FileCheck %s

// void functionone(double a, double b, double c){
//     double d = a * b + c;
// }

// void functiontwo(double a, double b){
//     double c = 2 + a * b;
// }

// void functionthree(double a){
//     double b = 2 + a * 8;
// }
module {
  llvm.func @functionone(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %2 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %1 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg1, %2 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg2, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %4 = llvm.load %1 {alignment = 8 : i64} : !llvm.ptr -> f64
    %5 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> f64
    %6 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64 
    %7 = llvm.fmul %4, %5 : f64
    %8 = llvm.fadd %6, %7 : f64
    // CHECK-NOT: %7 = llvm.fmul %4, %5 : f64
    // CHECK-NOT: %8 = llvm.fadd %6, %7 : f64
    // CHECK: %7 = llvm.intr.fma(%4, %5, %6) : (f64, f64, f64) -> f64
    llvm.return
  }
  llvm.func @functiontwo(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2.0 : f64) : f64
    %2 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %2 {alignment = 8 : i64} : f64, !llvm.ptr
    llvm.store %arg1, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %4 = llvm.load %2 {alignment = 8 : i64} : !llvm.ptr -> f64
    %5 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %6 = llvm.fmul %4, %5 : f64
    %7 = llvm.fadd %1, %6 : f64
    // CHECK-NOT: %6 = llvm.fmul %4, %5 : f64
    // CHECK-NOT: %7 = llvm.fadd %1, %6 : f64
    // CHECK: %6 = llvm.intr.fma(%4, %5, %1) : (f64, f64, f64) -> f64
    llvm.return
  }
llvm.func @functionthree(%arg0: f64 {llvm.noundef}) attributes {passthrough = ["mustprogress", "noinline", "nounwind", "optnone", ["uwtable", "2"], ["frame-pointer", "all"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x8-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(1 : i32) : i32
    %1 = llvm.mlir.constant(2.0 : f64) : f64
    %2 = llvm.mlir.constant(8.0 : f64) : f64
    %3 = llvm.alloca %0 x f64 {alignment = 8 : i64} : (i32) -> !llvm.ptr
    llvm.store %arg0, %3 {alignment = 8 : i64} : f64, !llvm.ptr
    %4 = llvm.load %3 {alignment = 8 : i64} : !llvm.ptr -> f64
    %5 = llvm.fmul %4, %2 : f64
    %6 = llvm.fadd %5, %1 : f64
    // CHECK-NOT: %6 = llvm.fadd %5, %1 : f64
    // CHECK-NOT: %5 = llvm.fmul %4, %2 : f64
    // CHECK: %5 = llvm.intr.fma(%4, %2, %1) : (f64, f64, f64) -> f64
    llvm.return
}
}