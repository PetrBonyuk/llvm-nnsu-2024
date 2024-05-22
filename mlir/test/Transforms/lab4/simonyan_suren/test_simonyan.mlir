// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/SimonyanSurenFMAPass%shlibext --pass-pipeline="builtin.module(llvm.func(simonyan_suren_fma))" %s | FileCheck %s

// double func1(double a, double b) {
  // double constant1 = 2.0;
  // double constant2 = 5.0;
  // double result = a * constant1 + b * constant2;
  // return result;
// }

// double func2(double a, double b) {
  // double constant1 = 5.0;
  // double constant2 = 2.0;
  // double mul = a * b;
  // double add1 = mul + constant1;
  // double mul2 = mul * constant2;
  // double result = mul2 + add1;
  // return result;
// }

// double func4(double a, double b) {
  // double constant1 = 2.0;
  // double constant2 = -7.0;
  // double mul = a * constant1;
  // double result = mul + constant2;
  // return result;
// }


module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  llvm.func local_unnamed_addr @_Z5func1dd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %2 = llvm.fmul %arg0, %0  : f64
    %3 = llvm.fmul %arg1, %1  : f64
    %4 = llvm.fadd %2, %3  : f64
    llvm.return %4 : f64
    // CHECK: %2 = llvm.fmul %arg1, %1  : f64
    // CHECK-NEXT: %3 = llvm.intr.fma(%arg0, %0, %2)  : (f64, f64, f64) -> f64
    // CHECK-NEXT: llvm.return %3 : f64
  }

  llvm.func local_unnamed_addr @_Z5func2dd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(5.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %2 = llvm.fmul %arg0, %arg1  : f64
    %3 = llvm.fadd %2, %0  : f64
    %4 = llvm.fmul %2, %1  : f64
    %5 = llvm.fadd %4, %3  : f64
    llvm.return %5 : f64
    // CHECK: %3 = llvm.intr.fma(%arg0, %arg1, %0)  : (f64, f64, f64) -> f64
    // CHEK-NEXT: %4 = llvm.intr.fma(%2, %1, %3)  : (f64, f64, f64) -> f64
    // CHEK-NEXT: llvm.return %5 : f64
  }

  llvm.func local_unnamed_addr @_Z5func3dd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(2.000000e+00 : f64) : f64
    %1 = llvm.mlir.constant(-7.000000e+00 : f64) : f64
    %2 = llvm.fmul %arg0, %0  : f64
    %3 = llvm.fadd %2, %1  : f64
    llvm.return %3 : f64
    // CHECK: %2 = llvm.intr.fma(%arg0, %0, %1)  : (f64, f64, f64) -> f64
    // CHECK-NEXT: llvm.return %2 : f64
  }
}