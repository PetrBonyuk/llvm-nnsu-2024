// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/KosarevEgorFMAPass%shlibext --pass-pipeline="builtin.module(llvm.func(KosarevEgorFMAPass))" %s | FileCheck %s

module attributes {dlti.dl_spec = #dlti.dl_spec<#dlti.dl_entry<!llvm.ptr<272>, dense<64> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<271>, dense<32> : vector<4xi32>>, #dlti.dl_entry<!llvm.ptr<270>, dense<32> : vector<4xi32>>, #dlti.dl_entry<f128, dense<128> : vector<2xi32>>, #dlti.dl_entry<i64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f80, dense<128> : vector<2xi32>>, #dlti.dl_entry<i8, dense<8> : vector<2xi32>>, #dlti.dl_entry<i1, dense<8> : vector<2xi32>>, #dlti.dl_entry<!llvm.ptr, dense<64> : vector<4xi32>>, #dlti.dl_entry<f64, dense<64> : vector<2xi32>>, #dlti.dl_entry<f16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i16, dense<16> : vector<2xi32>>, #dlti.dl_entry<i32, dense<32> : vector<2xi32>>, #dlti.dl_entry<"dlti.stack_alignment", 128 : i32>, #dlti.dl_entry<"dlti.endianness", "little">>} {
  // CHECK-LABEL: @_Z8funcZeroddd
  llvm.func local_unnamed_addr @_Z8funcZeroddd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.fmul %arg0, %arg2  : f64
    %1 = llvm.fadd %0, %arg1  : f64
    llvm.return %1 : f64

    // CHECK: %0 = math.fma %arg0, %arg2, %arg1 : f64
    // CHECK-NEXT: llvm.return %0 : f64
  }

  // CHECK-LABEL: @_Z7funcOnedddd
  llvm.func local_unnamed_addr @_Z7funcOnedddd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}, %arg3: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.fmul %arg0, %arg1  : f64
    %1 = llvm.fadd %0, %arg2  : f64
    %2 = llvm.fmul %arg2, %arg3  : f64
    %3 = llvm.fadd %1, %2  : f64
    llvm.return %3 : f64

    // CHECK: %0 = math.fma %arg0, %arg1, %arg2 : f64
    // CHECK-NEXT: %1 = math.fma %arg2, %arg3, %0 : f64
    // CHECK-NEXT: llvm.return %1 : f64
  }

  // CHECK-LABEL: @_Z7funcTwoddd
  llvm.func local_unnamed_addr @_Z7funcTwoddd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.fmul %arg1, %arg2  : f64
    %1 = llvm.fadd %arg0, %arg1  : f64
    %2 = llvm.fsub %1, %arg2  : f64
    %3 = llvm.fadd %0, %arg0  : f64
    %4 = llvm.fadd %2, %3  : f64
    llvm.return %4 : f64

    // CHECK: %0 = llvm.fadd %arg0, %arg1  : f64
    // CHECK-NEXT: %1 = llvm.fsub %0, %arg2  : f64
    // CHECK-NEXT: %2 = math.fma %arg1, %arg2, %arg0 : f64
    // CHECK-NEXT: %3 = llvm.fadd %1, %2  : f64
    // CHECK-NEXT: llvm.return %3 : f64
  }

  // CHECK-LABEL: @_Z9funcThreeddd
  llvm.func local_unnamed_addr @_Z9funcThreeddd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.mlir.constant(-2.000000e+00 : f64) : f64
    %1 = llvm.fmul %arg0, %arg2  : f64
    %2 = llvm.fadd %1, %0  : f64
    %3 = llvm.fadd %2, %arg1  : f64
    llvm.return %3 : f64

    // CHECK: %0 = llvm.mlir.constant(-2.000000e+00 : f64) : f64
    // CHECK-NEXT: %1 = math.fma %arg0, %arg2, %0 : f64
    // CHECK-NEXT: %2 = llvm.fadd %1, %arg1  : f64
    // CHECK-NEXT: llvm.return %2 : f64
  }

  // CHECK-LABEL: @_Z8funcFourddd
  llvm.func local_unnamed_addr @_Z8funcFourddd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.fmul %arg0, %arg2  : f64
    %1 = llvm.fmul %0, %arg1  : f64
    %2 = llvm.fadd %0, %arg0  : f64
    %3 = llvm.fadd %2, %1  : f64
    llvm.return %3 : f64

    // CHECK: %0 = llvm.fmul %arg0, %arg2  : f64
    // CHECK-NEXT: %1 = llvm.fadd %0, %arg0  : f64
    // CHECK-NEXT: %2 = math.fma %0, %arg1, %1 : f64
    // CHECK-NEXT: llvm.return %2 : f64
  }

  // CHECK-LABEL: @_Z8funcFiveddd
  llvm.func local_unnamed_addr @_Z8funcFiveddd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.fmul %arg0, %arg1  : f64
    %1 = llvm.fadd %0, %arg2  : f64
    llvm.return %1 : f64

    // CHECK: %0 = math.fma %arg0, %arg1, %arg2 : f64
    // CHECK-NEXT: llvm.return %0 : f64
  }

  // CHECK-LABEL: @_Z7funcSixddd
  llvm.func local_unnamed_addr @_Z7funcSixddd(%arg0: f64 {llvm.noundef}, %arg1: f64 {llvm.noundef}, %arg2: f64 {llvm.noundef}) -> (f64 {llvm.noundef}) attributes {memory = #llvm.memory_effects<other = none, argMem = none, inaccessibleMem = none>, passthrough = ["mustprogress", "nofree", "norecurse", "nosync", "nounwind", "willreturn", ["uwtable", "2"], ["min-legal-vector-width", "0"], ["no-trapping-math", "true"], ["stack-protector-buffer-size", "8"], ["target-cpu", "x86-64"], ["target-features", "+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87"], ["tune-cpu", "generic"]]} {
    %0 = llvm.fmul %arg0, %arg1  : f64
    %1 = llvm.fadd %0, %arg2  : f64
    %2 = llvm.fadd %0, %1  : f64
    llvm.return %2 : f64

    // CHECK: %0 = llvm.fmul %arg0, %arg1  : f64
    // CHECK-NEXT: %1 = llvm.fadd %0, %arg2  : f64
    // CHECK-NEXT: %2 = llvm.fadd %0, %1  : f64
    // CHECK-NEXT: llvm.return %2 : f64
  }
}
