// RUN: split-file %s %t
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/SadikovMLIRDepthPass%shlibext --pass-pipeline="builtin.module(func.func(sadikov-mlir-depth-pass))" %t/func1.mlir | FileCheck %t/func1.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/SadikovMLIRDepthPass%shlibext --pass-pipeline="builtin.module(func.func(sadikov-mlir-depth-pass))" %t/func2.mlir | FileCheck %t/func2.mlir
// RUN: mlir-opt -load-pass-plugin=%mlir_lib_dir/SadikovMLIRDepthPass%shlibext --pass-pipeline="builtin.module(func.func(sadikov-mlir-depth-pass))" %t/func3.mlir | FileCheck %t/func3.mlir


//--- func1.mlir
module{
    // CHECK: func.func @func1() -> f32 attributes {depth_of_func = 1 : i32} {
    func.func @func1() -> f32 {
        %sum_0 = arith.constant 0.0 : f32
        return %sum_0 : f32
    }    
}

//--- func2.mlir
module{
    // CHECK: func.func @func2(%arg0: memref<1024xf32>, %arg1: index, %arg2: index, %arg3: index) -> f32 attributes {depth_of_func = 2 : i32} {
    func.func @func2(%buffer: memref<1024xf32>, %lb: index, %ub: index, %step: index) -> f32 {
        %sum_0 = arith.constant 0.0 : f32
        %sum = scf.for %iv = %lb to %ub step %step iter_args(%sum_iter = %sum_0) -> (f32) {
            %sum_next = arith.addf %sum_iter, %sum_iter : f32
            scf.yield %sum_next : f32
        }
        return %sum : f32
    }
}

//--- func3.mlir
module{ 
    // CHECK: func.func @func3(%arg0: memref<1024xf32>, %arg1: index, %arg2: index, %arg3: index) -> f32 attributes {depth_of_func = 3 : i32} {
    func.func @func3(%buffer: memref<1024xf32>, %lb: index, %ub: index, %step: index) -> f32 {
        %sum_0 = arith.constant 0.0 : f32
        %c0 = arith.constant 0.0 : f32
        %sum = scf.while (%arg1 = %c0) : (f32) -> f32 {
            %condition = arith.constant 1 : i1
            scf.condition(%condition) %arg1 : f32
        } do {
            ^bb0(%sum_iter: f32):
            %cond = arith.constant 1 : i1
            %sum_next = scf.if %cond -> (f32) {
                %new_sum = arith.addf %sum_iter, %sum_iter : f32
                scf.yield %new_sum : f32
            } else {
                scf.yield %sum_iter : f32
            }
            scf.yield %sum_next : f32
        }
        return %sum : f32
    }
}
