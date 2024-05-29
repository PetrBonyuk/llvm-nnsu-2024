; RUN: opt -load-pass-plugin=%llvmshlibdir/LebedevaBitwiseShiftPlugin%shlibext -passes=lebedeva-bitwise-shift -S %s | FileCheck %s

define dso_local noundef i32 @_Z4func1v() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 3, ptr %1, align 4
  store i32 6, ptr %2, align 4
  %3 = load i32, ptr %2, align 4
  %4 = mul nsw i32 16, %3
  ret i32 %4
  ; CHECK-LABEL: @_Z4func1v
  ; CHECK: %3 = load i32, ptr %2, align 4
  ; CHECK-NEXT: %4 = shl i32 %3, 4
  ; CHECK-NEXT: ret i32 %4
}

define dso_local noundef i32 @_Z4func2v() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 2, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 3, %3
  store i32 %4, ptr %2, align 4
  %5 = load i32, ptr %2, align 4
  %6 = mul nsw i32 3, %5
  ret i32 %6
  ; CHECK-LABEL: @_Z4func2v
  ; CHECK: %3 = load i32, ptr %1, align 4
  ; CHECK-NEXT: %4 = mul nsw i32 3, %3
  ; CHECK-NEXT: store i32 %4, ptr %2, align 4
  ; CHECK: %5 = load i32, ptr %2, align 4
  ; CHECK-NEXT: %6 = mul nsw i32 3, %5
  ; CHECK-NEXT: ret i32 %6
}

define dso_local noundef i32 @_Z4func3v() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 8, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 %3, 1
  store i32 %4, ptr %2, align 4
  %5 = load i32, ptr %2, align 4
  %6 = mul nsw i32 %5, 0
  ret i32 %6
  ; CHECK-LABEL: @_Z4func3v
  ; CHECK: %3 = load i32, ptr %1, align 4
  ; CHECK-NEXT: %4 = shl i32 %3, 0
  ; CHECK-NEXT: store i32 %4, ptr %2, align 4
  ; CHECK: %5 = load i32, ptr %2, align 4
  ; CHECK-NEXT: %6 = mul nsw i32 %5, 0
  ; CHECK-NEXT: ret i32 %6
}

define dso_local noundef i32 @_Z4func4v() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 -4, ptr %1, align 4
  store i32 2, ptr %2, align 4
  %3 = load i32, ptr %1, align 4
  %4 = load i32, ptr %2, align 4
  %5 = mul nsw i32 %3, %4
  store i32 %5, ptr %1, align 4
  %6 = load i32, ptr %1, align 4
  %7 = mul nsw i32 %6, 8
  ret i32 %7
  ; CHECK-LABEL: @_Z4func4v
  ; CHECK: %4 = load i32, ptr %2, align 4
  ; CHECK-NEXT: %5 = mul nsw i32 %3, %4
  ; CHECK-NEXT: store i32 %5, ptr %1, align 4
  ; CHECK: %6 = load i32, ptr %1, align 4
  ; CHECK-NEXT: %7 = shl i32 %6, 3
  ; CHECK-NEXT: ret i32 %7
}

define dso_local noundef i32 @_Z4func5v() #0 {
  %1 = alloca double, align 8
  %2 = alloca double, align 8
  store double 2.000000e+00, ptr %1, align 8
  %3 = load double, ptr %1, align 8
  %4 = fmul double %3, 4.000000e+00
  store double %4, ptr %2, align 8
  %5 = load double, ptr %2, align 8
  %6 = fmul double %5, 8.000000e+00
  %7 = fptosi double %6 to i32
  ret i32 %7
  ; CHECK-LABEL: @_Z4func5v
  ; CHECK: %3 = load double, ptr %1, align 8
  ; CHECK-NEXT: %4 = fmul double %3, 4.000000e+00
  ; CHECK-NEXT: store double %4, ptr %2, align 8
  ; CHECK: %5 = load double, ptr %2, align 8
  ; CHECK-NEXT: %6 = fmul double %5, 8.000000e+00
  ; CHECK-NEXT: %7 = fptosi double %6 to i32
}

define dso_local void @_Z4func6v() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 2, ptr %1, align 4
  %4 = load i32, ptr %1, align 4
  %5 = add nsw i32 %4, 8
  store i32 %5, ptr %2, align 4
  %6 = load i32, ptr %1, align 4
  %7 = sub nsw i32 8, %6
  store i32 %7, ptr %3, align 4
  %8 = load i32, ptr %3, align 4
  %9 = sdiv i32 %8, 8
  store i32 %9, ptr %1, align 4
  ret void
  ; CHECK-LABEL: @_Z4func6v
  ; CHECK: %4 = load i32, ptr %1, align 4
  ; CHECK-NEXT: %5 = add nsw i32 %4, 8
  ; CHECK-NEXT: store i32 %5, ptr %2, align 4
  ; CHECK: %6 = load i32, ptr %1, align 4
  ; CHECK-NEXT: %7 = sub nsw i32 8, %6
  ; CHECK-NEXT: store i32 %7, ptr %3, align 4
  ; CHECK: %8 = load i32, ptr %3, align 4
  ; CHECK-NEXT: %9 = sdiv i32 %8, 8
  ; CHECK-NEXT: store i32 %9, ptr %1, align 4
}
