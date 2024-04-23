; RUN: opt -load-pass-plugin=%llvmshlibdir/VeselovBitwiseShiftMultiplier%shlibext -passes=vesel-mult-shift -S %s | FileCheck %s

;int f1(int a){
;    int c = a + 4;
;    return c;

define dso_local i32 @f1(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  %5 = add nsw i32 %4, 4
  store i32 %5, ptr %3, align 4
  %6 = load i32, ptr %3, align 4
  ret i32 %6
}

; CHECK-LABEL: @f1
; CHECK: %4 = load i32, ptr %2, align 4
; CHECK-NEXT: %5 = add nsw i32 %4, 4
; CHECK-NEXT: store i32 %5, ptr %3, align 4
;}


;int f2(int a){
;    int c = a * 4;
;    return c;

define dso_local i32 @f2(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, ptr %2, align 4
  %4 = load i32, ptr %2, align 4
  %5 = mul nsw i32 %4, 4
  store i32 %5, ptr %3, align 4
  %6 = load i32, ptr %3, align 4
  ret i32 %6
}

; CHECK-LABEL: @f2
; CHECK: %4 = load i32, ptr %2, align 4
; CHECK-NEXT: %5 = shl i32 %4, 2
; CHECK-NEXT: store i32 %5, ptr %3, align 4
;}


;void f3(){
;    int a = 3;
;    int c = a * 4;

define dso_local void @f3() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 3, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 %3, 4
  store i32 %4, ptr %2, align 4
  ret void
}

; CHECK-LABEL: @f3
; CHECK: %3 = load i32, ptr %1, align 4
; CHECK-NEXT: %4 = shl i32 %3, 2
; CHECK-NEXT: store i32 %4, ptr %2, align 4
;}


;void f4(){
;    int a = 3;
;    int c = 4 * a;

define dso_local void @f4() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 3, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 4, %3
  store i32 %4, ptr %2, align 4
  ret void
}

; CHECK-LABEL: @f4
; CHECK: %3 = load i32, ptr %1, align 4
; CHECK-NEXT: %4 = shl i32 %3, 2
; CHECK-NEXT: store i32 %4, ptr %2, align 4
;}


;void f5(){
;    int a = 4;
;    int c = a * 1;
;}

define dso_local void @f5() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 4, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 %3, 1
  store i32 %4, ptr %2, align 4
  ret void
}

; CHECK-LABEL: @f5
; CHECK: %3 = load i32, ptr %1, align 4
; CHECK-NEXT: %4 = shl i32 1, 2
; CHECK-NEXT: store i32 %4, ptr %2, align 4


;void f6(){
;    int a = 3;
;    int c = 0 * a;
;}

define dso_local void @f6() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 3, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 0, %3
  store i32 %4, ptr %2, align 4
  ret void
}

; CHECK-LABEL: @f6
; CHECK: %3 = load i32, ptr %1, align 4
; CHECK-NEXT: %4 = mul nsw i32 0, %3
; CHECK-NEXT: store i32 %4, ptr %2, align 4


;void f7(){
;    int a = 4;
;    int c = a * 3;
;}

define dso_local void @f7() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 4, ptr %1, align 4
  %3 = load i32, ptr %1, align 4
  %4 = mul nsw i32 %3, 3
  store i32 %4, ptr %2, align 4
  ret void
}

; CHECK-LABEL: @f7
; CHECK: %3 = load i32, ptr %1, align 4
; CHECK-NEXT: %4 = shl i32 3, 2
; CHECK-NEXT: store i32 %4, ptr %2, align 4
