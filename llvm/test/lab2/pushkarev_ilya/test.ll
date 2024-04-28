
; RUN: opt -load-pass-plugin %llvmshlibdir/PushkarevFunctionInliningPass%pluginext\
; RUN: -passes=pushkarev-function-inlining -S %s | FileCheck %s

;--------------------
;|      TEST 1      |
;--------------------

;void void_no_arg() //expect inline
;{
;    int a = 0;
;    a += 1;
;}

;int foo1(int num)//expect inline
;{
;    void_no_arg();
;    return num;
;}

define dso_local void @_Z11void_no_argv() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = load i32, i32* %1, align 4
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4
  ret void
}

define dso_local noundef i32 @_Z4foo1i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @_Z11void_no_argv()
  %3 = load i32, i32* %2, align 4
  ret i32 %3
}

; CHECK: define dso_local void @_Z11void_no_argv() {
; CHECK-NEXT:   %1 = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %1, align 4
; CHECK-NEXT:   %2 = load i32, ptr %1, align 4
; CHECK-NEXT:   %3 = add nsw i32 %2, 1
; CHECK-NEXT:   store i32 %3, ptr %1, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define dso_local noundef i32 @_Z4foo1i(i32 noundef %0) {
; CHECK: .split:
; CHECK-NEXT:   %1 = alloca i32, align 4
; CHECK-NEXT:   store i32 %0, ptr %1, align 4
; CHECK-NEXT:   br label %2

; CHECK: 2:                                                ; preds = %.split
; CHECK-NEXT:   %3 = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %3, align 4
; CHECK-NEXT:   %4 = load i32, ptr %3, align 4
; CHECK-NEXT:   %5 = add nsw i32 %4, 1
; CHECK-NEXT:   store i32 %5, ptr %3, align 4
; CHECK-NEXT:   br label %6

; CHECK: 6:                                                ; preds = %2
; CHECK-NEXT:   %7 = load i32, ptr %1, align 4
; CHECK-NEXT:   ret i32 %7
; CHECK-NEXT: }

;--------------------
;|      TEST 2      |
;--------------------

define dso_local void @_Z8void_argii(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %6 = load i32, i32* %3, align 4
  %7 = load i32, i32* %4, align 4
  %8 = add nsw i32 %6, %7
  store i32 %8, i32* %5, align 4
  ret void
}


define dso_local noundef i32 @_Z4foo2i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = load i32, i32* %2, align 4
  call void @_Z8void_argii(i32 noundef %3, i32 noundef %4)
  %5 = load i32, i32* %2, align 4
  ret i32 %5
}

; CHECK: define dso_local void @_Z8void_argii(i32 noundef %0, i32 noundef %1) {
; CHECK-NEXT:   %3 = alloca i32, align 4
; CHECK-NEXT:   %4 = alloca i32, align 4
; CHECK-NEXT:   %5 = alloca i32, align 4
; CHECK-NEXT:   store i32 %0, ptr %3, align 4
; CHECK-NEXT:   store i32 %1, ptr %4, align 4
; CHECK-NEXT:   %6 = load i32, ptr %3, align 4
; CHECK-NEXT:   %7 = load i32, ptr %4, align 4
; CHECK-NEXT:   %8 = add nsw i32 %6, %7
; CHECK-NEXT:   store i32 %8, ptr %5, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define dso_local noundef i32 @_Z4foo2i(i32 noundef %0) {
; CHECK-NEXT:   %2 = alloca i32, align 4
; CHECK-NEXT:   store i32 %0, ptr %2, align 4
; CHECK-NEXT:   %3 = load i32, ptr %2, align 4
; CHECK-NEXT:   %4 = load i32, ptr %2, align 4
; CHECK-NEXT:   call void @_Z8void_argii(i32 noundef %3, i32 noundef %4)
; CHECK-NEXT:   %5 = load i32, ptr %2, align 4
; CHECK-NEXT:   ret i32 %5
; CHECK-NEXT: }

;--------------------
;|      TEST 3      |
;--------------------

define dso_local noundef i32 @_Z10int_no_argv() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = load i32, i32* %1, align 4
  ret i32 %2
}


define dso_local noundef i32 @_Z4foo3i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = call noundef i32 @_Z10int_no_argv()
  store i32 %3, i32* %2, align 4
  %4 = load i32, i32* %2, align 4
  ret i32 %4
}

; CHECK: define dso_local noundef i32 @_Z10int_no_argv() {
; CHECK-NEXT:  %1 = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %1, align 4
; CHECK-NEXT:  %2 = load i32, ptr %1, align 4
; CHECK-NEXT:  ret i32 %2
; CHECK-NEXT:}

; CHECK: define dso_local noundef i32 @_Z4foo3i(i32 noundef %0) {
; CHECK-NEXT:  %2 = alloca i32, align 4
; CHECK-NEXT:  store i32 %0, ptr %2, align 4
; CHECK-NEXT:  %3 = call noundef i32 @_Z10int_no_argv()
; CHECK-NEXT:  store i32 %3, ptr %2, align 4
; CHECK-NEXT:  %4 = load i32, ptr %2, align 4
; CHECK-NEXT:  ret i32 %4
; CHECK-NEXT:}

;--------------------
;|      TEST 4      |
;--------------------

define dso_local noundef i32 @_Z7int_argii(i32 noundef %0, i32 noundef %1) #0 {
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 %0, i32* %3, align 4
  store i32 %1, i32* %4, align 4
  %6 = load i32, i32* %3, align 4
  %7 = load i32, i32* %4, align 4
  %8 = add nsw i32 %6, %7
  store i32 %8, i32* %5, align 4
  %9 = load i32, i32* %5, align 4
  ret i32 %9
}


define dso_local noundef i32 @_Z4foo4i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = load i32, i32* %2, align 4
  %5 = call noundef i32 @_Z7int_argii(i32 noundef %3, i32 noundef %4)
  %6 = load i32, i32* %2, align 4
  ret i32 %6
} 

; CHECK: define dso_local noundef i32 @_Z7int_argii(i32 noundef %0, i32 noundef %1) {
; CHECK-NEXT:  %3 = alloca i32, align 4
; CHECK-NEXT:  %4 = alloca i32, align 4
; CHECK-NEXT:  %5 = alloca i32, align 4
; CHECK-NEXT:  store i32 %0, ptr %3, align 4
; CHECK-NEXT:  store i32 %1, ptr %4, align 4
; CHECK-NEXT:  %6 = load i32, ptr %3, align 4
; CHECK-NEXT:  %7 = load i32, ptr %4, align 4
; CHECK-NEXT:  %8 = add nsw i32 %6, %7
; CHECK-NEXT:  store i32 %8, ptr %5, align 4
; CHECK-NEXT:  %9 = load i32, ptr %5, align 4
; CHECK-NEXT:  ret i32 %9
; CHECK-NEXT:}

; CHECK: define dso_local noundef i32 @_Z4foo4i(i32 noundef %0) {
; CHECK-NEXT:  %2 = alloca i32, align 4
; CHECK-NEXT:  store i32 %0, ptr %2, align 4
; CHECK-NEXT:  %3 = load i32, ptr %2, align 4
; CHECK-NEXT:  %4 = load i32, ptr %2, align 4
; CHECK-NEXT:  %5 = call noundef i32 @_Z7int_argii(i32 noundef %3, i32 noundef %4)
; CHECK-NEXT:  %6 = load i32, ptr %2, align 4
; CHECK-NEXT:  ret i32 %6
; CHECK-NEXT:}

;--------------------
;|      TEST 5      |
;--------------------

define dso_local void @_Z15void_calls_voidv() #0 {
  call void @_Z11void_no_argv()
  ret void
}


define dso_local noundef i32 @_Z4foo5i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @_Z15void_calls_voidv()
  %3 = load i32, i32* %2, align 4
  ret i32 %3
}

; CHECK: define dso_local void @_Z15void_calls_voidv() {
; CHECK: .split:
; CHECK-NEXT:  br label %0

; CHECK: 0:                                                ; preds = %.split
; CHECK-NEXT:  %1 = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %1, align 4
; CHECK-NEXT:  %2 = load i32, ptr %1, align 4
; CHECK-NEXT:  %3 = add nsw i32 %2, 1
; CHECK-NEXT:  store i32 %3, ptr %1, align 4
; CHECK-NEXT:  br label %4

; CHECK: 4:                                                ; preds = %0
; CHECK-NEXT:  ret void
; CHECK-NEXT:}

; CHECK: define dso_local noundef i32 @_Z4foo5i(i32 noundef %0) {
; CHECK: .split:
; CHECK-NEXT:  %1 = alloca i32, align 4
; CHECK-NEXT:  store i32 %0, ptr %1, align 4
; CHECK-NEXT:  br label %2

; CHECK: 2:                                                ; preds = %.split
; CHECK-NEXT:  br label %3

; CHECK: 3:                                                ; preds = %2
; CHECK-NEXT:  %4 = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %4, align 4
; CHECK-NEXT:  %5 = load i32, ptr %4, align 4
; CHECK-NEXT:  %6 = add nsw i32 %5, 1
; CHECK-NEXT:  store i32 %6, ptr %4, align 4
; CHECK-NEXT:  br label %7

; CHECK: 7:                                                ; preds = %3
; CHECK-NEXT:  br label %8

; CHECK: 8:                                                ; preds = %7
; CHECK-NEXT:  %9 = load i32, ptr %1, align 4
; CHECK-NEXT:  ret i32 %9
; CHECK-NEXT:}

;--------------------
;|      TEST 6      |
;--------------------
;
;void void_with_cycle()
;{
;    volatile int a;
;    for (int i = 0; i < 10; i++)
;    {
;        a = i;
;    }
;}
;
;void void_with_if()
;{
;    volatile int a = 3;
;    if (a%2)
;    {
;        a++;
;    }
;}
;
;int foo6(int num)//expect inline
;{
;    void_with_cycle();
;    void_with_if();
;    return num;
;}

define dso_local void @_Z15void_with_cyclev() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, i32* %2, align 4
  br label %3

3:                                                ; preds = %8, %0
  %4 = load i32, i32* %2, align 4
  %5 = icmp slt i32 %4, 10
  br i1 %5, label %6, label %11

6:                                                ; preds = %3
  %7 = load i32, i32* %2, align 4
  store volatile i32 %7, i32* %1, align 4
  br label %8

8:                                                ; preds = %6
  %9 = load i32, i32* %2, align 4
  %10 = add nsw i32 %9, 1
  store i32 %10, i32* %2, align 4
  br label %3, !llvm.loop !6

11:                                               ; preds = %3
  ret void
}


define dso_local void @_Z12void_with_ifv() #0 {
  %1 = alloca i32, align 4
  store volatile i32 3, i32* %1, align 4
  %2 = load volatile i32, i32* %1, align 4
  %3 = srem i32 %2, 2
  %4 = icmp ne i32 %3, 0
  br i1 %4, label %5, label %8

5:                                                ; preds = %0
  %6 = load volatile i32, i32* %1, align 4
  %7 = add nsw i32 %6, 1
  store volatile i32 %7, i32* %1, align 4
  br label %8

8:                                                ; preds = %5, %0
  ret void
}


define dso_local noundef i32 @_Z4foo6i(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @_Z15void_with_cyclev()
  call void @_Z12void_with_ifv()
  %3 = load i32, i32* %2, align 4
  ret i32 %3
}

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}

; CHECK:define dso_local void @_Z15void_with_cyclev() {
; CHECK-NEXT:  %1 = alloca i32, align 4
; CHECK-NEXT:  %2 = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %2, align 4
; CHECK-NEXT:  br label %3

; CHECK:3:                                                ; preds = %8, %0
; CHECK-NEXT:  %4 = load i32, ptr %2, align 4
; CHECK-NEXT:  %5 = icmp slt i32 %4, 10
; CHECK-NEXT:  br i1 %5, label %6, label %11

; CHECK:6:                                                ; preds = %3
; CHECK-NEXT:  %7 = load i32, ptr %2, align 4
; CHECK-NEXT:  store volatile i32 %7, ptr %1, align 4
; CHECK-NEXT:  br label %8

; CHECK:8:                                                ; preds = %6
; CHECK-NEXT:  %9 = load i32, ptr %2, align 4
; CHECK-NEXT:  %10 = add nsw i32 %9, 1
; CHECK-NEXT:  store i32 %10, ptr %2, align 4
; CHECK-NEXT:  br label %3, !llvm.loop !6

; CHECK:11:                                               ; preds = %3
; CHECK-NEXT:  ret void
; CHECK-NEXT:}

; CHECK:define dso_local void @_Z12void_with_ifv() {
; CHECK-NEXT:  %1 = alloca i32, align 4
; CHECK-NEXT:  store volatile i32 3, ptr %1, align 4
; CHECK-NEXT:  %2 = load volatile i32, ptr %1, align 4
; CHECK-NEXT:  %3 = srem i32 %2, 2
; CHECK-NEXT:  %4 = icmp ne i32 %3, 0
; CHECK-NEXT:  br i1 %4, label %5, label %8

; CHECK:5:                                                ; preds = %0
; CHECK-NEXT:  %6 = load volatile i32, ptr %1, align 4
; CHECK-NEXT:  %7 = add nsw i32 %6, 1
; CHECK-NEXT:  store volatile i32 %7, ptr %1, align 4
; CHECK-NEXT:  br label %8

; CHECK:8:                                                ; preds = %5, %0
; CHECK-NEXT:  ret void
; CHECK-NEXT:}

; CHECK:define dso_local noundef i32 @_Z4foo6i(i32 noundef %0) {
; CHECK:.split:
; CHECK-NEXT:  %1 = alloca i32, align 4
; CHECK-NEXT:  store i32 %0, ptr %1, align 4
; CHECK-NEXT:  br label %2

; CHECK:2:                                                ; preds = %.split
; CHECK-NEXT:  %3 = alloca i32, align 4
; CHECK-NEXT:  %4 = alloca i32, align 4
; CHECK-NEXT:  store i32 0, ptr %4, align 4
; CHECK-NEXT:  br label %5

; CHECK:5:                                                ; preds = %10, %2
; CHECK-NEXT:  %6 = load i32, ptr %4, align 4
; CHECK-NEXT:  %7 = icmp slt i32 %6, 10
; CHECK-NEXT:  br i1 %7, label %8, label %13

; CHECK:8:                                                ; preds = %5
; CHECK-NEXT:  %9 = load i32, ptr %4, align 4
; CHECK-NEXT:  store volatile i32 %9, ptr %3, align 4
; CHECK-NEXT:  br label %10

; CHECK:10:                                               ; preds = %8
; CHECK-NEXT:  %11 = load i32, ptr %4, align 4
; CHECK-NEXT:  %12 = add nsw i32 %11, 1
; CHECK-NEXT:  store i32 %12, ptr %4, align 4
; CHECK-NEXT:  br label %5, !llvm.loop !6

; CHECK:13:                                               ; preds = %5
; CHECK-NEXT:  br label %.split1

; CHECK:.split1:                                          ; preds = %13
; CHECK-NEXT:  br label %14

; CHECK:14:                                               ; preds = %.split1
; CHECK-NEXT:  %15 = alloca i32, align 4
; CHECK-NEXT:  store volatile i32 3, ptr %15, align 4
; CHECK-NEXT:  %16 = load volatile i32, ptr %15, align 4
; CHECK-NEXT:  %17 = srem i32 %16, 2
; CHECK-NEXT:  %18 = icmp ne i32 %17, 0
; CHECK-NEXT:  br i1 %18, label %19, label %22

; CHECK:19:                                               ; preds = %14
; CHECK-NEXT:  %20 = load volatile i32, ptr %15, align 4
; CHECK-NEXT:  %21 = add nsw i32 %20, 1
; CHECK-NEXT:  store volatile i32 %21, ptr %15, align 4
; CHECK-NEXT:  br label %22

; CHECK:22:                                               ; preds = %19, %14
; CHECK-NEXT:  br label %23

; CHECK:23:                                               ; preds = %22
; CHECK-NEXT:  %24 = load i32, ptr %1, align 4
; CHECK-NEXT:  ret i32 %24
; CHECK-NEXT:}