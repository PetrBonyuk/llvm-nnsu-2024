; RUN: split-file %s %t
; RUN: opt -passes="vyunov-magic-inline" -S %t/test1.ll | FileCheck %t/test1.ll
; RUN: opt -passes="vyunov-magic-inline" -S %t/test2.ll | FileCheck %t/test2.ll
; RUN: opt -passes="vyunov-magic-inline" -S %t/test3.ll | FileCheck %t/test3.ll
; RUN: opt -passes="vyunov-magic-inline" -S %t/test4.ll | FileCheck %t/test4.ll
; RUN: opt -passes="vyunov-magic-inline" -S %t/test5.ll | FileCheck %t/test5.ll
; RUN: opt -passes="vyunov-magic-inline" -S %t/test6.ll | FileCheck %t/test6.ll
; RUN: opt -passes="vyunov-magic-inline" -S %t/test7.ll | FileCheck %t/test7.ll
; RUN: opt -passes="vyunov-magic-inline" -S %t/test8.ll | FileCheck %t/test8.ll

;--- test1.ll
; COM: Simple magic inline check. Expect inline

;void foo() {
;  float a = 1.0f;
;  a += 1.0f;
;}
;
;void bar() {
;  int a = 0;
;  foo();
;  a++;
;}

; COM: The result is:

;define dso_local void @_Z3foov() {
;  %1 = alloca float, align 4
;  store float 1.000000e+00, ptr %1, align 4
;  %2 = load float, ptr %1, align 4
;  %3 = fadd float %2, 1.000000e+00
;  store float %3, ptr %1, align 4
;  ret void
;}
;
;define dso_local void @_Z3barv() {
;.split:
;  %0 = alloca i32, align 4
;  store i32 0, ptr %0, align 4
;  br label %1
;
;1:                                                ; preds = %.split
;  %2 = alloca float, align 4
;  store float 1.000000e+00, ptr %2, align 4
;  %3 = load float, ptr %2, align 4
;  %4 = fadd float %3, 1.000000e+00
;  store float %4, ptr %2, align 4
;  br label %5
;
;5:                                                ; preds = %1
;  %6 = load i32, ptr %0, align 4
;  %7 = add nsw i32 %6, 1
;  store i32 %7, ptr %0, align 4
;  ret void
;}

; COM: Begin checking

; CHECK: define dso_local void @_Z3barv() {
; CHECK: store i32 0, ptr %0, align 4
; CHECK-NEXT: br label %1
; CHECK: %2 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %2, align 4
; CHECK-NEXT: %3 = load float, ptr %2, align 4
; CHECK-NEXT: %4 = fadd float %3, 1.000000e+00
; CHECK-NEXT: store float %4, ptr %2, align 4
; CHECK-NEXT: br label %5
; COM: After inline function...
; CHECK: %6 = load i32, ptr %0, align 4

define dso_local void @_Z3foov() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 1.000000e+00
  store float %3, float* %1, align 4
  ret void
}

define dso_local void @_Z3barv() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @_Z3foov()
  %2 = load i32, i32* %1, align 4
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4
  ret void
}

;--- test2.ll
; COM: Expect not inline because of the argument

;void foo(int) {
;  float a = 1.0f;
;  a += 1.0f;
;}
;
;void bar() {
;  int a = 0;
;  foo(a);
;  a++;
;}

; COM: Exactly the same output as testing one

; COM: Begin checking

; CHECK: define dso_local void @_Z3barv() {
; CHECK: %2 = load i32, ptr %1, align 4
; CHECK-NEXT: call void @_Z3fooi(i32 noundef %2)
; CHECK-NOT: %{{[\d]+}} = alloca float, align 4

define dso_local void @_Z3fooi(i32 noundef %0) {
  %2 = alloca i32, align 4
  %3 = alloca float, align 4
  store i32 %0, i32* %2, align 4
  store float 1.000000e+00, float* %3, align 4
  %4 = load float, float* %3, align 4
  %5 = fadd float %4, 1.000000e+00
  store float %5, float* %3, align 4
  ret void
}

define dso_local void @_Z3barv() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = load i32, i32* %1, align 4
  call void @_Z3fooi(i32 noundef %2)
  %3 = load i32, i32* %1, align 4
  %4 = add nsw i32 %3, 1
  store i32 %4, i32* %1, align 4
  ret void
}

;--- test3.ll
; COM: Expect not inline because of the non-void return

;float foo() {
;  float a = 1.0f;
;  a += 1.0f;
;  return a;
;}
;
;void bar() {
;  int a = 0;
;  foo();
;  a++;
;}

; COM: Exactly the same output as testing one

; COM: Begin checking

; CHECK: define dso_local void @_Z3barv() {
; CHECK: store i32 0, ptr %1, align 4
; CHECK-NEXT: %2 = call noundef float @_Z3foov()
; CHECK-NOT: %{{[\d]+}} = alloca float, align 4

define dso_local noundef float @_Z3foov() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 1.000000e+00
  store float %3, float* %1, align 4
  %4 = load float, float* %1, align 4
  ret float %4
}

define dso_local void @_Z3barv() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  %2 = call noundef float @_Z3foov()
  %3 = load i32, i32* %1, align 4
  %4 = add nsw i32 %3, 1
  store i32 %4, i32* %1, align 4
  ret void
}

;--- test4.ll
; COM: Expect foo to be inlined, but still with foo() call inside because of the recursion

;void foo() {
;  float a = 1.0f;
;  a += 1.0f;
;  foo();
;}
;
;void bar() {
;  int a = 0;
;  foo();
;  a++;
;}

; COM: The result is:

;define dso_local void @_Z3foov() {
;  %1 = alloca float, align 4
;  store float 1.000000e+00, ptr %1, align 4
;  %2 = load float, ptr %1, align 4
;  %3 = fadd float %2, 1.000000e+00
;  store float %3, ptr %1, align 4
;  call void @_Z3foov()
;  ret void
;}
;
;define dso_local void @_Z3barv() {
;.split:
;  %0 = alloca i32, align 4
;  store i32 0, ptr %0, align 4
;  br label %1
;
;1:                                                ; preds = %.split
;  %2 = alloca float, align 4
;  store float 1.000000e+00, ptr %2, align 4
;  %3 = load float, ptr %2, align 4
;  %4 = fadd float %3, 1.000000e+00
;  store float %4, ptr %2, align 4
;  call void @_Z3foov()
;  br label %5
;
;5:                                                ; preds = %1
;  %6 = load i32, ptr %0, align 4
;  %7 = add nsw i32 %6, 1
;  store i32 %7, ptr %0, align 4
;  ret void
;}

; COM: Begin checking

; CHECK: define dso_local void @_Z3barv() {
; CHECK: store i32 0, ptr %0, align 4
; CHECK-NEXT: br label %1
; CHECK: %2 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %2, align 4
; CHECK-NEXT: %3 = load float, ptr %2, align 4
; CHECK-NEXT: %4 = fadd float %3, 1.000000e+00
; CHECK-NEXT: store float %4, ptr %2, align 4
; CHECK-NEXT: call void @_Z3foov()
; CHECK-NEXT: br label %5
; COM: After inline function...
; CHECK: %6 = load i32, ptr %0, align 4

define dso_local void @_Z3foov() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 1.000000e+00
  store float %3, float* %1, align 4
  call void @_Z3foov()
  ret void
}

define dso_local void @_Z3barv() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @_Z3foov()
  %2 = load i32, i32* %1, align 4
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4
  ret void
}

;--- test5.ll
; COM: Magic inline check with loops.

;void foo() {
;  float a = 1.0f;
;  a += 1.0f;
;  while (a >= 0.0f) {
;    a -= 1.0f;
;  }
;}
;
;void bar() {
;  int a = 0;
;  foo();
;  a++;
;}

; COM: The result is:

;define dso_local void @_Z3foov() {
;  %1 = alloca float, align 4
;  store float 1.000000e+00, ptr %1, align 4
;  %2 = load float, ptr %1, align 4
;  %3 = fadd float %2, 1.000000e+00
;  store float %3, ptr %1, align 4
;  br label %4
;
;4:                                                ; preds = %7, %0
;  %5 = load float, ptr %1, align 4
;  %6 = fcmp oge float %5, 0.000000e+00
;  br i1 %6, label %7, label %10
;
;7:                                                ; preds = %4
;  %8 = load float, ptr %1, align 4
;  %9 = fsub float %8, 1.000000e+00
;  store float %9, ptr %1, align 4
;  br label %4
;
;10:                                               ; preds = %4
;  ret void
;}
;
;define dso_local void @_Z3barv() {
;.split:
;  %0 = alloca i32, align 4
;  store i32 0, ptr %0, align 4
;  br label %1
;
;1:                                                ; preds = %.split
;  %2 = alloca float, align 4
;  store float 1.000000e+00, ptr %2, align 4
;  %3 = load float, ptr %2, align 4
;  %4 = fadd float %3, 1.000000e+00
;  store float %4, ptr %2, align 4
;  br label %5
;
;5:                                                ; preds = %8, %1
;  %6 = load float, ptr %2, align 4
;  %7 = fcmp oge float %6, 0.000000e+00
;  br i1 %7, label %8, label %11
;
;8:                                                ; preds = %5
;  %9 = load float, ptr %2, align 4
;  %10 = fsub float %9, 1.000000e+00
;  store float %10, ptr %2, align 4
;  br label %5
;
;11:                                               ; preds = %5
;  br label %12
;
;12:                                               ; preds = %11
;  %13 = load i32, ptr %0, align 4
;  %14 = add nsw i32 %13, 1
;  store i32 %14, ptr %0, align 4
;  ret void
;}

; COM: Begin checking

; CHECK: define dso_local void @_Z3barv() {
; CHECK: store i32 0, ptr %0, align 4
; CHECK-NEXT: br label %1
; CHECK-EMPTY:
; CHECK-NEXT: 1:
; CHECK-NEXT: %2 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %2, align 4
; CHECK-NEXT: %3 = load float, ptr %2, align 4
; CHECK-NEXT: %4 = fadd float %3, 1.000000e+00
; CHECK-NEXT: store float %4, ptr %2, align 4
; CHECK-NEXT: br label %5
; CHECK-EMPTY:
; CHECK-NEXT: 5:
; CHECK-NEXT: %6 = load float, ptr %2, align 4
; CHECK-NEXT: %7 = fcmp oge float %6, 0.000000e+00
; CHECK-NEXT: br i1 %7, label %8, label %11
; CHECK-EMPTY:
; CHECK-NEXT: 8:
; CHECK-NEXT: %9 = load float, ptr %2, align 4
; CHECK-NEXT: %10 = fsub float %9, 1.000000e+00
; CHECK-NEXT: store float %10, ptr %2, align 4
; CHECK-NEXT: br label %5
; CHECK-EMPTY:
; CHECK-NEXT: 11:
; CHECK-NEXT: br label %12
; CHECK-EMPTY:
; CHECK-NEXT: 12:
; CHECK-NEXT: %13 = load i32, ptr %0, align 4

define dso_local void @_Z3foov() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 1.000000e+00
  store float %3, float* %1, align 4
  br label %4

4:
  %5 = load float, float* %1, align 4
  %6 = fcmp oge float %5, 0.000000e+00
  br i1 %6, label %7, label %10

7:
  %8 = load float, float* %1, align 4
  %9 = fsub float %8, 1.000000e+00
  store float %9, float* %1, align 4
  br label %4

10:
  ret void
}

define dso_local void @_Z3barv() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @_Z3foov()
  %2 = load i32, i32* %1, align 4
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4
  ret void
}

;--- test6.ll
; COM: 2 loops

;void foo() {
;  float a = 1.0f;
;  a += 1.0f;
;  while (a >= 0.0f) {
;    a -= 1.0f;
;  }
;}
;
;void bar() {
;  int a = 1;
;  while (a) {
;    foo();
;    a--;
;  }
;  a++;
;}

; COM: The result is

;define dso_local void @_Z3foov() {
;  %1 = alloca float, align 4
;  store float 1.000000e+00, ptr %1, align 4
;  %2 = load float, ptr %1, align 4
;  %3 = fadd float %2, 1.000000e+00
;  store float %3, ptr %1, align 4
;  br label %4
;
;4:                                                ; preds = %7, %0
;  %5 = load float, ptr %1, align 4
;  %6 = fcmp oge float %5, 0.000000e+00
;  br i1 %6, label %7, label %10
;
;7:                                                ; preds = %4
;  %8 = load float, ptr %1, align 4
;  %9 = fsub float %8, 1.000000e+00
;  store float %9, ptr %1, align 4
;  br label %4
;
;10:                                               ; preds = %4
;  ret void
;}
;
;define dso_local void @_Z3barv() {
;  %1 = alloca i32, align 4
;  store i32 1, ptr %1, align 4
;  br label %2
;
;2:                                                ; preds = %16, %0
;  %3 = load i32, ptr %1, align 4
;  %4 = icmp ne i32 %3, 0
;  br i1 %4, label %.split, label %19
;
;.split:                                           ; preds = %2
;  br label %5
;
;5:                                                ; preds = %.split
;  %6 = alloca float, align 4
;  store float 1.000000e+00, ptr %6, align 4
;  %7 = load float, ptr %6, align 4
;  %8 = fadd float %7, 1.000000e+00
;  store float %8, ptr %6, align 4
;  br label %9
;
;9:                                                ; preds = %12, %5
;  %10 = load float, ptr %6, align 4
;  %11 = fcmp oge float %10, 0.000000e+00
;  br i1 %11, label %12, label %15
;
;12:                                               ; preds = %9
;  %13 = load float, ptr %6, align 4
;  %14 = fsub float %13, 1.000000e+00
;  store float %14, ptr %6, align 4
;  br label %9
;
;15:                                               ; preds = %9
;  br label %16
;
;16:                                               ; preds = %15
;  %17 = load i32, ptr %1, align 4
;  %18 = add nsw i32 %17, -1
;  store i32 %18, ptr %1, align 4
;  br label %2
;
;19:                                               ; preds = %2
;  %20 = load i32, ptr %1, align 4
;  %21 = add nsw i32 %20, 1
;  store i32 %21, ptr %1, align 4
;  ret void
;}

; COM: Begin checking

; CHECK: define dso_local void @_Z3barv() {
; CHECK: %4 = icmp ne i32 %3, 0
; CHECK-NEXT: br i1 %4, label %.split, label %19
; CHECK-EMPTY:
; CHECK-NEXT: .split
; CHECK-NEXT: br label %5
; CHECK-EMPTY:
; CHECK-NEXT: 5:
; CHECK-NEXT: %6 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %6, align 4
; CHECK-NEXT: %7 = load float, ptr %6, align 4
; CHECK-NEXT: %8 = fadd float %7, 1.000000e+00
; CHECK-NEXT: store float %8, ptr %6, align 4
; CHECK-NEXT: br label %9
; CHECK-EMPTY:
; CHECK-NEXT: 9:
; CHECK-NEXT: %10 = load float, ptr %6, align 4
; CHECK-NEXT: %11 = fcmp oge float %10, 0.000000e+00
; CHECK-NEXT: br i1 %11, label %12, label %15
; CHECK-EMPTY:
; CHECK-NEXT: 12:
; CHECK-NEXT: %13 = load float, ptr %6, align 4
; CHECK-NEXT: %14 = fsub float %13, 1.000000e+00
; CHECK-NEXT: store float %14, ptr %6, align 4
; CHECK-NEXT: br label %9
; CHECK-EMPTY:
; CHECK-NEXT: 15:
; CHECK-NEXT: br label %16
; CHECK-EMPTY:
; CHECK-NEXT: 16:
; COM: The rest of it is not very interesting. It's just renaming.

define dso_local void @_Z3foov() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 1.000000e+00
  store float %3, float* %1, align 4
  br label %4

4:
  %5 = load float, float* %1, align 4
  %6 = fcmp oge float %5, 0.000000e+00
  br i1 %6, label %7, label %10

7:
  %8 = load float, float* %1, align 4
  %9 = fsub float %8, 1.000000e+00
  store float %9, float* %1, align 4
  br label %4

10:
  ret void
}

define dso_local void @_Z3barv() {
  %1 = alloca i32, align 4
  store i32 1, i32* %1, align 4
  br label %2

2:
  %3 = load i32, i32* %1, align 4
  %4 = icmp ne i32 %3, 0
  br i1 %4, label %5, label %8

5:
  call void @_Z3foov()
  %6 = load i32, i32* %1, align 4
  %7 = add nsw i32 %6, -1
  store i32 %7, i32* %1, align 4
  br label %2

8:
  %9 = load i32, i32* %1, align 4
  %10 = add nsw i32 %9, 1
  store i32 %10, i32* %1, align 4
  ret void
}

;--- test7.ll
; COM: Inline composition with loops

;void foo1() {
;  float a = 1.0f;
;  a += 1.0f;
;  while (a >= 0.0f) {
;    a -= 1.0f;
;  }
;}
;
;void foo2() {
;  float a = 4.0f;
;  a += 2.0f;
;  while (a >= 0.0f) {
;    foo1();
;    a -= 2.0f;
;  }
;}
;
;void bar() {
;  int a = 0;
;  foo2();
;  a++;
;}

; COM: The result is

;define dso_local void @_Z4foo1v() {
;  %1 = alloca float, align 4
;  store float 1.000000e+00, ptr %1, align 4
;  %2 = load float, ptr %1, align 4
;  %3 = fadd float %2, 1.000000e+00
;  store float %3, ptr %1, align 4
;  br label %4
;
;4:                                                ; preds = %7, %0
;  %5 = load float, ptr %1, align 4
;  %6 = fcmp oge float %5, 0.000000e+00
;  br i1 %6, label %7, label %10
;
;7:                                                ; preds = %4
;  %8 = load float, ptr %1, align 4
;  %9 = fsub float %8, 1.000000e+00
;  store float %9, ptr %1, align 4
;  br label %4
;
;10:                                               ; preds = %4
;  ret void
;}
;
;define dso_local void @_Z4foo2v() {
;  %1 = alloca float, align 4
;  store float 4.000000e+00, ptr %1, align 4
;  %2 = load float, ptr %1, align 4
;  %3 = fadd float %2, 2.000000e+00
;  store float %3, ptr %1, align 4
;  br label %4
;
;4:                                                ; preds = %18, %0
;  %5 = load float, ptr %1, align 4
;  %6 = fcmp oge float %5, 0.000000e+00
;  br i1 %6, label %.split, label %21
;
;.split:                                           ; preds = %4
;  br label %7
;
;7:                                                ; preds = %.split
;  %8 = alloca float, align 4
;  store float 1.000000e+00, ptr %8, align 4
;  %9 = load float, ptr %8, align 4
;  %10 = fadd float %9, 1.000000e+00
;  store float %10, ptr %8, align 4
;  br label %11
;
;11:                                               ; preds = %14, %7
;  %12 = load float, ptr %8, align 4
;  %13 = fcmp oge float %12, 0.000000e+00
;  br i1 %13, label %14, label %17
;
;14:                                               ; preds = %11
;  %15 = load float, ptr %8, align 4
;  %16 = fsub float %15, 1.000000e+00
;  store float %16, ptr %8, align 4
;  br label %11
;
;17:                                               ; preds = %11
;  br label %18
;
;18:                                               ; preds = %17
;  %19 = load float, ptr %1, align 4
;  %20 = fsub float %19, 2.000000e+00
;  store float %20, ptr %1, align 4
;  br label %4
;
;21:                                               ; preds = %4
;  ret void
;}
;
;define dso_local void @_Z3barv() {
;.split:
;  %0 = alloca i32, align 4
;  store i32 0, ptr %0, align 4
;  br label %1
;
;1:                                                ; preds = %.split
;  %2 = alloca float, align 4
;  store float 4.000000e+00, ptr %2, align 4
;  %3 = load float, ptr %2, align 4
;  %4 = fadd float %3, 2.000000e+00
;  store float %4, ptr %2, align 4
;  br label %5
;
;5:                                                ; preds = %20, %1
;  %6 = load float, ptr %2, align 4
;  %7 = fcmp oge float %6, 0.000000e+00
;  br i1 %7, label %8, label %23
;
;8:                                                ; preds = %5
;  br label %9
;
;9:                                                ; preds = %8
;  %10 = alloca float, align 4
;  store float 1.000000e+00, ptr %10, align 4
;  %11 = load float, ptr %10, align 4
;  %12 = fadd float %11, 1.000000e+00
;  store float %12, ptr %10, align 4
;  br label %13
;
;13:                                               ; preds = %16, %9
;  %14 = load float, ptr %10, align 4
;  %15 = fcmp oge float %14, 0.000000e+00
;  br i1 %15, label %16, label %19
;
;16:                                               ; preds = %13
;  %17 = load float, ptr %10, align 4
;  %18 = fsub float %17, 1.000000e+00
;  store float %18, ptr %10, align 4
;  br label %13
;
;19:                                               ; preds = %13
;  br label %20
;
;20:                                               ; preds = %19
;  %21 = load float, ptr %2, align 4
;  %22 = fsub float %21, 2.000000e+00
;  store float %22, ptr %2, align 4
;  br label %5
;
;23:                                               ; preds = %5
;  br label %24
;
;24:                                               ; preds = %23
;  %25 = load i32, ptr %0, align 4
;  %26 = add nsw i32 %25, 1
;  store i32 %26, ptr %0, align 4
;  ret void
;}

; COM: Begin checking

; COM: First check foo2
; CHECK: define dso_local void @_Z4foo2v() {
; CHECK: %6 = fcmp oge float %5, 0.000000e+00
; CHECK-NEXT: br i1 %6, label %.split, label %21
; CHECK-EMPTY:
; CHECK-NEXT: .split:
; CHECK-NEXT: br label %7
; CHECK-EMPTY:
; CHECK-NEXT: 7:
; CHECK-NEXT: %8 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %8, align 4
; COM: Basically copied. Skipping...
; CHECK: %16 = fsub float %15, 1.000000e+00
; CHECK-NEXT: store float %16, ptr %8, align 4
; CHECK-NEXT: br label %11
; CHECK-EMPTY:
; CHECK-NEXT: 17:
; CHECK-NEXT: br label %18
; CHECK-EMPTY:
; CHECK-NEXT: 18:
; CHECK-NEXT: %19 = load float, ptr %1, align 4
; CHECK-NEXT: %20 = fsub float %19, 2.000000e+00
; CHECK-NEXT: store float %20, ptr %1, align 4
; CHECK-NEXT: br label %4
; CHECK-EMPTY:
; CHECK-NEXT: 21:
; CHECK-NEXT: ret void
; COM: now bar
; CHECK: define dso_local void @_Z3barv() {
; COM: see foo2()
; CHECK: store i32 0, ptr %0, align 4
; CHECK-NEXT: br label %1
; CHECK-EMPTY:
; CHECK-NEXT: 1:
; CHECK-NEXT: %2 = alloca float, align 4
; CHECK-NEXT: store float 4.000000e+00, ptr %2, align 4
; COM: let's skip right into inline foo1()
; CHECK: %7 = fcmp oge float %6, 0.000000e+00
; CHECK-NEXT: br i1 %7, label %8, label %23
; CHECK-EMPTY:
; CHECK-NEXT: 8:
; CHECK-NEXT: br label %9
; CHECK-EMPTY:
; CHECK-NEXT: 9:
; CHECK-NEXT: %10 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %10, align 4
; CHECK-NEXT: %11 = load float, ptr %10, align 4
; CHECK-NEXT: %12 = fadd float %11, 1.000000e+00
; COM: continue bar()
; CHECK: 23:
; CHECK-NEXT: br label %24
; CHECK-EMPTY:
; CHECK-NEXT: 24:
; CHECK-NEXT: %25 = load i32, ptr %0, align 4
; CHECK-NEXT: %26 = add nsw i32 %25, 1
; CHECK-NEXT: store i32 %26, ptr %0, align 4
; CHECK-NEXT: ret void

define dso_local void @_Z4foo1v() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 1.000000e+00
  store float %3, float* %1, align 4
  br label %4

4:
  %5 = load float, float* %1, align 4
  %6 = fcmp oge float %5, 0.000000e+00
  br i1 %6, label %7, label %10

7:
  %8 = load float, float* %1, align 4
  %9 = fsub float %8, 1.000000e+00
  store float %9, float* %1, align 4
  br label %4

10:
  ret void
}

define dso_local void @_Z4foo2v() {
  %1 = alloca float, align 4
  store float 4.000000e+00, float* %1, align 4
  %2 = load float, float* %1, align 4
  %3 = fadd float %2, 2.000000e+00
  store float %3, float* %1, align 4
  br label %4

4:
  %5 = load float, float* %1, align 4
  %6 = fcmp oge float %5, 0.000000e+00
  br i1 %6, label %7, label %10

7:
  call void @_Z4foo1v()
  %8 = load float, float* %1, align 4
  %9 = fsub float %8, 2.000000e+00
  store float %9, float* %1, align 4
  br label %4

10:
  ret void
}

define dso_local void @_Z3barv() {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @_Z4foo2v()
  %2 = load i32, i32* %1, align 4
  %3 = add nsw i32 %2, 1
  store i32 %3, i32* %1, align 4
  ret void
}

;--- test8.ll
; COM: Circular call.

;void foo();
;void bar();
;
;void foo() {
;  float a = 1.0f;
;  bar();
;}
;
;void bar() {
;  float a = 2.0f;
;  foo();
;}
;
;void checker() {
;  int c = 4;
;  foo();
;  bar();
;}

; COM: The result is

;define dso_local void @_Z3foov() {
;.split:
;  %0 = alloca float, align 4
;  store float 1.000000e+00, ptr %0, align 4
;  br label %1
;
;1:                                                ; preds = %.split
;  %2 = alloca float, align 4
;  store float 2.000000e+00, ptr %2, align 4
;  call void @_Z3foov()
;  br label %3
;
;3:                                                ; preds = %1
;  ret void
;}
;
;define dso_local void @_Z3barv() {
;.split:
;  %0 = alloca float, align 4
;  store float 2.000000e+00, ptr %0, align 4
;  br label %1
;
;1:                                                ; preds = %.split
;  %2 = alloca float, align 4
;  store float 1.000000e+00, ptr %2, align 4
;  br label %3
;
;3:                                                ; preds = %1
;  %4 = alloca float, align 4
;  store float 2.000000e+00, ptr %4, align 4
;  call void @_Z3foov()
;  br label %5
;
;5:                                                ; preds = %3
;  br label %6
;
;6:                                                ; preds = %5
;  ret void
;}
;
;define dso_local void @_Z7checkerv() {
;.split:
;  %0 = alloca i32, align 4
;  store i32 4, ptr %0, align 4
;  br label %1
;
;1:                                                ; preds = %.split
;  %2 = alloca float, align 4
;  store float 1.000000e+00, ptr %2, align 4
;  br label %3
;
;3:                                                ; preds = %1
;  %4 = alloca float, align 4
;  store float 2.000000e+00, ptr %4, align 4
;  call void @_Z3foov()
;  br label %5
;
;5:                                                ; preds = %3
;  br label %.split1
;
;.split1:                                          ; preds = %5
;  br label %6
;
;6:                                                ; preds = %.split1
;  %7 = alloca float, align 4
;  store float 2.000000e+00, ptr %7, align 4
;  br label %8
;
;8:                                                ; preds = %6
;  %9 = alloca float, align 4
;  store float 1.000000e+00, ptr %9, align 4
;  br label %10
;
;10:                                               ; preds = %8
;  %11 = alloca float, align 4
;  store float 2.000000e+00, ptr %11, align 4
;  call void @_Z3foov()
;  br label %12
;
;12:                                               ; preds = %10
;  br label %13
;
;13:                                               ; preds = %12
;  br label %14
;
;14:                                               ; preds = %13
;  ret void
;}

; COM: Begin checking

; COM: foo() has inline bar() that doesn't has inline foo()
; CHECK: define dso_local void @_Z3foov() {
; CHECK-NEXT: .split:
; CHECK-NEXT: %0 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %0, align 4
; CHECK-NEXT: br label %1
; CHECK-EMPTY:
; CHECK-NEXT: 1:
; CHECK-NEXT: %2 = alloca float, align 4
; CHECK-NEXT: store float 2.000000e+00, ptr %2, align 4
; CHECK-NEXT: call void @_Z3foov()
; CHECK-NEXT: br label %3
; COM: bar() has inline foo() that has inline bar that doesn't inline foo()
; CHECK: define dso_local void @_Z3barv() {
; CHECK-NEXT: .split:
; CHECK-NEXT: %0 = alloca float, align 4
; CHECK-NEXT: store float 2.000000e+00, ptr %0, align 4
; CHECK-NEXT: br label %1
; CHECK-EMPTY:
; CHECK-NEXT: 1:
; CHECK-NEXT: %2 = alloca float, align 4
; CHECK-NEXT: store float 1.000000e+00, ptr %2, align
; CHECK-NEXT: br label %3
; CHECK-EMPTY:
; CHECK-NEXT: 3:
; CHECK-NEXT: %4 = alloca float, align 4
; CHECK-NEXT: store float 2.000000e+00, ptr %4, align 4
; CHECK-NEXT: call void @_Z3foov()
; CHECK-NEXT: br label %5
; COM: the check() function will obviously work as expected. Skipping...

define dso_local void @_Z3foov() {
  %1 = alloca float, align 4
  store float 1.000000e+00, float* %1, align 4
  call void @_Z3barv()
  ret void
}

define dso_local void @_Z3barv() {
  %1 = alloca float, align 4
  store float 2.000000e+00, float* %1, align 4
  call void @_Z3foov()
  ret void
}

define dso_local void @_Z7checkerv() {
  %1 = alloca i32, align 4
  store i32 4, i32* %1, align 4
  call void @_Z3foov()
  call void @_Z3barv()
  ret void
}
