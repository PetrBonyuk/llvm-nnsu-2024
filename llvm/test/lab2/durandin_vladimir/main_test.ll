; RUN: opt -load-pass-plugin %llvmshlibdir/durandin_vladimir_inline_pass_plugin%pluginext\
; RUN: -passes=custom-inlining -S %s | FileCheck %s


; COM: Simple inline check. Expect inline

;void func()
;{
;    int a = 0;
;    a += 1;
;}
;
;int foo(int num)
;{
;    func();
;    return num;
;}


define dso_local void @_Z4funcv() #0 {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %0 = load i32, ptr %a, align 4
  %add = add nsw i32 %0, 1
  store i32 %add, ptr %a, align 4
  ret void
}

define dso_local noundef i32 @_Z3fooi(i32 noundef %num) #0 {
entry:
  %num.addr = alloca i32, align 4
  store i32 %num, ptr %num.addr, align 4
  call void @_Z4funcv()
  %0 = load i32, ptr %num.addr, align 4
  ret i32 %0
}

; CHECK: define dso_local void @_Z4funcv() {
; CHECK: entry:
; CHECK-NEXT:   %a = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %a, align 4
; CHECK-NEXT:   %0 = load i32, ptr %a, align 4
; CHECK-NEXT:   %add = add nsw i32 %0, 1
; CHECK-NEXT:   store i32 %add, ptr %a, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; CHECK: define dso_local noundef i32 @_Z3fooi(i32 noundef %num) {
; CHECK: entry:
; CHECK-NEXT:   %num.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 %num, ptr %num.addr, align 4
; CHECK-NEXT:   br label %entry.inlined.0
; CHECK-EMPTY: 
; CHECK: entry.inlined.0:                                  ; preds = %entry
; CHECK-NEXT:   %0 = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %0, align 4
; CHECK-NEXT:   %1 = load i32, ptr %0, align 4
; CHECK-NEXT:   %2 = add nsw i32 %1, 1
; CHECK-NEXT:   store i32 %2, ptr %0, align 4
; CHECK-NEXT:   br label %entry.splited.0
; CHECK-EMPTY: 
; CHECK: entry.splited.0:                                  ; preds = %entry.inlined.0
; CHECK-NEXT:   %3 = load i32, ptr %num.addr, align 4
; CHECK-NEXT:   ret i32 %3
; CHECK-NEXT: }

; --------------------------------------------------------------------

; COM: Expect not inline because of the non-void return

; float foo() { 
;   float a = 1.0f; 
;   a += 1.0f; 
;   return  a; 
; } 
;  
; void bar() { 
;   int a = 0; 
;   foo(); 
;   a++; 
; }

define dso_local noundef float @_Z3foov() #0 {
entry:
  %a = alloca float, align 4
  store float 1.000000e+00, ptr %a, align 4
  %0 = load float, ptr %a, align 4
  %add = fadd float %0, 1.000000e+00
  store float %add, ptr %a, align 4
  %1 = load float, ptr %a, align 4
  ret float %1
}

define dso_local void @_Z3barv() #0 {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  %call = call noundef float @_Z3foov()
  %0 = load i32, ptr %a, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr %a, align 4
  ret void
}

; CHECK: define dso_local noundef float @_Z3foov() {
; CHECK: entry:
; CHECK-NEXT:   %a = alloca float, align 4
; CHECK-NEXT:   store float 1.000000e+00, ptr %a, align 4
; CHECK-NEXT:   %0 = load float, ptr %a, align 4
; CHECK-NEXT:   %add = fadd float %0, 1.000000e+00
; CHECK-NEXT:   store float %add, ptr %a, align 4
; CHECK-NEXT:   %1 = load float, ptr %a, align 4
; CHECK-NEXT:   ret float %1
; CHECK-NEXT: }
; CHECK: define dso_local void @_Z3barv() {
; CHECK: entry:
; CHECK-NEXT:   %a = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %a, align 4
; CHECK-NEXT:   %call = call noundef float @_Z3foov()
; CHECK-NEXT:   %0 = load i32, ptr %a, align 4
; CHECK-NEXT:   %inc = add nsw i32 %0, 1
; CHECK-NEXT:   store i32 %inc, ptr %a, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


; --------------------------------------------------------------------

; COM: Simple while loop

; void foo_loop() {
;   float a = 1.0f;
;   a += 1.0f;
;   while (a >= 0.0f) {
;     a -= 1.0f;
;   }
; }
; 
; void bar_loop_inline() {
;   int a = 0;
;   foo_loop();
;   a++;
; }


define dso_local void @_Z8foo_loopv() #0 {
entry:
  %a = alloca float, align 4
  store float 1.000000e+00, ptr %a, align 4
  %0 = load float, ptr %a, align 4
  %add = fadd float %0, 1.000000e+00
  store float %add, ptr %a, align 4
  br label %while.cond

while.cond:                                       ; preds = %while.body, %entry
  %1 = load float, ptr %a, align 4
  %cmp = fcmp oge float %1, 0.000000e+00
  br i1 %cmp, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %2 = load float, ptr %a, align 4
  %sub = fsub float %2, 1.000000e+00
  store float %sub, ptr %a, align 4
  br label %while.cond

while.end:                                        ; preds = %while.cond
  ret void
}

define dso_local void @_Z15bar_loop_inlinev() #0 {
entry:
  %a = alloca i32, align 4
  store i32 0, ptr %a, align 4
  call void @_Z8foo_loopv()
  %0 = load i32, ptr %a, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr %a, align 4
  ret void
}

; CHECK: define dso_local void @_Z8foo_loopv() {
; CHECK: entry:
; CHECK-NEXT:   %a = alloca float, align 4
; CHECK-NEXT:   store float 1.000000e+00, ptr %a, align 4
; CHECK-NEXT:   %0 = load float, ptr %a, align 4
; CHECK-NEXT:   %add = fadd float %0, 1.000000e+00
; CHECK-NEXT:   store float %add, ptr %a, align 4
; CHECK-NEXT:   br label %while.cond
; CHECK-EMPTY: 
; CHECK: while.cond:                                       ; preds = %while.body, %entry
; CHECK-NEXT:   %1 = load float, ptr %a, align 4
; CHECK-NEXT:   %cmp = fcmp oge float %1, 0.000000e+00
; CHECK-NEXT:   br i1 %cmp, label %while.body, label %while.end
; CHECK-EMPTY: 
; CHECK: while.body:                                       ; preds = %while.cond
; CHECK-NEXT:   %2 = load float, ptr %a, align 4
; CHECK-NEXT:   %sub = fsub float %2, 1.000000e+00
; CHECK-NEXT:   store float %sub, ptr %a, align 4
; CHECK-NEXT:   br label %while.cond
; CHECK-EMPTY: 
; CHECK: while.end:                                        ; preds = %while.cond
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


; CHECK: define dso_local void @_Z15bar_loop_inlinev() {
; CHECK: entry:
; CHECK-NEXT:   %a = alloca i32, align 4
; CHECK-NEXT:   store i32 0, ptr %a, align 4
; CHECK-NEXT:   br label %entry.inlined.0
; CHECK-EMPTY: 
; CHECK: entry.inlined.0:                                  ; preds = %entry
; CHECK-NEXT:   %0 = alloca float, align 4
; CHECK-NEXT:   store float 1.000000e+00, ptr %0, align 4
; CHECK-NEXT:   %1 = load float, ptr %0, align 4
; CHECK-NEXT:   %2 = fadd float %1, 1.000000e+00
; CHECK-NEXT:   store float %2, ptr %0, align 4
; CHECK-NEXT:   br label %while.cond.inlined.0
; CHECK-EMPTY: 
; CHECK: while.cond.inlined.0:                             ; preds = %while.body.inlined.0, %entry.inlined.0
; CHECK-NEXT:   %3 = load float, ptr %0, align 4
; CHECK-NEXT:   %4 = fcmp oge float %3, 0.000000e+00
; CHECK-NEXT:   br i1 %4, label %while.body.inlined.0, label %while.end.inlined.0
; CHECK-EMPTY: 
; CHECK: while.body.inlined.0:                             ; preds = %while.cond.inlined.0
; CHECK-NEXT:   %5 = load float, ptr %0, align 4
; CHECK-NEXT:   %6 = fsub float %5, 1.000000e+00
; CHECK-NEXT:   store float %6, ptr %0, align 4
; CHECK-NEXT:   br label %while.cond.inlined.0
; CHECK-EMPTY: 
; CHECK: while.end.inlined.0:                              ; preds = %while.cond.inlined.0
; CHECK-NEXT:   br label %entry.splited.0
; CHECK-EMPTY: 
; CHECK: entry.splited.0:                                  ; preds = %while.end.inlined.0
; CHECK-NEXT:   %7 = load i32, ptr %a, align 4
; CHECK-NEXT:   %inc = add nsw i32 %7, 1
; CHECK-NEXT:   store i32 %inc, ptr %a, align 4
; CHECK-NEXT:   ret void
; CHECK-NEXT: }