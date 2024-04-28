; RUN: opt -load-pass-plugin=%llvmshlibdir/BonyukLoopPlugin%shlibext -passes=bonyuk-loop-plugin -S %s | FileCheck %s

; void WhileFunc() {
;     int i = 0;
;     while (i < 10) {
;         i++;
;         if (i == 5)
;              break;
;     }
; }
define dso_local void @WhileFunc() {
entry:
  %i = alloca i32
  store i32 0, i32* %i
  br label %loop.cond
  ; CHECK: call void @loop_start

loop.cond:
  %0 = load i32, i32* %i
  %cmp = icmp slt i32 %0, 10
  br i1 %cmp, label %loop.body, label %loop.end

loop.body:
  %1 = load i32, i32* %i
  %inc = add nsw i32 %1, 1
  store i32 %inc, i32* %i
  %2 = load i32, i32* %i
  %cmp1 = icmp eq i32 %2, 5
  br i1 %cmp1, label %loop.break, label %loop.cond

loop.break:
  br label %loop.end

loop.end:
  ret void
}