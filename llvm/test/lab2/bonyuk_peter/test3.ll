; RUN: opt -load-pass-plugin=%llvmshlibdir/BonyukLoopPlugin%shlibext -passes=bonyuk-loop-plugin -S %s | FileCheck %s

;void whileFoo() {
;    int i = 0;
;    while (i < 5) {
;        i++;
;    }
;}

define dso_local void @whileFoo() {
entry:
%i = alloca i32, align 4
store i32 0, ptr %i, align 4

; CHECK: call void @loop_start()
; CHECK-NEXT: br label %while.cond

br label %while.cond

while.cond:
%0 = load i32, ptr %i, align 4
%cmp = icmp slt i32 %0, 5
br i1 %cmp, label %while.body, label %while.end

while.body:
%1 = load i32, ptr %i, align 4
%inc = add nsw i32 %1, 1
store i32 %inc, ptr %i, align 4
br label %while.cond

; CHECK: while.end:
; CHECK-NEXT: call void @loop_end()

while.end:
ret void
}