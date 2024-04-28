; RUN: opt -load-pass-plugin=%llvmshlibdir/BonyukLoopPlugin%shlibext -passes=bonyuk-loop-plugin -S %s | FileCheck %s
; void ForFunc() {
;     for(int i = 5; i > 0; i--){
;         i+=2;
;         if(i == 10)
;             break;
;     }
; }

define dso_local void @ForFunc() {
entry:
  %i = alloca i32
  store i32 5, i32* %i
  br label %for.cond

for.cond:
  %0 = load i32, i32* %i
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %1 = load i32, i32* %i
  %inc = add nsw i32 %1, 2
  store i32 %inc, i32* %i

  %2 = load i32, i32* %i
  %cmp_eq = icmp eq i32 %2, 10
  br i1 %cmp_eq, label %for.end, label %for.cond

  ; CHECK: call void @loop_end()
  ; CHECK-NEXT: ret void

for.end:
  ret void
}