; RUN: opt -load-pass-plugin=%llvmshlibdir/PetrovWrapLoopPlugin%shlibext -passes=PetrovWrapLoopPlugin -S %s | FileCheck %s


; void sum() {
;     int sum = 0;
;     for (int i = 1; i < 5; i++) {
;         sum++;
;     }
; }

; void simple_while() {
;     int i = 0;
;     while (i < 10) {
;         ++i;
;     }
; }

; void nested_loop() {
;     for (int i = 0; i < 3; i++) {
;         for (int j = 0; j < 5; j++) {
;             // Body of the inner loop
;         }
;     }
; }


define dso_local void @sum() {
entry:
  %sum = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 0, ptr %sum, align 4
  store i32 1, ptr %i, align 4
  br label %for.cond
; CHECK:    call void @loop_start()
; CHECK-NEXT:    br label %for.cond

for.cond:
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 5
  br i1 %cmp, label %for.body, label %for.end

for.body:
  %1 = load i32, ptr %sum, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %sum, align 4
  br label %for.inc

for.inc:
  %2 = load i32, ptr %i, align 4
  %inc1 = add nsw i32 %2, 1
  store i32 %inc1, ptr %i, align 4
  br label %for.cond

; CHECK:       for.end:
; CHECK-NEXT:    call void @loop_end()
for.end:
  ret void
}


define dso_local void @simple_while() {
entry:
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %while.cond
; CHECK:      call void @loop_start()
; CHECK-NEXT: br label %while.cond

while.cond:
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 10
  br i1 %cmp, label %while.body, label %while.end

while.body:
  %1 = load i32, ptr %i, align 4
  %inc = add nsw i32 %1, 1
  store i32 %inc, ptr %i, align 4
  br label %while.cond

; CHECK:       while.end:
; CHECK-NEXT:  call void @loop_end()
while.end:
  ret void
}

define dso_local void @nested_loop() {
entry:
  %i = alloca i32, align 4
  store i32 0, ptr %i, align 4
  br label %outer.loop.cond
; CHECK:    call void @loop_start()
; CHECK-NEXT:    br label %outer.loop.cond

outer.loop.cond:
  %0 = load i32, ptr %i, align 4
  %cmp1 = icmp slt i32 %0, 3
  br i1 %cmp1, label %outer.loop.body, label %outer.loop.end

outer.loop.body:
  %j = alloca i32, align 4
  store i32 0, ptr %j, align 4
  br label %inner.loop.cond
; CHECK:       call void @loop_start()
; CHECK-NEXT:    br label %inner.loop.cond

inner.loop.cond:
  %1 = load i32, ptr %j, align 4
  %cmp2 = icmp slt i32 %1, 5
  br i1 %cmp2, label %inner.loop.body, label %inner.loop.end
  
inner.loop.body:
  ; Body of the inner loop
  br label %inner.loop.inc

inner.loop.inc:
  %2 = load i32, ptr %j, align 4
  %inc2 = add nsw i32 %2, 1
  store i32 %inc2, ptr %j, align 4
  br label %inner.loop.cond

; CHECK:       inner.loop.end:
; CHECK-NEXT:    call void @loop_end()
inner.loop.end:
  br label %outer.loop.inc

outer.loop.inc:
  %3 = load i32, ptr %i, align 4
  %inc1 = add nsw i32 %3, 1
  store i32 %inc1, ptr %i, align 4
  br label %outer.loop.cond

; CHECK:       outer.loop.end:
; CHECK-NEXT:    call void @loop_end()
outer.loop.end:
  ret void
}     
