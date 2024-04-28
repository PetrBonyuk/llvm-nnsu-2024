; RUN: opt -load-pass-plugin %llvmshlibdir/sredneva_anastasiya_loop_wrapper_pass_plugin%pluginext\
; RUN: -passes=loop-wrapper -S %s | FileCheck %s

; int myFun(int a)
; {
;     int sum = 0;
;     for (int i = 0; i < 100; ++i)
;     {
;         sum += a;
;     }
;     return sum;
; }

define dso_local noundef i32 @_Z5myFuni(i32 noundef %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %sum = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 0, ptr %sum, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %a.addr, align 4
  %2 = load i32, ptr %sum, align 4
  %add = add nsw i32 %2, %1
  store i32 %add, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %4 = load i32, ptr %sum, align 4
  ret i32 %4
}

; CHECK: define dso_local noundef i32 @_Z5myFuni(i32 noundef %a) {
; CHECK: entry:
; CHECK-NEXT:   %a.addr = alloca i32, align 4
; CHECK-NEXT:   %sum = alloca i32, align 4
; CHECK-NEXT:   %i = alloca i32, align 4
; CHECK-NEXT:   store i32 %a, ptr %a.addr, align 4
; CHECK-NEXT:   store i32 0, ptr %sum, align 4
; CHECK-NEXT:   store i32 0, ptr %i, align 4
; CHECK-NEXT:   call void @loop_start()
; CHECK-NEXT:   br label %for.cond
; CHECK-EMPTY: 
; CHECK: for.cond:                                         ; preds = %for.inc, %entry
; CHECK-NEXT:   %0 = load i32, ptr %i, align 4
; CHECK-NEXT:   %cmp = icmp slt i32 %0, 100
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end
; CHECK-EMPTY: 
; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %1 = load i32, ptr %a.addr, align 4
; CHECK-NEXT:   %2 = load i32, ptr %sum, align 4
; CHECK-NEXT:   %add = add nsw i32 %2, %1
; CHECK-NEXT:   store i32 %add, ptr %sum, align 4
; CHECK-NEXT:   br label %for.inc
; CHECK-EMPTY: 
; CHECK: for.inc:                                          ; preds = %for.body
; CHECK-NEXT:   %3 = load i32, ptr %i, align 4
; CHECK-NEXT:   %inc = add nsw i32 %3, 1
; CHECK-NEXT:   store i32 %inc, ptr %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK-EMPTY: 
; CHECK: for.end:                                          ; preds = %for.cond
; CHECK-NEXT:   call void @loop_end()
; CHECK-NEXT:   %4 = load i32, ptr %sum, align 4
; CHECK-NEXT:   ret i32 %4
; CHECK-NEXT: }

; CHECK: declare void @loop_start()
; CHECK: declare void @loop_end()


; void loop_start();
; void loop_end();

; void simple_while(int n, int inv)
; {
;     if (n > 0)
;     {
;         int res = 0;
;         unsigned counter = 0;
;         while(counter < n)
;         {
;             res -= inv;
;             counter++;
;         }
;     }
; }

declare void @loop_start()
declare void @loop_end()

define dso_local void @_Z12simple_whileii(i32 noundef %n, i32 noundef %inv) #0 {
entry:
  %n.addr = alloca i32, align 4
  %inv.addr = alloca i32, align 4
  %res = alloca i32, align 4
  %counter = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store i32 %inv, ptr %inv.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.end

if.then:                                          ; preds = %entry
  store i32 0, ptr %res, align 4
  store i32 0, ptr %counter, align 4
  br label %while.cond

while.cond:                                       ; preds = %while.body, %if.then
  %1 = load i32, ptr %counter, align 4
  %2 = load i32, ptr %n.addr, align 4
  %cmp1 = icmp ult i32 %1, %2
  br i1 %cmp1, label %while.body, label %while.end

while.body:                                       ; preds = %while.cond
  %3 = load i32, ptr %inv.addr, align 4
  %4 = load i32, ptr %res, align 4
  %sub = sub nsw i32 %4, %3
  store i32 %sub, ptr %res, align 4
  %5 = load i32, ptr %counter, align 4
  %inc = add i32 %5, 1
  store i32 %inc, ptr %counter, align 4
  br label %while.cond

while.end:                                        ; preds = %while.cond
  br label %if.end

if.end:                                           ; preds = %while.end, %entry
  ret void
}

; CHECK: define dso_local void @_Z12simple_whileii(i32 noundef %n, i32 noundef %inv) {
; CHECK: entry:
; CHECK-NEXT:   %n.addr = alloca i32, align 4
; CHECK-NEXT:   %inv.addr = alloca i32, align 4
; CHECK-NEXT:   %res = alloca i32, align 4
; CHECK-NEXT:   %counter = alloca i32, align 4
; CHECK-NEXT:   store i32 %n, ptr %n.addr, align 4
; CHECK-NEXT:   store i32 %inv, ptr %inv.addr, align 4
; CHECK-NEXT:   %0 = load i32, ptr %n.addr, align 4
; CHECK-NEXT:   %cmp = icmp sgt i32 %0, 0
; CHECK-NEXT:   br i1 %cmp, label %if.then, label %if.end
; CHECK-EMPTY: 
; CHECK: if.then:                                          ; preds = %entry
; CHECK-NEXT:   store i32 0, ptr %res, align 4
; CHECK-NEXT:   store i32 0, ptr %counter, align 4
; CHECK-NEXT:   call void @loop_start()
; CHECK-NEXT:   br label %while.cond
; CHECK-EMPTY: 
; CHECK: while.cond:                                       ; preds = %while.body, %if.then
; CHECK-NEXT:   %1 = load i32, ptr %counter, align 4
; CHECK-NEXT:   %2 = load i32, ptr %n.addr, align 4
; CHECK-NEXT:   %cmp1 = icmp ult i32 %1, %2
; CHECK-NEXT:   br i1 %cmp1, label %while.body, label %while.end
; CHECK-EMPTY: 
; CHECK: while.body:                                       ; preds = %while.cond
; CHECK-NEXT:   %3 = load i32, ptr %inv.addr, align 4
; CHECK-NEXT:   %4 = load i32, ptr %res, align 4
; CHECK-NEXT:   %sub = sub nsw i32 %4, %3
; CHECK-NEXT:   store i32 %sub, ptr %res, align 4
; CHECK-NEXT:   %5 = load i32, ptr %counter, align 4
; CHECK-NEXT:   %inc = add i32 %5, 1
; CHECK-NEXT:   store i32 %inc, ptr %counter, align 4
; CHECK-NEXT:   br label %while.cond
; CHECK-EMPTY: 
; CHECK-NEXT: while.end:                                        ; preds = %while.cond
; CHECK-NEXT:   call void @loop_end()
; CHECK-NEXT:   br label %if.end
; CHECK-EMPTY:
; CHECK-NEXT: if.end:                                           ; preds = %while.end, %entry
; CHECK-NEXT:   ret void
; CHECK-NEXT: }


;int no_loop(int n, int inv) {
;	if (n > 0) {
;		inv *= 2;
;	} else {
;		inv *= 4;
;	}
;	return inv;
;}

define dso_local noundef i32 @_Z7no_loopii(i32 noundef %n, i32 noundef %inv) #0 {
entry:
  %n.addr = alloca i32, align 4
  %inv.addr = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store i32 %inv, ptr %inv.addr, align 4
  %0 = load i32, ptr %n.addr, align 4
  %cmp = icmp sgt i32 %0, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %1 = load i32, ptr %inv.addr, align 4
  %mul = mul nsw i32 %1, 2
  store i32 %mul, ptr %inv.addr, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %2 = load i32, ptr %inv.addr, align 4
  %mul1 = mul nsw i32 %2, 4
  store i32 %mul1, ptr %inv.addr, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  %3 = load i32, ptr %inv.addr, align 4
  ret i32 %3
}

; CHECK: define dso_local noundef i32 @_Z7no_loopii(i32 noundef %n, i32 noundef %inv) {
; CHECK: entry:
; CHECK-NEXT:   %n.addr = alloca i32, align 4
; CHECK-NEXT:   %inv.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 %n, ptr %n.addr, align 4
; CHECK-NEXT:   store i32 %inv, ptr %inv.addr, align 4
; CHECK-NEXT:   %0 = load i32, ptr %n.addr, align 4
; CHECK-NEXT:   %cmp = icmp sgt i32 %0, 0
; CHECK-NEXT:   br i1 %cmp, label %if.then, label %if.else
; CHECK-EMPTY: 
; CHECK: if.then:                                          ; preds = %entry
; CHECK-NEXT:   %1 = load i32, ptr %inv.addr, align 4
; CHECK-NEXT:   %mul = mul nsw i32 %1, 2
; CHECK-NEXT:   store i32 %mul, ptr %inv.addr, align 4
; CHECK-NEXT:   br label %if.end
; CHECK-EMPTY: 
; CHECK: if.else:                                          ; preds = %entry
; CHECK-NEXT:   %2 = load i32, ptr %inv.addr, align 4
; CHECK-NEXT:   %mul1 = mul nsw i32 %2, 4
; CHECK-NEXT:   store i32 %mul1, ptr %inv.addr, align 4
; CHECK-NEXT:   br label %if.end
; CHECK-EMPTY: 
; CHECK: if.end:                                           ; preds = %if.else, %if.then
; CHECK-NEXT:   %3 = load i32, ptr %inv.addr, align 4
; CHECK-NEXT:   ret i32 %3
; CHECK-NEXT: }


;void loop_start();
;void loop_end();

;int alreadyHaveLoopWrapper(int a) {
;	 int sum = 0;
;	 loop_start();
;	 for (int i = 0; i < 100; i++) {
;	 	 sum += a;
;	 }
;	 loop_end();
;	 return sum;
;}

define dso_local noundef i32 @_Z22alreadyHaveLoopWrapperi(i32 noundef %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  %sum = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 0, ptr %sum, align 4
  call void @loop_start()
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 100
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %a.addr, align 4
  %2 = load i32, ptr %sum, align 4
  %add = add nsw i32 %2, %1
  store i32 %add, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %3 = load i32, ptr %i, align 4
  %inc = add nsw i32 %3, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  call void @loop_end()
  %4 = load i32, ptr %sum, align 4
  ret i32 %4
}

; CHECK: define dso_local noundef i32 @_Z22alreadyHaveLoopWrapperi(i32 noundef %a) {
; CHECK: entry:
; CHECK-NEXT:   %a.addr = alloca i32, align 4
; CHECK-NEXT:   %sum = alloca i32, align 4
; CHECK-NEXT:   %i = alloca i32, align 4
; CHECK-NEXT:   store i32 %a, ptr %a.addr, align 4
; CHECK-NEXT:   store i32 0, ptr %sum, align 4
; CHECK-NEXT:   call void @loop_start()
; CHECK-NEXT:   store i32 0, ptr %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK-EMPTY: 
; CHECK: for.cond:                                         ; preds = %for.inc, %entry
; CHECK-NEXT:   %0 = load i32, ptr %i, align 4
; CHECK-NEXT:   %cmp = icmp slt i32 %0, 100
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end
; CHECK-EMPTY: 
; CHECK: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %1 = load i32, ptr %a.addr, align 4
; CHECK-NEXT:   %2 = load i32, ptr %sum, align 4
; CHECK-NEXT:   %add = add nsw i32 %2, %1
; CHECK-NEXT:   store i32 %add, ptr %sum, align 4
; CHECK-NEXT:   br label %for.inc
; CHECK-EMPTY: 
; CHECK: for.inc:                                          ; preds = %for.body
; CHECK-NEXT:   %3 = load i32, ptr %i, align 4
; CHECK-NEXT:   %inc = add nsw i32 %3, 1
; CHECK-NEXT:   store i32 %inc, ptr %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK-EMPTY: 
; CHECK: for.end:                                          ; preds = %for.cond
; CHECK-NEXT:   call void @loop_end()
; CHECK-NEXT:   %4 = load i32, ptr %sum, align 4
; CHECK-NEXT:   ret i32 %4
; CHECK-NEXT: }
