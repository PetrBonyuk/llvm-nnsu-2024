; RUN: opt -load-pass-plugin %llvmshlibdir/akopyan_zal_inline_pass_plugin%pluginext\
; RUN: -passes=akopyan-inlining -S %s | FileCheck %s


target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; COM: Expected inline

; void func() {
;   int a[] = {1, 2, 3};
;   int sum = 0;
;   for (int i = 0; i < 3; ++i) {
;     sum += a[i];
;   }
; }
; 
; int myfoo(int a, int b) {
;   a++;
;   b++;
;   func();
;   return a + b;
; }

@__const._Z4funcv.a = private unnamed_addr constant [3 x i32] [i32 1, i32 2, i32 3], align 4

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z4funcv() #0 {
entry:
  %a = alloca [3 x i32], align 4
  %sum = alloca i32, align 4
  %i = alloca i32, align 4
  call void @llvm.memcpy.p0.p0.i64(ptr align 4 %a, ptr align 4 @__const._Z4funcv.a, i64 12, i1 false)
  store i32 0, ptr %sum, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %cmp = icmp slt i32 %0, 3
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %1 = load i32, ptr %i, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [3 x i32], ptr %a, i64 0, i64 %idxprom
  %2 = load i32, ptr %arrayidx, align 4
  %3 = load i32, ptr %sum, align 4
  %add = add nsw i32 %3, %2
  store i32 %add, ptr %sum, align 4
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %4 = load i32, ptr %i, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; Function Attrs: nocallback nofree nounwind willreturn memory(argmem: readwrite)
declare void @llvm.memcpy.p0.p0.i64(ptr noalias nocapture writeonly, ptr noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z5myfooii(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %inc = add nsw i32 %0, 1
  store i32 %inc, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %inc1 = add nsw i32 %1, 1
  store i32 %inc1, ptr %b.addr, align 4
  call void @_Z4funcv()
  %2 = load i32, ptr %a.addr, align 4
  %3 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %2, %3
  ret i32 %add
}

; CHECK: define dso_local noundef i32 @_Z5myfooii(i32 noundef %a, i32 noundef %b) #0 {
; CHECK: entry:
; CHECK-NEXT:   %a.addr = alloca i32, align 4
; CHECK-NEXT:   %b.addr = alloca i32, align 4
; CHECK-NEXT:   store i32 %a, ptr %a.addr, align 4
; CHECK-NEXT:   store i32 %b, ptr %b.addr, align 4
; CHECK-NEXT:   %0 = load i32, ptr %a.addr, align 4
; CHECK-NEXT:   %inc = add nsw i32 %0, 1
; CHECK-NEXT:   store i32 %inc, ptr %a.addr, align 4
; CHECK-NEXT:   %1 = load i32, ptr %b.addr, align 4
; CHECK-NEXT:   %inc1 = add nsw i32 %1, 1
; CHECK-NEXT:   store i32 %inc1, ptr %b.addr, align 4
; CHECK-NEXT:   br label %entry.inlined.0
; CHECK-EMPTY: 
; CHECK-NEXT: entry.inlined.0:                                  ; preds = %entry
; CHECK-NEXT:   %2 = alloca [3 x i32], align 4
; CHECK-NEXT:   %3 = alloca i32, align 4
; CHECK-NEXT:   %4 = alloca i32, align 4
; CHECK-NEXT:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %2, ptr align 4 @__const._Z4funcv.a, i64 12, i1 false)
; CHECK-NEXT:   store i32 0, ptr %3, align 4
; CHECK-NEXT:   store i32 0, ptr %4, align 4
; CHECK-NEXT:   br label %for.cond.inlined.0
; CHECK-EMPTY: 
; CHECK-NEXT: for.cond.inlined.0:                               ; preds = %for.inc.inlined.0, %entry.inlined.0
; CHECK-NEXT:   %5 = load i32, ptr %4, align 4
; CHECK-NEXT:   %6 = icmp slt i32 %5, 3
; CHECK-NEXT:   br i1 %6, label %for.body.inlined.0, label %for.end.inlined.0
; CHECK-EMPTY: 
; CHECK-NEXT: for.body.inlined.0:                               ; preds = %for.cond.inlined.0
; CHECK-NEXT:   %7 = load i32, ptr %4, align 4
; CHECK-NEXT:   %8 = sext i32 %7 to i64
; CHECK-NEXT:   %9 = getelementptr inbounds [3 x i32], ptr %2, i64 0, i64 %8
; CHECK-NEXT:   %10 = load i32, ptr %9, align 4
; CHECK-NEXT:   %11 = load i32, ptr %3, align 4
; CHECK-NEXT:   %12 = add nsw i32 %11, %10
; CHECK-NEXT:   store i32 %12, ptr %3, align 4
; CHECK-NEXT:   br label %for.inc.inlined.0
; CHECK-EMPTY: 
; CHECK-NEXT: for.inc.inlined.0:                                ; preds = %for.body.inlined.0
; CHECK-NEXT:   %13 = load i32, ptr %4, align 4
; CHECK-NEXT:   %14 = add nsw i32 %13, 1
; CHECK-NEXT:   store i32 %14, ptr %4, align 4
; CHECK-NEXT:   br label %for.cond.inlined.0
; CHECK-EMPTY:  
; CHECK-NEXT: for.end.inlined.0:                                ; preds = %for.cond.inlined.0
; CHECK-NEXT:   br label %entry.splited.0
; CHECK-EMPTY:  
; CHECK-NEXT: entry.splited.0:                                  ; preds = %for.end.inlined.0
; CHECK-NEXT:   %15 = load i32, ptr %a.addr, align 4
; CHECK-NEXT:   %16 = load i32, ptr %b.addr, align 4
; CHECK-NEXT:   %add = add nsw i32 %15, %16
; CHECK-NEXT:   ret i32 %add
; CHECK-NEXT: }

; ----------------------------------------------------------------------------------

; COM: Wasn't expected inline

; int isEven(int num) {
;     if (num % 2 == 0) {
;         return true;
;     } else {
;         return false;
;     }
; }

; void filterEvenNumbers(int* numbers, int size, int* evenNumbers, int& evenSize) {
;     evenSize = 0;
;     for (int i = 0; i < size; i++) {
;         if (isEven(numbers[i])) {
;             evenNumbers[evenSize++] = numbers[i];
;         }
;     }
; }

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z6isEveni(i32 noundef %num) #0 {
entry:
  %retval = alloca i32, align 4
  %num.addr = alloca i32, align 4
  store i32 %num, ptr %num.addr, align 4
  %0 = load i32, ptr %num.addr, align 4
  %rem = srem i32 %0, 2
  %cmp = icmp eq i32 %rem, 0
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  store i32 1, ptr %retval, align 4
  br label %return

if.else:                                          ; preds = %entry
  store i32 0, ptr %retval, align 4
  br label %return

return:                                           ; preds = %if.else, %if.then
  %1 = load i32, ptr %retval, align 4
  ret i32 %1
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z17filterEvenNumbersPiiS_Ri(ptr noundef %numbers, i32 noundef %size, ptr noundef %evenNumbers, ptr noundef nonnull align 4 dereferenceable(4) %evenSize) #0 {
entry:
  %numbers.addr = alloca ptr, align 8
  %size.addr = alloca i32, align 4
  %evenNumbers.addr = alloca ptr, align 8
  %evenSize.addr = alloca ptr, align 8
  %i = alloca i32, align 4
  store ptr %numbers, ptr %numbers.addr, align 8
  store i32 %size, ptr %size.addr, align 4
  store ptr %evenNumbers, ptr %evenNumbers.addr, align 8
  store ptr %evenSize, ptr %evenSize.addr, align 8
  %0 = load ptr, ptr %evenSize.addr, align 8
  store i32 0, ptr %0, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %1 = load i32, ptr %i, align 4
  %2 = load i32, ptr %size.addr, align 4
  %cmp = icmp slt i32 %1, %2
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %3 = load ptr, ptr %numbers.addr, align 8
  %4 = load i32, ptr %i, align 4
  %idxprom = sext i32 %4 to i64
  %arrayidx = getelementptr inbounds i32, ptr %3, i64 %idxprom
  %5 = load i32, ptr %arrayidx, align 4
  %call = call noundef i32 @_Z6isEveni(i32 noundef %5)
  %tobool = icmp ne i32 %call, 0
  br i1 %tobool, label %if.then, label %if.end

if.then:                                          ; preds = %for.body
  %6 = load ptr, ptr %numbers.addr, align 8
  %7 = load i32, ptr %i, align 4
  %idxprom1 = sext i32 %7 to i64
  %arrayidx2 = getelementptr inbounds i32, ptr %6, i64 %idxprom1
  %8 = load i32, ptr %arrayidx2, align 4
  %9 = load ptr, ptr %evenNumbers.addr, align 8
  %10 = load ptr, ptr %evenSize.addr, align 8
  %11 = load i32, ptr %10, align 4
  %inc = add nsw i32 %11, 1
  store i32 %inc, ptr %10, align 4
  %idxprom3 = sext i32 %11 to i64
  %arrayidx4 = getelementptr inbounds i32, ptr %9, i64 %idxprom3
  store i32 %8, ptr %arrayidx4, align 4
  br label %if.end

if.end:                                           ; preds = %if.then, %for.body
  br label %for.inc

for.inc:                                          ; preds = %if.end
  %12 = load i32, ptr %i, align 4
  %inc5 = add nsw i32 %12, 1
  store i32 %inc5, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  ret void
}

; CHECK: define dso_local void @_Z17filterEvenNumbersPiiS_Ri(ptr noundef %numbers, i32 noundef %size, ptr noundef %evenNumbers, ptr noundef nonnull align 4 dereferenceable(4) %evenSize) #0 {
; CHECK: entry:
; CHECK-NEXT:   %numbers.addr = alloca ptr, align 8
; CHECK-NEXT:   %size.addr = alloca i32, align 4
; CHECK-NEXT:   %evenNumbers.addr = alloca ptr, align 8
; CHECK-NEXT:   %evenSize.addr = alloca ptr, align 8
; CHECK-NEXT:   %i = alloca i32, align 4
; CHECK-NEXT:   store ptr %numbers, ptr %numbers.addr, align 8
; CHECK-NEXT:   store i32 %size, ptr %size.addr, align 4
; CHECK-NEXT:   store ptr %evenNumbers, ptr %evenNumbers.addr, align 8
; CHECK-NEXT:   store ptr %evenSize, ptr %evenSize.addr, align 8
; CHECK-NEXT:   %0 = load ptr, ptr %evenSize.addr, align 8
; CHECK-NEXT:   store i32 0, ptr %0, align 4
; CHECK-NEXT:   store i32 0, ptr %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK-EMPTY: 
; CHECK-NEXT: for.cond:                                         ; preds = %for.inc, %entry
; CHECK-NEXT:   %1 = load i32, ptr %i, align 4
; CHECK-NEXT:   %2 = load i32, ptr %size.addr, align 4
; CHECK-NEXT:   %cmp = icmp slt i32 %1, %2
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end
; CHECK-EMPTY: 
; CHECK-NEXT: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %3 = load ptr, ptr %numbers.addr, align 8
; CHECK-NEXT:   %4 = load i32, ptr %i, align 4
; CHECK-NEXT:   %idxprom = sext i32 %4 to i64
; CHECK-NEXT:   %arrayidx = getelementptr inbounds i32, ptr %3, i64 %idxprom
; CHECK-NEXT:   %5 = load i32, ptr %arrayidx, align 4
; CHECK-NEXT:   %call = call noundef i32 @_Z6isEveni(i32 noundef %5)
; CHECK-NEXT:   %tobool = icmp ne i32 %call, 0
; CHECK-NEXT:   br i1 %tobool, label %if.then, label %if.end
; CHECK-EMPTY: 
; CHECK-NEXT: if.then:                                          ; preds = %for.body
; CHECK-NEXT:   %6 = load ptr, ptr %numbers.addr, align 8
; CHECK-NEXT:   %7 = load i32, ptr %i, align 4
; CHECK-NEXT:   %idxprom1 = sext i32 %7 to i64
; CHECK-NEXT:   %arrayidx2 = getelementptr inbounds i32, ptr %6, i64 %idxprom1
; CHECK-NEXT:   %8 = load i32, ptr %arrayidx2, align 4
; CHECK-NEXT:   %9 = load ptr, ptr %evenNumbers.addr, align 8
; CHECK-NEXT:   %10 = load ptr, ptr %evenSize.addr, align 8
; CHECK-NEXT:   %11 = load i32, ptr %10, align 4
; CHECK-NEXT:   %inc = add nsw i32 %11, 1
; CHECK-NEXT:   store i32 %inc, ptr %10, align 4
; CHECK-NEXT:   %idxprom3 = sext i32 %11 to i64
; CHECK-NEXT:   %arrayidx4 = getelementptr inbounds i32, ptr %9, i64 %idxprom3
; CHECK-NEXT:   store i32 %8, ptr %arrayidx4, align 4
; CHECK-NEXT:   br label %if.end
; CHECK-EMPTY: 
; CHECK-NEXT: if.end:                                           ; preds = %if.then, %for.body
; CHECK-NEXT:   br label %for.inc
; CHECK-EMPTY: 
; CHECK-NEXT: for.inc:                                          ; preds = %if.end
; CHECK-NEXT:   %12 = load i32, ptr %i, align 4
; CHECK-NEXT:   %inc5 = add nsw i32 %12, 1
; CHECK-NEXT:   store i32 %inc5, ptr %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK-EMPTY: 
; CHECK-NEXT: for.end:                                          ; preds = %for.cond
; CHECK-NEXT:   ret void
; CHECK-NEXT: }

; ----------------------------------------------------------------------------------

; COM: Expected inline

; void someFun() {
;   int a = 4;
;   int b = 6;
;   if (a < b) {
;     a++;
;   } else {
;     b++;
;   }
; }

; int forLoop(int n) {
;   int sum = 0;
;   for (int i = 0; i < n; i++) {
;     sum += i;
;     someFun();
;   }
;   return sum;
; }

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local void @_Z7someFunv() #0 {
entry:
  %a = alloca i32, align 4
  %b = alloca i32, align 4
  store i32 4, ptr %a, align 4
  store i32 6, ptr %b, align 4
  %0 = load i32, ptr %a, align 4
  %1 = load i32, ptr %b, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load i32, ptr %a, align 4
  %inc = add nsw i32 %2, 1
  store i32 %inc, ptr %a, align 4
  br label %if.end

if.else:                                          ; preds = %entry
  %3 = load i32, ptr %b, align 4
  %inc1 = add nsw i32 %3, 1
  store i32 %inc1, ptr %b, align 4
  br label %if.end

if.end:                                           ; preds = %if.else, %if.then
  ret void
}

; Function Attrs: mustprogress noinline nounwind optnone uwtable
define dso_local noundef i32 @_Z7forLoopi(i32 noundef %n) #0 {
entry:
  %n.addr = alloca i32, align 4
  %sum = alloca i32, align 4
  %i = alloca i32, align 4
  store i32 %n, ptr %n.addr, align 4
  store i32 0, ptr %sum, align 4
  store i32 0, ptr %i, align 4
  br label %for.cond

for.cond:                                         ; preds = %for.inc, %entry
  %0 = load i32, ptr %i, align 4
  %1 = load i32, ptr %n.addr, align 4
  %cmp = icmp slt i32 %0, %1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %2 = load i32, ptr %i, align 4
  %3 = load i32, ptr %sum, align 4
  %add = add nsw i32 %3, %2
  store i32 %add, ptr %sum, align 4
  call void @_Z7someFunv()
  br label %for.inc

for.inc:                                          ; preds = %for.body
  %4 = load i32, ptr %i, align 4
  %inc = add nsw i32 %4, 1
  store i32 %inc, ptr %i, align 4
  br label %for.cond

for.end:                                          ; preds = %for.cond
  %5 = load i32, ptr %sum, align 4
  ret i32 %5
}

; CHECK: define dso_local noundef i32 @_Z7forLoopi(i32 noundef %n) #0 {
; CHECK: entry:
; CHECK-NEXT:   %n.addr = alloca i32, align 4
; CHECK-NEXT:   %sum = alloca i32, align 4
; CHECK-NEXT:   %i = alloca i32, align 4
; CHECK-NEXT:   store i32 %n, ptr %n.addr, align 4
; CHECK-NEXT:   store i32 0, ptr %sum, align 4
; CHECK-NEXT:   store i32 0, ptr %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK-EMPTY: 
; CHECK-NEXT: for.cond:                                         ; preds = %for.inc, %entry
; CHECK-NEXT:   %0 = load i32, ptr %i, align 4
; CHECK-NEXT:   %1 = load i32, ptr %n.addr, align 4
; CHECK-NEXT:   %cmp = icmp slt i32 %0, %1
; CHECK-NEXT:   br i1 %cmp, label %for.body, label %for.end
; CHECK-EMPTY: 
; CHECK-NEXT: for.body:                                         ; preds = %for.cond
; CHECK-NEXT:   %2 = load i32, ptr %i, align 4
; CHECK-NEXT:   %3 = load i32, ptr %sum, align 4
; CHECK-NEXT:   %add = add nsw i32 %3, %2
; CHECK-NEXT:   store i32 %add, ptr %sum, align 4
; CHECK-NEXT:   br label %entry.inlined.0
; CHECK-EMPTY: 
; CHECK-NEXT: for.inc:                                          ; preds = %for.body.splited.0
; CHECK-NEXT:   %4 = load i32, ptr %i, align 4
; CHECK-NEXT:   %inc = add nsw i32 %4, 1
; CHECK-NEXT:   store i32 %inc, ptr %i, align 4
; CHECK-NEXT:   br label %for.cond
; CHECK-EMPTY: 
; CHECK-NEXT: for.end:                                          ; preds = %for.cond
; CHECK-NEXT:   %5 = load i32, ptr %sum, align 4
; CHECK-NEXT:   br label %for.body.splited.0
; CHECK-EMPTY: 
; CHECK-NEXT: entry.inlined.0:                                  ; preds = %for.body
; CHECK-NEXT:   %6 = alloca i32, align 4
; CHECK-NEXT:   %7 = alloca i32, align 4
; CHECK-NEXT:   store i32 4, ptr %6, align 4
; CHECK-NEXT:   store i32 6, ptr %7, align 4
; CHECK-NEXT:   %8 = load i32, ptr %6, align 4
; CHECK-NEXT:   %9 = load i32, ptr %7, align 4
; CHECK-NEXT:   %10 = icmp slt i32 %8, %9
; CHECK-NEXT:   br i1 %10, label %if.then.inlined.0, label %if.else.inlined.0
; CHECK-EMPTY: 
; CHECK-NEXT: if.then.inlined.0:                                ; preds = %entry.inlined.0
; CHECK-NEXT:   %11 = load i32, ptr %6, align 4
; CHECK-NEXT:   %12 = add nsw i32 %11, 1
; CHECK-NEXT:   store i32 %12, ptr %6, align 4
; CHECK-NEXT:   br label %if.end.inlined.0
; CHECK-EMPTY: 
; CHECK-NEXT: if.else.inlined.0:                                ; preds = %entry.inlined.0
; CHECK-NEXT:   %13 = load i32, ptr %7, align 4
; CHECK-NEXT:   %14 = add nsw i32 %13, 1
; CHECK-NEXT:   store i32 %14, ptr %7, align 4
; CHECK-NEXT:   br label %if.end.inlined.0
; CHECK-EMPTY: 
; CHECK-NEXT: if.end.inlined.0:                                 ; preds = %if.else.inlined.0, %if.then.inlined.0
; CHECK-NEXT:   br label %for.body.splited.0
; CHECK-EMPTY: 
; CHECK-NEXT: for.body.splited.0:                               ; preds = %if.end.inlined.0, %for.end
; CHECK-NEXT:   br label %for.inc
; CHECK-NEXT: }

attributes #0 = { mustprogress noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cmov,+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { nocallback nofree nounwind willreturn memory(argmem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 8, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 2}
!4 = !{i32 7, !"frame-pointer", i32 2}
