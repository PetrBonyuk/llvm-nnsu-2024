; RUN: opt -load-pass-plugin %llvmshlibdir/SafronovInstrumentFunctions%pluginext -passes=instr_func -S %s | FileCheck %s

define dso_local void @instrument_start() {
entry:
  ret void
}

define dso_local void @instrument_end() {
entry:
  ret void
}

define dso_local void @_Z5emptyv() #0 {
entry:
  ret void
}

; CHECK-LABEL: @_Z5emptyv
; CHECK: call void @instrument_start()
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret void

define dso_local void @_Z5empty_with_calls() #0 {
entry:
  call void @instrument_start()
  call void @instrument_end()
  ret void
}

; CHECK-LABEL: @_Z5empty_with_calls
; CHECK: call void @instrument_start()
; CHECK-NOT: call void @instrument_start()
; CHECK: call void @instrument_end()
; CHECK-NOT: call void @instrument_end()
; CHECK: ret void

define dso_local noundef i32 @_Z4multi(i32 noundef %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %a.addr, align 4
  %mul = mul nsw i32 %0, %1
  ret i32 %mul
}

; CHECK-LABEL: @_Z4multi
; CHECK: call void @instrument_start()
; CHECK-NEXT: %a.addr = alloca i32, align 4
; CHECK-NEXT: store i32 %a, ptr %a.addr, align 4
; CHECK-NEXT: %0 = load i32, ptr %a.addr, align 4
; CHECK-NEXT: %1 = load i32, ptr %a.addr, align 4
; CHECK-NEXT: %mul = mul nsw i32 %0, %1
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret i32 %mul

define dso_local noundef i32 @_Z9incrementi(i32 noundef %x) #0 {
entry:
  %x.addr = alloca i32, align 4
  store i32 %x, ptr %x.addr, align 4
  %0 = load i32, ptr %x.addr, align 4
  %add = add nsw i32 %0, 1
  ret i32 %add
}

; CHECK-LABEL: @_Z9incrementi
; CHECK: call void @instrument_start()
; CHECK-NEXT: %x.addr = alloca i32, align 4
; CHECK-NEXT: store i32 %x, ptr %x.addr, align 4
; CHECK-NEXT: %0 = load i32, ptr %x.addr, align 4
; CHECK-NEXT: %add = add nsw i32 %0, 1
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret i32 %add

define dso_local noundef i32 @_Z15complexFunctionii(i32 noundef %a, i32 noundef %b) #0 {
entry:
  %a.addr = alloca i32, align 4
  %b.addr = alloca i32, align 4
  %c = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  store i32 %b, ptr %b.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %b.addr, align 4
  %add = add nsw i32 %0, %1
  store i32 %add, ptr %c, align 4
  %2 = load i32, ptr %c, align 4
  %mul = mul nsw i32 %2, 2
  store i32 %mul, ptr %c, align 4
  %3 = load i32, ptr %c, align 4
  ret i32 %3
}

; CHECK-LABEL: @_Z15complexFunctionii
; CHECK: call void @instrument_start()
; CHECK-NEXT: %a.addr = alloca i32, align 4
; CHECK-NEXT: %b.addr = alloca i32, align 4
; CHECK-NEXT: %c = alloca i32, align 4
; CHECK-NEXT: store i32 %a, ptr %a.addr, align 4
; CHECK-NEXT: store i32 %b, ptr %b.addr, align 4
; CHECK-NEXT: %0 = load i32, ptr %a.addr, align 4
; CHECK-NEXT: %1 = load i32, ptr %b.addr, align 4
; CHECK-NEXT: %add = add nsw i32 %0, %1
; CHECK-NEXT: store i32 %add, ptr %c, align 4
; CHECK-NEXT: %2 = load i32, ptr %c, align 4
; CHECK-NEXT: %mul = mul nsw i32 %2, 2
; CHECK-NEXT: store i32 %mul, ptr %c, align 4
; CHECK-NEXT: %3 = load i32, ptr %c, align 4
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret i32 %3
