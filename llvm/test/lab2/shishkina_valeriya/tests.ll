; RUN: opt -load-pass-plugin %llvmshlibdir/ShishkinaInstrumentFunctions%pluginext -passes=instrumentation-functions -S %s | FileCheck %s
define dso_local void @instrument_start() {
entry:
  ret void
}

define dso_local void @instrument_end() {
entry:
  ret void
}

; void first() {
;     return;
; }

; int second(int a) {
;     return a * a;
; }

; int third(int s){
;     @instrument_start();
;     s=s+10;
;     s=s+30;
;     @instrument_end();
;     return s;
; }

define dso_local void @_Z5firstv() #0 {
entry:
  ret void
}
; CHECK-LABEL: @_Z5firstv
; CHECK: call void @instrument_start()
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret void

define dso_local noundef i32 @_Z6secondi(i32 noundef %a) #0 {
entry:
  %a.addr = alloca i32, align 4
  store i32 %a, ptr %a.addr, align 4
  %0 = load i32, ptr %a.addr, align 4
  %1 = load i32, ptr %a.addr, align 4
  %mul = mul nsw i32 %0, %1
  ret i32 %mul
}

; CHECK-LABEL: @_Z6secondi
; CHECK: call void @instrument_start()
; CHECK-NEXT: %a.addr = alloca i32, align 4
; CHECK-NEXT: store i32 %a, ptr %a.addr, align 4
; CHECK-NEXT: %0 = load i32, ptr %a.addr, align 4
; CHECK-NEXT: %1 = load i32, ptr %a.addr, align 4
; CHECK-NEXT: %mul = mul nsw i32 %0, %1
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret i32 %mul

define dso_local noundef i32 @_Z5thirdi(i32 noundef %s) #0 {
entry:
  %s.addr = alloca i32, align 4
  store i32 %s, ptr %s.addr, align 4
  call void @instrument_start()
  %0 = load i32, ptr %s.addr, align 4
  %add = add nsw i32 %0, 10
  store i32 %add, ptr %s.addr, align 4
  %1 = load i32, ptr %s.addr, align 4
  %add1 = add nsw i32 %1, 30
  store i32 %add1, ptr %s.addr, align 4
  %2 = load i32, ptr %s.addr, align 4
  call void @instrument_end()
  ret i32 %2
}

; CHECK-LABEL: @_Z5thirdi
; CHECK: %s.addr = alloca i32, align 4
; CHECK-NEXT: store i32 %s, ptr %s.addr, align 4
; CHECK-NEXT: call void @instrument_start()
; CHECK-NEXT: %0 = load i32, ptr %s.addr, align 4
; CHECK-NEXT: %add = add nsw i32 %0, 10
; CHECK-NEXT: store i32 %add, ptr %s.addr, align 4
; CHECK-NEXT: %1 = load i32, ptr %s.addr, align 4
; CHECK-NEXT: %add1 = add nsw i32 %1, 30
; CHECK-NEXT: store i32 %add1, ptr %s.addr, align 4
; CHECK-NEXT: %2 = load i32, ptr %s.addr, align 4
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret i32 %2