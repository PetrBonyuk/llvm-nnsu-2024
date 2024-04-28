; RUN: opt -load-pass-plugin %llvmshlibdir/Instrument_Functions_Pass_Bodrov_Daniil_FIIT1%pluginext\
; RUN: -passes=instrument_functions_pass -S %s | FileCheck %s

; CHECK-LABEL: @_Z11calculate_xi
; CHECK: call void @instrument_start()
; CHECL-NEXT: %num.addr = alloca i32, align 4
; CHECK: call void @instrument_end()
; CHECK-NEXT: ret i32 %4

define dso_local noundef i32 @_Z11calculate_xi(i32 noundef %num) #0 {
entry:
  %num.addr = alloca i32, align 4
  store i32 %num, ptr %num.addr, align 4
  %0 = load i32, ptr %num.addr, align 4
  %add = add nsw i32 %0, 1
  %1 = load i32, ptr %num.addr, align 4
  %add1 = add nsw i32 %1, %add
  store i32 %add1, ptr %num.addr, align 4
  %2 = load i32, ptr %num.addr, align 4
  %3 = load i32, ptr %num.addr, align 4
  %mul = mul nsw i32 %3, %2
  store i32 %mul, ptr %num.addr, align 4
  %4 = load i32, ptr %num.addr, align 4
  ret i32 %4
}

; CHECK-LABEL: @_Z4funcv
; CHECK: call void @instrument_start()
; CHECK-NEXT: call void @instrument_end()
; CHECK-NEXT: ret void
define dso_local void @_Z4funcv() #0 {
entry:
  ret void
}

; CHECK-LABEL:  @_Z5clampddd
; CHECK: call void @instrument_start()
; CHECK-NEXT: %retval = alloca double, align 8
; CHECK: call void @instrument_end()
; CHECK-NEXT: ret double %7
define dso_local noundef double @_Z5clampddd(double noundef %v, double noundef %lo, double noundef %hi) #0 {
entry:
  call void @instrument_start()
  %retval = alloca double, align 8
  %v.addr = alloca double, align 8
  %lo.addr = alloca double, align 8
  %hi.addr = alloca double, align 8
  store double %v, ptr %v.addr, align 8
  store double %lo, ptr %lo.addr, align 8
  store double %hi, ptr %hi.addr, align 8
  %0 = load double, ptr %v.addr, align 8
  %1 = load double, ptr %hi.addr, align 8
  %cmp = fcmp ogt double %0, %1
  br i1 %cmp, label %if.then, label %if.else

if.then:                                          ; preds = %entry
  %2 = load double, ptr %hi.addr, align 8
  store double %2, ptr %retval, align 8
  br label %return

if.else:                                          ; preds = %entry
  %3 = load double, ptr %v.addr, align 8
  %4 = load double, ptr %lo.addr, align 8
  %cmp1 = fcmp olt double %3, %4
  br i1 %cmp1, label %if.then2, label %if.end

if.then2:                                         ; preds = %if.else
  %5 = load double, ptr %lo.addr, align 8
  store double %5, ptr %retval, align 8
  br label %return

if.end:                                           ; preds = %if.else
  br label %if.end3

if.end3:                                          ; preds = %if.end
  %6 = load double, ptr %v.addr, align 8
  store double %6, ptr %retval, align 8
  br label %return

return:                                           ; preds = %if.end3, %if.then2, %if.then
  %7 = load double, ptr %retval, align 8
  call void @instrument_end()
  ret double %7
}

declare void @instrument_start()
declare void @instrument_end()
