// RUN: split-file %s %t

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=variable\
// RUN: -plugin-arg-rename first-name=a\
// RUN: -plugin-arg-rename second-name=new_var %t/rename_var.cpp
// RUN: FileCheck %s < %t/rename_var.cpp --check-prefix=VAR

// VAR: int func() {
// VAR-NEXT: int new_var = 2, b = 2, c = 2;
// VAR-NEXT: new_var = new_var + b - c;
// VAR-NEXT: new_var++;
// VAR-NEXT: c = b;
// VAR-NEXT:  return new_var;
// VAR-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=variable\
// RUN: -plugin-arg-rename first-name=d\
// RUN: -plugin-arg-rename second-name=new_var %t/rename_other_var.cpp
// RUN: FileCheck %s < %t/rename_other_var.cpp --check-prefix=NON_EXIST_VAR

// NON_EXIST_VAR: int func() {
// NON_EXIST_VAR-NEXT: int a = 2, b = 2, c = 2;
// NON_EXIST_VAR-NEXT: a = a + b - c;
// NON_EXIST_VAR-NEXT: a++;
// NON_EXIST_VAR-NEXT: c = b;
// NON_EXIST_VAR-NEXT:  return a;
// NON_EXIST_VAR-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=function\
// RUN: -plugin-arg-rename first-name=function\
// RUN: -plugin-arg-rename second-name=new_func %t/rename_func.cpp
// RUN: FileCheck %s < %t/rename_func.cpp --check-prefix=FUNC

// FUNC: int new_func(int a) {
// FUNC-NEXT: b = 2;
// FUNC-NEXT: return a + b;
// FUNC-NEXT: }
// FUNC-NEXT: int function2(){
// FUNC-NEXT: int a = new_func(2) + 2;
// FUNC-NEXT: return a;
// FUNC-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=function\
// RUN: -plugin-arg-rename first-name=function\
// RUN: -plugin-arg-rename second-name=new_func %t/rename_other_func.cpp
// RUN: FileCheck %s < %t/rename_other_func.cpp --check-prefix=NON_EXIST_FUNC

// NON_EXIST_FUNC: int func(int a) {
// NON_EXIST_FUNC-NEXT: return a;
// NON_EXIST_FUNC-NEXT: }
// NON_EXIST_FUNC: void func2() {
// NON_EXIST_FUNC-NEXT: int a = func(2);
// NON_EXIST_FUNC-NEXT: int b = func(a) + func(2);
// NON_EXIST_FUNC-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=class\
// RUN: -plugin-arg-rename first-name=abc\
// RUN: -plugin-arg-rename second-name=new_class %t/rename_class.cpp
// RUN: FileCheck %s < %t/rename_class.cpp --check-prefix=CLASS

// CLASS: class new_class{
// CLASS-NEXT: private:
// CLASS-NEXT: int a;
// CLASS-NEXT: public:
// CLASS-NEXT: new_class() {}
// CLASS-NEXT: new_class(int a): a(a) {}
// CLASS-NEXT: ~new_class();
// CLASS-NEXT: };
// CLASS: void func() {
// CLASS-NEXT: new_class a;
// CLASS-NEXT: new_class* var = new new_class(1);
// CLASS-NEXT: delete var;
// CLASS-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=class\
// RUN: -plugin-arg-rename first-name=ab\
// RUN: -plugin-arg-rename second-name=new_class %t/rename_tmp_class.cpp
// RUN: FileCheck %s < %t/rename_tmp_class.cpp --check-prefix=TMP_CLASS

// TMP_CLASS: template <typename T> class new_class {
// TMP_CLASS-NEXT: T a;
// TMP_CLASS-NEXT: ab() {}
// TMP_CLASS-NEXT: ~ab();
// TMP_CLASS-NEXT: };

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=class\
// RUN: -plugin-arg-rename first-name=rty\
// RUN: -plugin-arg-rename second-name=new_class %t/rename_inherited_class.cpp
// RUN: FileCheck %s < %t/rename_inherited_class.cpp --check-prefix=INHERITED_CLASS

// INHERITED_CLASS: class qwe {
// INHERITED_CLASS-NEXT: qwe();
// INHERITED_CLASS-NEXT: ~qwe();
// INHERITED_CLASS-NEXT: };
// INHERITED_CLASS: class new_class : public qwe {
// INHERITED_CLASS-NEXT: new_class();
// INHERITED_CLASS-NEXT: ~new_class();
// INHERITED_CLASS-NEXT: };

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=class\
// RUN: -plugin-arg-rename first-name=ab\
// RUN: -plugin-arg-rename second-name=new_class %t/rename_delete_var_class.cpp
// RUN: FileCheck %s < %t/rename_delete_var_class.cpp --check-prefix=DELETE_VAR

// DELETE_VAR: class abcd {};
// DELETE_VAR: void f() {
// DELETE_VAR-NEXT: abcd* aaa = new abcd();
// DELETE_VAR-NEXT: delete aaa;
// DELETE_VAR-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=class\
// RUN: -plugin-arg-rename first-name=B\
// RUN: -plugin-arg-rename second-name=C %t/rename_other_class.cpp
// RUN: FileCheck %s < %t/rename_other_class.cpp --check-prefix=NON_EXIST_CLASS

// NON_EXIST_CLASS: class A{
// NON_EXIST_CLASS-NEXT: private:
// NON_EXIST_CLASS-NEXT: int var1;
// NON_EXIST_CLASS-NEXT: public:
// NON_EXIST_CLASS-NEXT: A() {};
// NON_EXIST_CLASS-NEXT: ~A() {};
// NON_EXIST_CLASS-NEXT: };
// NON_EXIST_CLASS: void func() {
// NON_EXIST_CLASS-NEXT: A var1;
// NON_EXIST_CLASS-NEXT: A* var2 = new A;
// NON_EXIST_CLASS-NEXT: delete var2;
// NON_EXIST_CLASS-NEXT: }

// RUN: %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename help\
// RUN: 2>&1 | FileCheck %s --check-prefix=HELP

// HELP: Specify three required arguments:
// HELP-NEXT: -plugin-arg-rename type=["variable", "function", "class"]
// HELP-NEXT: -plugin-arg-rename first-name="Current identifier name"
// HELP-NEXT: -plugin-arg-rename second-name="New identifier name"

// RUN: not %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename first-name=B\
// RUN: -plugin-arg-rename second-name=C\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: not %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename first-name=B\
// RUN: -plugin-arg-rename second-name=C\
// RUN: -plugin-arg-rename param=val\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: not %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: -plugin-arg-rename type=undefined\
// RUN: -plugin-arg-rename first-name=B\
// RUN: -plugin-arg-rename second-name=C\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

// RUN: not %clang_cc1 -load %llvmshlibdir/RenamePluginUlyanov%pluginext\
// RUN: -add-plugin rename\
// RUN: 2>&1 | FileCheck %s --check-prefix=ERROR

//ERROR: Invalid arguments
//ERROR-NEXT: Specify "-plugin-arg-rename help" for usage

//--- rename_var.cpp
int func() {
  int a = 2, b = 2, c = 2;
  a = a + b - c;
  a++;
  c = b;
  return a;
}
//--- rename_other_var.cpp
int func() {
  int a = 2, b = 2, c = 2;
  a = a + b - c;
  a++;
  c = b;
  return a;
}
//--- rename_func.cpp
int function(int a) {
    int b = 2;
    return a + b;
}
int function2(){
  int a = function(2) + 2;
  return a;
}
//--- rename_other_func.cpp
int func(int a) {
  return a;
}
void func2() {
  int a = func(2);
  int b = func(a) + func(2);
}
//--- rename_class.cpp
class abc{
 private:
  int a;
 public:
  abc() {}
  abc(int a): a(a) {}
  ~abc();
};
void func() {
  abc a;
  abc* var = new abc(1);
  delete var;
}
//--- rename_delete_var_class.cpp
class abcd {};
void f() {
  abcd* aaa = new abcd();
  delete aaa;
}
//--- rename_tmp_class.cpp
template <typename T> class ab {
  T a;
  ab() {}
  ~ab();
};
//--- rename_inherited_class.cpp
class qwe {
 qwe();
 ~qwe();
};
class rty : public qwe {
 rty();
 ~rty();
};
//--- rename_other_class.cpp
class A{
 private:
  int var1;
 public:
 A() {};
 ~A() {};
};
void func() {
  A var1;
  A* var2 = new A;
  delete var2;
}