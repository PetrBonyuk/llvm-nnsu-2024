// RUN: %clang_cc1 -load %llvmshlibdir/AlwaysInlinePluginAkopyan%pluginext\
// RUN: -add-plugin always-inline %s\
// RUN: -ast-dump %s -ast-dump-filter test | FileCheck %s

namespace {
// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testMultiply 'int \(int, int\)'}}
// CHECK: `-AlwaysInlineAttr {{.* always_inline}}
int testMultiply(int valueOne, int valueTwo) { return valueOne * valueTwo; }

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testSum 'int \(int, int\)'}}
// CHECK: `-AlwaysInlineAttr {{.* always_inline}}
int testSum(int valueOne, int valueTwo) {
  {} {} {{} {}} {} {{{}} {} {}} {} {
    { return valueOne + valueTwo; }
  }
}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testEmptyFunc 'void \(\)'}}
// CHECK: `-AlwaysInlineAttr {{.* always_inline}}
void testEmptyFunc() {
  {} {{}} {{} {} {{{{}}}}} {}
}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testIfStmt 'bool \(int, int\)'}}
bool testIfStmt(int a, int b) {
  if (a < b)
    return true;
  return false;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testLoop 'void \(int\)'}}
void testLoop(int value) {
  for (int i = 0; i < value; ++i) {
    // do nothing
  }
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}
} // namespace

// RUN: %clang_cc1 -load %llvmshlibdir/AlwaysInlinePluginAkopyan%pluginext\
// RUN: -plugin always-inline\
// RUN: -plugin-arg-always-inline --help %s 2>&1 | FileCheck %s --check-prefix=CHECK-HELP
// CHECK-HELP: Applies the always_inline attribute to functions that don't contain conditional statements