// RUN: %clang_cc1 -load %llvmshlibdir/AddAttrAlwaysInlinePlugin%pluginext\
// RUN: -plugin add_attr_always_inline_plugin\
// RUN: -plugin-arg-add_attr_always_inline_plugin --help %s 2>&1 | FileCheck %s --check-prefix=CHECK-HELP
// CHECK-HELP: Adds the always_inline attribute to functions without conditions

// RUN: %clang_cc1 -load %llvmshlibdir/AddAttrAlwaysInlinePlugin%pluginext\
// RUN: -add-plugin add_attr_always_inline_plugin %s\
// RUN: -ast-dump %s -ast-dump-filter test | FileCheck %s

namespace {
// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testSquare 'int \(int\)'}}
// CHECK: `-AlwaysInlineAttr {{.* always_inline}}
int testSquare(int value) { return value * value; }

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testDiff 'int \(int, int\)'}}
// CHECK: `-AlwaysInlineAttr {{.* always_inline}}
int testDiff(int valueOne, int valueTwo) {
  {} {} {{} {}} {} {{{}} {} {}} {} {
    { return valueOne - valueTwo; }
  }
}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testEmptyFunc 'void \(\)'}}
// CHECK: `-AlwaysInlineAttr {{.* always_inline}}
void testEmptyFunc() {
  {} {{}} {{} {} {{{{}}}}} {}
}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testIfStmt 'bool \(int\)'}}
bool testIfStmt(int value) {
  if (value % 2)
    return false;
  return true;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testSwitchStmt 'int \(int\)'}}
int testSwitchStmt(int value) {
  switch (value) {
  case 1:
    return value;
  case 2:
    return value;
  case 3:
    return value;
  default:
    return value;
  }
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testWhileStmt 'void \(int\)'}}
void testWhileStmt(int value) {
  {
    while (value--) {
    }
  }
  {} {}
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testDoStmt 'void \(int\)'}}
void testDoStmt(int value) {
  do {
    --value;
  } while (value);
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+tests\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) testForStmt 'void \(unsigned int\)'}}
void testForStmt(unsigned value) {
  for (unsigned i = 0; i < value; ++i) {
  }
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> Implicit always_inline}}
} // namespace
