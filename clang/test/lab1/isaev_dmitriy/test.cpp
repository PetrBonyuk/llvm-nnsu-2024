// RUN: %clang_cc1 -ast-dump -ast-dump-filter foo -load %llvmshlibdir/IsaevAlwaysInlinePlugin%pluginext -add-plugin add-always-inline %s | FileCheck %s

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) foo1 'int \(\)'}}
int foo1() {
  bool f = false;
  if (f) {
    f = true;
  }
  return 1;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) foo2 'int \(\)'}}
int foo2() {
  for (int i = 0; i < 3; i++) {
  }
  return 1;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) foo3 'int \(\)'}}
int foo3() {
  while (false) {
    if (true) {
      int d = 1;
      if (d) {
        return d;
      }
    }
  }
  return 1;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) foo4 'int \(int, int\)'}}
int foo4(int a, int b) {
  int c;
  c = a - b;
  return c;
}
// CHECK: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) foo5 'int \(\)'}}
int foo5() {
  do {

  } while (false);
  return 1;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) foo6 'int \(\)'}}
int foo6() {
  int a = 2;
  switch (a) {
  case 2:
    break;
  default:
    break;
  }
  return a;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) foo7 'int \(\)'}}
int foo7() {
  while (false) {
  }
  return 1;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) foo8 'int \(\)'}}
int foo8() { return 1; }
// CHECK: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) foo9 'int \(\)'}}
int foo9() {
  int c;
  {
    {
      while (false) {
        ;
      }
    }
  }
  return 1;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) foo10 'int \(\)'}}
int foo10() {
  int c;
  {
    while (false) {
      ;
    }
  }
  return 1;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}