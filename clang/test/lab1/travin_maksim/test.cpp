// RUN: %clang_cc1 -ast-dump -ast-dump-filter test -load %llvmshlibdir/TravinAlwaysInlinePlugin%pluginext -add-plugin add-always-inline %s | FileCheck %s

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]+), line:([0-9]+:[0-9]+)> line:([0-9]+:[0-9]+) test0 'void \(int, int\)'}}
void __attribute__((always_inline)) test0(int a, int b) {
    a = b;
}
// CHECK: AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <line:4:21> always_inline}}
// CHECK-NOT: `-AlwaysInlineAttr {{.* always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) test1 'void \(\)'}}
void test1();
// CHECK: `-AlwaysInlineAttr {{.* always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) test2 'int \(int\)'}}
int test2(int a) {
    return a;
}
// CHECK: `-AlwaysInlineAttr {{.* always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) test3 'int \(int, int\)'}}
int test3(int a, int b) {
	if (a > b) return 1;
	return 0;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) test4 'int \(int, int\)'}}
int test4(int a, int b) {
	for (; a < b;) {}
	return a;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}

// CHECK: FunctionDecl {{0[xX][0-9a-fA-F]+ <.+test\.cpp:([0-9]+:[0-9]|[0-9]+), (line|col):([0-9]+:[0-9]|[0-9]+)> (line|col):([0-9]+:[0-9]|[0-9]+) test5 'int \(int, int\)'}}
int test5(int a, int b) {
	switch (a) {
		case 1:
			return a;
		case 2:
			return b;
	}
	return 0;
}
// CHECK-NOT: `-AlwaysInlineAttr {{0[xX][0-9a-fA-F]+ <(line|col):([0-9]+:[0-9]|[0-9]+)> always_inline}}