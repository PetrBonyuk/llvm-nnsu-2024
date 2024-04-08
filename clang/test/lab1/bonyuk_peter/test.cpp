// RUN: %clang_cc1 -load %llvmshlibdir/depWarningPluginBonyuk%pluginext -plugin deprecated-warning %s 2>&1 | FileCheck %s

// CHECK: warning: Function contains 'deprecated' in its name
void deprecated();

// CHECK: warning: Function contains 'deprecated' in its name
void cfgdeprecatedasad();

// CHECK: warning: Function contains 'deprecated' in its name
void yufdeprecatedasSVDfd();

// CHECK-NOT: warning: Function contains 'deprecated' in its name
void something();

// CHECK-NOT: warning: Function contains 'deprecated' in its name
void deprecatend();

// CHECK-NOT: warning: Function contains 'deprecated' in its name
void deprecate();