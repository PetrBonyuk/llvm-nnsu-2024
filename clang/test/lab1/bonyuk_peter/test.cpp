// RUN: %clang_cc1 -load %llvmshlibdir/depWarningPluginBonyuk%pluginext -plugin deprecated-warning %s 2>&1 | FileCheck %s

// CHECK: warning: The function name contains the word 'deprecated'
void deprecated();

// CHECK: warning: The function name contains the word 'deprecated'
void cfgdeprecatedasad();

// CHECK: warning: The function name contains the word 'deprecated'
void yufdeprecatedasSVDfd();

// CHECK-NOT: warning: The function name contains the word 'deprecated'
void something();

// CHECK-NOT: warning: The function name contains the word 'deprecated'
void deprecatend();

// CHECK-NOT: warning: The function name contains the word 'deprecated'
void deprecate();