// RUN: %clang_cc1 -load %llvmshlibdir/depWarningPluginBonyuk%pluginext -plugin deprecated-warning %s 2>&1 | FileCheck %s

// CHECK: warning: The 'deprecated' is in the function name
void deprecated();

// CHECK: warning: The 'deprecated' is in the function name
void deprecatedasad();

// CHECK: warning: The 'deprecated' is in the function name
void deprecatedasSVDfd();

// CHECK-NOT: warning: The 'deprecated' is in the function name
void something();

// CHECK-NOT: warning: The 'deprecated' is in the function name
void deprecatend();

// CHECK-NOT: warning: The 'deprecated' is in the function name
void deprecateqwERe();
