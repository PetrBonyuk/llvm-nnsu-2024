// RUN: %clang_cc1 -load %llvmshlibdir/depWarningPluginBonyuk%pluginext -plugin deprecated-warning -plugin-arg-deprecated-warning help %s 2>&1 | FileCheck %s --check-prefix=CHECK-HELP
// RUN: %clang_cc1 -load %llvmshlibdir/depWarningPluginBonyuk%pluginext -plugin deprecated-warning -plugin-arg-deprecated-warning case-insensitive %s 2>&1 | FileCheck %s --check-prefix=CHECK-CASE-INSENSITIVE

// CHECK-HELP: DeprecFuncPlugin: Checks for deprecated functions in the code.
// CHECK-HELP-NOT: The function name contains the word 'deprecated'
// CHECK-CASE-INSENSITIVE: warning: The function name contains the word 'deprecated'
void DeprecatedFunction();

// CHECK-CASE-INSENSITIVE: warning: The function name contains the word 'deprecated'
void DEPRECATEDFUNCTION();

// CHECK-NOT: warning: The function name contains the word 'deprecated'
void something();

// CHECK-NOT: warning: The function name contains the word 'deprecated'
void deprecatend();

// CHECK-NOT: warning: The function name contains the word 'deprecated'
void deprecate();

// RUN: %clang_cc1 -load %llvmshlibdir/depWarningPluginBonyuk%pluginext -plugin deprecated-warning %s 2>&1 | FileCheck %s --check-prefix=CHECK-CASE-SENSITIVE

// CHECK-CASE-SENSITIVE: warning: The function name contains the word 'deprecated'
void afcdeprecatedasad();

// CHECK-CASE-SENSITIVE-NOT: warning: The function name contains the word 'deprecated'
void yufDeprecatedasSVDfd();