#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class CallCounterPass
    : public PassWrapper<CallCounterPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "KruglovFunctionCallCntPass"; }
  StringRef getDescription() const final {
    return "Counts the number of calls to each function in the module";
  }

  void runOnOperation() override {
    llvm::DenseMap<StringRef, int> callCounter;

    getOperation()->walk([&](LLVM::CallOp callOp) {
      if (auto callee = callOp.getCallee()) {
        callCounter[callee.value()]++;
      }
    });

    getOperation()->walk([&](LLVM::LLVMFuncOp funcOp) {
      StringRef funcName = funcOp.getName();
      int callCount = callCounter.lookup(funcName);
      funcOp->setAttr(
          "CallCount",
          IntegerAttr::get(IntegerType::get(funcOp.getContext(), 32),
                           callCount));
    });
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(CallCounterPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(CallCounterPass)

PassPluginLibraryInfo getCallCounterPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "KruglovFunctionCallCntPass",
          LLVM_VERSION_STRING, []() { PassRegistration<CallCounterPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getCallCounterPassPluginInfo();
}
