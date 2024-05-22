#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class FunctionInvocationCounterPass
    : public PassWrapper<FunctionInvocationCounterPass,
                         OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "VeselovFuncCallCnt"; }
  StringRef getDescription() const final {
    return "Counts the number of calls to each function in the module";
  }

  void runOnOperation() override {
    std::vector<LLVM::LLVMFuncOp> funcs;
    std::map<StringRef, int> callCounter;

    getOperation()->walk([&](Operation *op) {
      if (auto func = dyn_cast<LLVM::LLVMFuncOp>(op)) {
        funcs.push_back(func);
      }

      if (auto call = dyn_cast<LLVM::CallOp>(op)) {
        StringRef funcName = call.getCallee().value();
        callCounter[funcName]++;
      }
    });

    StringRef attrName = "numCalls";
    for (auto &func : funcs) {
      int numCalls = callCounter[func.getName()];
      auto attrValue =
          IntegerAttr::get(IntegerType::get(func.getContext(), 32), numCalls);
      func->setAttr(attrName, attrValue);
    }
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(FunctionInvocationCounterPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(FunctionInvocationCounterPass)

PassPluginLibraryInfo getFunctionCallCounterPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "VeselovFuncCallCnt", LLVM_VERSION_STRING,
          []() { PassRegistration<FunctionInvocationCounterPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getFunctionCallCounterPassPluginInfo();
}
