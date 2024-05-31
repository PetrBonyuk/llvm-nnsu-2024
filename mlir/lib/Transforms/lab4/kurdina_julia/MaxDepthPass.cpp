#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class MaxDepthPass
    : public PassWrapper<MaxDepthPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const final { return "KurdinaMaxDepth"; }
  StringRef getDescription() const final {
    return "Counts the max depth of region nests in the function";
  }

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      int maxDepth = getMaxDepth(op);
      op->setAttr(
          "maxDepth",
          IntegerAttr::get(IntegerType::get(op->getContext(), 32), maxDepth));
    });
  }

private:
  int getMaxDepth(Operation *funcOp) {
    int maxDepth = 1;
    Operation *curOp;
    funcOp->walk([&](Operation *op) {
      curOp = op;
      int depth = -1;
      while (curOp) {
        if (curOp->getParentOp()) {
          depth++;
        }
        curOp = curOp->getParentOp();
      }
      maxDepth = std::max(maxDepth, depth);
    });
    return maxDepth;
  }
};
} // anonymous namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(MaxDepthPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(MaxDepthPass)

PassPluginLibraryInfo getMaxDepthPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "KurdinaMaxDepth", LLVM_VERSION_STRING,
          []() { PassRegistration<MaxDepthPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getMaxDepthPassPluginInfo();
}