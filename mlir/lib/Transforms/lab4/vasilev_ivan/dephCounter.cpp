#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

#include "mlir/IR/BuiltinOps.h"

using namespace mlir;

namespace {

class DepthCounter
    : public PassWrapper<DepthCounter, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const final { return "VasilevDepthCounter"; }
  StringRef getDescription() const final {
    return "Count the maximum depth of region nests in a function";
  }

  void runOnOperation() override {

    getOperation()->walk([&](Operation *op) {
      int maxDepth = computeMaxDepth(op);

      op->setAttr(
          "max_region_depth",
          IntegerAttr::get(IntegerType::get(op->getContext(), 32), maxDepth));
    });
  }

private:
  int computeMaxDepth(Operation *op, int currentDepth = 0) {
    int maxDepth = currentDepth;

    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        maxDepth = std::max(
            maxDepth, computeMaxDepth(block.getTerminator(), currentDepth + 1));
      }
    }

    return maxDepth;
  }
};
} // anonymous namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(DepthCounter)
MLIR_DEFINE_EXPLICIT_TYPE_ID(DepthCounter)

PassPluginLibraryInfo getDepthCounterPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "VasilevDepthCounter", LLVM_VERSION_STRING,
          []() { PassRegistration<DepthCounter>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getDepthCounterPluginInfo();
}
