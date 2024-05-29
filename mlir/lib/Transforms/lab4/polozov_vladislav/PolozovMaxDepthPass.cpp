#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include <map>
using namespace mlir;

namespace {
class PolozovMaxDepthPass
    : public PassWrapper<PolozovMaxDepthPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const final { return "max-depth-pass"; }
  StringRef getDescription() const final {
    return "Counts the max depth of region nests in a function.";
  }
  std::map<Operation *, int> Depth;
  void runOnOperation() override {
    Operation *func = getOperation();
    Depth[func] = 1;
    int maxDepth = 1;
    func->walk([&](Operation *cur) {
      maxDepth = std::max(maxDepth, getMaxDepth(cur));
    });
    func->setAttr(
        "maxDepth",
        IntegerAttr::get(IntegerType::get(func->getContext(), 32), maxDepth));
  }

private:
  int getMaxDepth(Operation *op) {
    if (Depth.count(op)) {
      return Depth[op];
    }
    int &answer = Depth[op];
    answer = (op->getNumRegions() > 0);
    if (op->getParentOp() != nullptr) {
      answer = std::max(answer, (op->getNumRegions() > 0) +
                                    getMaxDepth(op->getParentOp()));
    }
    return answer;
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(PolozovMaxDepthPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(PolozovMaxDepthPass)

PassPluginLibraryInfo getPolozovMaxDepthPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "max-depth-pass", LLVM_VERSION_STRING,
          []() { PassRegistration<PolozovMaxDepthPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getPolozovMaxDepthPassPluginInfo();
}