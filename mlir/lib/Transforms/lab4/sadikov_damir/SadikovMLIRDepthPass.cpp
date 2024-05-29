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
class SadikovMLIRDepthPass
    : public PassWrapper<SadikovMLIRDepthPass, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const final { return "sadikov-mlir-depth-pass"; }
  StringRef getDescription() const final {
    return "sadikov-mlir-depth-pass : pass that counts the max depth of region "
           "nests in the function";
  }
  void runOnOperation() override {
    Operation *func = getOperation();
    func->setAttr("depth_of_func",
                  IntegerAttr::get(IntegerType::get(func->getContext(), 32),
                                   getDepth(func)));
  }

private:
  std::map<Operation *, int> Depth;
  int getDepth(Operation *op) {
    if (auto it = Depth.find(op); it != Depth.end()) {
      return it->second;
    }
    int depth = 0;
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        for (Operation &operation : block) {
          depth = std::max(depth, getDepth(&operation) + 1);
        }
      }
    }
    return Depth[op] = depth;
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(SadikovMLIRDepthPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(SadikovMLIRDepthPass)

PassPluginLibraryInfo getSadikovMLIRDepthPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "sadikov-mlir-depth-pass",
          LLVM_VERSION_STRING,
          []() { PassRegistration<SadikovMLIRDepthPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getSadikovMLIRDepthPassPluginInfo();
}
