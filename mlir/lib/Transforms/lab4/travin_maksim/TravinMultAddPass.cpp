#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class TravinMultAddPass
    : public PassWrapper<TravinMultAddPass, OperationPass<LLVM::LLVMFuncOp>> {
public:
  StringRef getArgument() const final { return "merge-mult-add"; }
  StringRef getDescription() const final {
    return "Merges multiplication and addition into a single math.fma";
  }

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();

    func.walk([&](LLVM::FAddOp addOp) {
      Value left = addOp.getOperand(0);
      Value right = addOp.getOperand(1);
      if (tryMergeAddMul(addOp, left, right) ||
          tryMergeAddMul(addOp, right, left)) {
        addOp.erase();
      }
    });

    func.walk([&](LLVM::FMulOp mulOp) {
      if (mulOp.use_empty()) {
        mulOp.erase();
      }
    });
  }

private:
  bool tryMergeAddMul(LLVM::FAddOp addOp, Value lhs, Value rhs) {
    if (auto mulOp = lhs.getDefiningOp<LLVM::FMulOp>()) {
      OpBuilder builder(addOp);
      Value fma = builder.create<LLVM::FMAOp>(
          addOp.getLoc(), mulOp.getOperand(0), mulOp.getOperand(1), rhs);
      addOp.replaceAllUsesWith(fma);

      if (mulOp.use_empty()) {
        mulOp.erase();
      }
      return true;
    }
    return false;
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(TravinMultAddPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(TravinMultAddPass)

PassPluginLibraryInfo getTravinMultAddPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "merge-mult-add", LLVM_VERSION_STRING,
          []() { PassRegistration<TravinMultAddPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getTravinMultAddPassPluginInfo();
}