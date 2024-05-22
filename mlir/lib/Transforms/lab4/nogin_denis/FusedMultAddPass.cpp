#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class FusedMultAddPass
    : public PassWrapper<FusedMultAddPass, OperationPass<LLVM::LLVMFuncOp>> {
public:
  StringRef getArgument() const final { return "fused-mult-add"; }
  StringRef getDescription() const final {
    return "Fuses multiply and add operations into a single fma operation";
  }

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();

    func.walk([&](LLVM::FAddOp addOp) {
      Value addLHS = addOp.getOperand(0);
      Value addRHS = addOp.getOperand(1);

      auto tryFuse = [&](Value mulOperand, Value otherOperand) {
        if (auto mulOp = mulOperand.getDefiningOp<LLVM::FMulOp>()) {
          OpBuilder builder(addOp);
          Value fma =
              builder.create<LLVM::FMAOp>(addOp.getLoc(), mulOp.getOperand(0),
                                          mulOp.getOperand(1), otherOperand);
          addOp.replaceAllUsesWith(fma);
          return true;
        }
        return false;
      };

      if (tryFuse(addLHS, addRHS) || tryFuse(addRHS, addLHS)) {
        addOp.erase();
      }
    });
    func.walk([&](LLVM::FMulOp mulOp) {
      if (mulOp.use_empty()) {
        mulOp.erase();
      }
    });
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(FusedMultAddPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(FusedMultAddPass)

PassPluginLibraryInfo getFusedMultAddPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "fused-mult-add", LLVM_VERSION_STRING,
          []() { PassRegistration<FusedMultAddPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getFusedMultAddPassPluginInfo();
}
