#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace {
class KosarevEgorFMAPass
    : public PassWrapper<KosarevEgorFMAPass, OperationPass<LLVM::LLVMFuncOp>> {
public:
  StringRef getArgument() const final { return "KosarevEgorFMAPass"; }
  StringRef getDescription() const final { return "fma pass"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    LLVM::LLVMFuncOp function = getOperation();
    mlir::OpBuilder builder(function);

    auto replaceAndEraseOp = [&](mlir::LLVM::FMulOp &mulOp,
                                 mlir::LLVM::FAddOp &addOp,
                                 mlir::Value &thirdOperand) -> void {
      builder.setInsertionPoint(addOp);
      mlir::Value fmaResult =
          builder.create<mlir::math::FmaOp>(addOp.getLoc(), mulOp.getOperand(0),
                                            mulOp.getOperand(1), thirdOperand);
      addOp.replaceAllUsesWith(fmaResult);
      addOp.erase();
      mulOp.erase();
    };

    function.walk([&](mlir::LLVM::FAddOp addOp) {
      mlir::Value addLhs = addOp.getOperand(0);
      mlir::Value addRhs = addOp.getOperand(1);

      if (auto mulOp = addLhs.getDefiningOp<mlir::LLVM::FMulOp>()) {
        if (mulOp->hasOneUse()) {
          replaceAndEraseOp(mulOp, addOp, addRhs);
        }
      }

      else if (auto mulOp = addRhs.getDefiningOp<mlir::LLVM::FMulOp>()) {
        if (mulOp->hasOneUse()) {
          replaceAndEraseOp(mulOp, addOp, addLhs);
        }
      }
    });
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(KosarevEgorFMAPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(KosarevEgorFMAPass)

mlir::PassPluginLibraryInfo getFunctionCallCounterPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "KosarevEgorFMAPass", "0.1",
          []() { mlir::PassRegistration<KosarevEgorFMAPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK mlir::PassPluginLibraryInfo
mlirGetPassPluginInfo() {
  return getFunctionCallCounterPassPluginInfo();
}
