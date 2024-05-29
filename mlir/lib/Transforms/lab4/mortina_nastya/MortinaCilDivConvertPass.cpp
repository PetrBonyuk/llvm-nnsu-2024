#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {

class MortinaCilDivConvertPass
    : public PassWrapper<MortinaCilDivConvertPass,
                         OperationPass<LLVM::LLVMFuncOp>> {
public:
  StringRef getArgument() const final { return "mortina-cildiv-conv"; }
  StringRef getDescription() const final {
    return "Pass that breaks arith.ceildivui and arith.ceildivsi operations "
           "into calculation following the rule: ceildiv(a, b) = (a + b - 1) / "
           "b";
  }

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();

    func.walk([&](Operation *op) {
      if (auto ceilDivUI = dyn_cast<arith::CeilDivUIOp>(op)) {
        breaksCeilDiv(ceilDivUI, arith::DivUIOp::getOperationName());
      } else if (auto ceilDivSI = dyn_cast<arith::CeilDivSIOp>(op)) {
        breaksCeilDiv(ceilDivSI, arith::DivSIOp::getOperationName());
      }
    });
  }

private:
  void breaksCeilDiv(Operation *op, StringRef divOpName) {
    OpBuilder builder(op);
    Value a = op->getOperand(0);
    Value b = op->getOperand(1);
    Value one = builder.create<arith::ConstantIntOp>(op->getLoc(), 1,
                                                     builder.getI32Type());
    Value firstAdd = builder.create<arith::AddIOp>(op->getLoc(), a, b);
    Value secondSub =
        builder.create<arith::SubIOp>(op->getLoc(), firstAdd, one);
    Value thirdDiv;
    if (divOpName == arith::DivUIOp::getOperationName()) {
      thirdDiv = builder.create<arith::DivUIOp>(op->getLoc(), secondSub, b);
    } else {
      thirdDiv = builder.create<arith::DivSIOp>(op->getLoc(), secondSub, b);
    }

    op->replaceAllUsesWith(ValueRange(thirdDiv));
    op->erase();
  }
};
} // anonymous namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(MortinaCilDivConvertPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(MortinaCilDivConvertPass)

PassPluginLibraryInfo getMortinaCilDivConvertPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "mortina-cildiv-conv", LLVM_VERSION_STRING,
          []() { PassRegistration<MortinaCilDivConvertPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getMortinaCilDivConvertPassPluginInfo();
}
