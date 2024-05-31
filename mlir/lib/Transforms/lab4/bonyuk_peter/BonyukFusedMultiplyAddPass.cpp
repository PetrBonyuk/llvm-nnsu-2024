#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
class BonyukFusedMultiplyAddPass
    : public PassWrapper<BonyukFusedMultiplyAddPass,
                         OperationPass<LLVM::LLVMFuncOp>> {
public:
  StringRef getArgument() const final { return "bonyuk_fused_multiply_add"; }
  StringRef getDescription() const final {
    return "This Pass combines the operations of addition and multiplication "
           "into one";
  }

  void runOnOperation() override {
    LLVM::LLVMFuncOp func = getOperation();
    func.walk([&](LLVM::FAddOp AddOperation) {
      Value AddLeft = AddOperation.getOperand(0);
      Value AddRight = AddOperation.getOperand(1);

      if (auto MultiplyLeft = AddLeft.getDefiningOp<LLVM::FMulOp>()) {
        HandleMultiplyOperation(AddOperation, MultiplyLeft, AddRight);
      } else if (auto MultiplyRight = AddRight.getDefiningOp<LLVM::FMulOp>()) {
        HandleMultiplyOperation(AddOperation, MultiplyRight, AddLeft);
      }
    });

    func.walk([](LLVM::FMulOp MultiplyOperation) {
      if (MultiplyOperation.use_empty()) {
        MultiplyOperation.erase();
      }
    });
  }

private:
  void HandleMultiplyOperation(LLVM::FAddOp &AddOperation,
                               LLVM::FMulOp &MultiplyOperation,
                               Value &Operand) {
    OpBuilder builder(AddOperation);
    Value FMAOperation = builder.create<LLVM::FMAOp>(
        AddOperation.getLoc(), MultiplyOperation.getOperand(0),
        MultiplyOperation.getOperand(1), Operand);
    AddOperation.replaceAllUsesWith(FMAOperation);

    if (MultiplyOperation.use_empty()) {
      MultiplyOperation.erase();
    }

    if (AddOperation.use_empty()) {
      AddOperation.erase();
    }
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(BonyukFusedMultiplyAddPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(BonyukFusedMultiplyAddPass)

PassPluginLibraryInfo getFusedMultiplyAddPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "bonyuk_fused_multiply_add",
          LLVM_VERSION_STRING,
          []() { PassRegistration<BonyukFusedMultiplyAddPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getFusedMultiplyAddPassPluginInfo();
}
