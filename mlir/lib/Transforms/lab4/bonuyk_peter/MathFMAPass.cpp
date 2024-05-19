#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"

using namespace mlir;
using namespace mlir::StandardOps;

namespace {
class FusedMultiplyAddPass
    : public PassWrapper<FusedMultiplyAddPass, OperationPass<ModuleOp>> {
public:
  StringRef getArgument() const final { return "bonyuk_fused_multiply_add"; }
  StringRef getDescription() const final {
    return "This Pass combines the operations of addition and multiplication into one";
  }

  void runOnOperation() override {
	  ModuleOp module = getOperation();
	  module.walk([&](Operation *operation) {
		  if (auto addOperation = dyn_cast<AddFOp>(operation)) {
			  Value addLeft = addOperation.getOperand(0);
			  Value addRight = addOperation.getOperand(1);

			  if (auto multiplyLeft = addLeft.getDefiningOp<MulFOp>()) {
				  handleMultiplyOperation(addOperation, multiplyLeft, addRight);
			  }
			  else if (auto multiplyRight = addRight.getDefiningOp<MulFOp>()) {
				  handleMultiplyOperation(addOperation, multiplyRight, addLeft);
			  }
		  }
	  });

	  module.walk([&](Operation *operation) {
		  if (auto multiplyOperation = dyn_cast<MulFOp>(operation)) {
			  if (multiplyOperation.use_empty()) {
				  multiplyOperation.erase();
			  }
		  }
	  });
  }

private:
  void HandMultiplyOperation(LLVM::FAddOp &AddOperation, LLVM::FMulOp &MultiplyOperation,
                             Value &Operand) {
    OpBuilder builder(AddOperation);
    Value FMAOperation = builder.create<LLVM::FMAOp>(AddOperation.getLoc(), MultiplyOperation.getOperand(0),
      MultiplyOperation.getOperand(1), Operand);
    AddOperation.replaceAllUsesWith(FMAOperation);

    if (MultiplyOperation.use_empty()) {
      MultiplyOperation.erase();
    }

    if (FMAOperation.use_empty()) {
      AddOperation.erase();
    }
  }
};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(FusedMultiplyAddPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(FusedMultiplyAddPass)

PassPluginLibraryInfo getFusedMultiplyAddPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "bonyuk_fused_multiply_add", LLVM_VERSION_STRING,
          []() { PassRegistration<FusedMultiplyAddPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getFusedMultiplyAddPassPluginInfo();
}