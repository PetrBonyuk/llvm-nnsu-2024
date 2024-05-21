#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {
	class BonyukFusedMultiplyAddPass
		: public PassWrapper<BonyukFusedMultiplyAddPass, OperationPass> {
	public:
		StringRef getArgument() const final { return "bonyuk_fused_multiply_add"; }
		StringRef getDescription() const final {
			return "This Pass combines the operations of addition and multiplication into one";
		}

		void runOnOperation() override {
			ModuleOp module = getOperation();
			module.walk([&](Operation *operation) {
				if (auto addOp = dyn_cast<LLVM::FAddOp>(operation)) {
					Value addLeft = addOp.getOperand(0);
					Value addRight = addOp.getOperand(1);

					if (auto multiplyLeft = addLeft.getDefiningOp<LLVM::FMulOp>()) {
						handleMultiplyOperation(addOp, multiplyLeft, addRight);
					}
					else if (auto multiplyRight = addRight.getDefiningOp<LLVM::FMulOp>()) {
						handleMultiplyOperation(addOp, multiplyRight, addLeft);
					}
				}
			});

			module.walk([](LLVM::FMulOp mulOp) {
				if (mulOp.use_empty()) {
					mulOp.erase();
				}
			});
		}

	private:
		void handleMultiplyOperation(LLVM::FAddOp addOp, LLVM::FMulOp multiplyOp, Value operand) {
			OpBuilder builder(addOp);
			Value fmaOp = builder.create<LLVM::FMAOp>(addOp.getLoc(), multiplyOp.getOperand(0),
				multiplyOp.getOperand(1), operand);
			addOp.replaceAllUsesWith(fmaOp);

			if (multiplyOp.use_empty()) {
				multiplyOp.erase();
			}

			if (addOp.use_empty()) {
				addOp.erase();
			}
		}
	};
} // namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(BonyukFusedMultiplyAddPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(BonyukFusedMultiplyAddPass)

PassPluginLibraryInfo getFusedMultiplyAddPassPluginInfo() {
	return { MLIR_PLUGIN_API_VERSION, "bonyuk_fused_multiply_add", LLVM_VERSION_STRING,
			{createBonyukFusedMultiplyAddPass()} };
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
	return getFusedMultiplyAddPassPluginInfo();
}