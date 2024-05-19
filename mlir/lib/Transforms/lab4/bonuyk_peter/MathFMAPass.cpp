#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;

namespace {
	class FusedMultiplyAddPass : public PassWrapper<FusedMultiplyAddPass, OperationPass<ModuleOp>> {
	public:
		StringRef getArgument() const final { return "bonyuk_fused_multiply_add"; }
		StringRef getDescription() const final {
			return "This Pass combines the operations of addition and multiplication into one";
		}

		void runOnOperation() override {
			ModuleOp module = getOperation();
			module.walk([&](Operation *op) {
				if (auto addOp = dyn_cast<AddFOp>(op)) {
					handleAddOperation(addOp);
				}
				else if (auto mulOp = dyn_cast<MulFOp>(op)) {
					handleMultiplyOperation(mulOp);
				}
			});
		}

	private:
		void handleAddOperation(AddFOp addOp) {
			Value left = addOp.getOperand(0);
			Value right = addOp.getOperand(1);

			if (auto mulOp = dyn_cast<MulFOp>(left.getDefiningOp())) {
				replaceWithFMA(addOp, mulOp, right);
			}
			else if (auto mulOp = dyn_cast<MulFOp>(right.getDefiningOp())) {
				replaceWithFMA(addOp, mulOp, left);
			}
		}

		void handleMultiplyOperation(MulFOp mulOp) {
			if (mulOp.use_empty()) {
				mulOp.erase();
			}
		}

		void replaceWithFMA(AddFOp addOp, MulFOp mulOp, Value operand) {
			OpBuilder builder(addOp);
			Value fmaOp = builder.create<FMAFOp>(addOp.getLoc(), mulOp.getOperand(0),
				mulOp.getOperand(1), operand);
			addOp.replaceAllUsesWith(fmaOp);

			if (mulOp.use_empty()) {
				mulOp.erase();
			}

			if (fmaOp.use_empty()) {
				addOp.erase();
			}
		}
	};
} // namespace

static PassRegistration<FusedMultiplyAddPass> registration("bonyuk_fused_multiply_add", "Combine addition and multiplication operations");