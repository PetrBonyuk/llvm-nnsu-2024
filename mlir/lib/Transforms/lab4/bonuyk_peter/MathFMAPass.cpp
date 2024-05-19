#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Utils.h"

using namespace mlir;

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
			OwningRewritePatternList patterns(&getContext());
			patterns.insert<FusedMultiplyAddPattern>(&getContext());
			(void)applyPatternsAndFoldGreedily(module, std::move(patterns));
		}
	};

	class FusedMultiplyAddPattern : public OpRewritePattern<AddFOp> {
	public:
		using OpRewritePattern<AddFOp>::OpRewritePattern;

		LogicalResult matchAndRewrite(AddFOp addOp,
			PatternRewriter &rewriter) const override {
			Value left = addOp.getOperand(0), right = addOp.getOperand(1);
			Value lmul, rmul;

			if (matchPattern(left, m_Op<MulFOp>(m_Value(lmul), m_Value()))) {
				replaceFMA(addOp, rewriter, lmul, right);
			}
			else if (matchPattern(right, m_Op<MulFOp>(m_Value(rmul), m_Value()))) {
				replaceFMA(addOp, rewriter, rmul, left);
			}
			else {
				return failure();
			}
			return success();
		}

	private:
		void replaceFMA(AddFOp addOp, PatternRewriter &rewriter, Value mulOp,
			Value otherOperand) const {
			Value fmaOp = rewriter.create<FMAFOp>(addOp.getLoc(), mulOp, otherOperand,
				addOp.getResult());
			rewriter.replaceOp(addOp, fmaOp);
			if (mulOp.use_empty())
				rewriter.eraseOp(mulOp);
		}
	};
} // namespace

void populateFusedMultiplyAddPatterns(RewritePatternSet &patterns) {
	patterns.add<FusedMultiplyAddPattern>(&getContext());
}

void registerPasses() {
	PassRegistration<FusedMultiplyAddPass>();
	RewritePatternSet patterns(&getContext());
	populateFusedMultiplyAddPatterns(patterns);
	(void)applyPatternsAndFoldGreedily(getOperation(), std::move(patterns));
}

} // namespace mlir

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
	return { MLIR_PLUGIN_API_VERSION, "bonyuk_fused_multiply_add", LLVM_VERSION_STRING,
			registerPasses };
}