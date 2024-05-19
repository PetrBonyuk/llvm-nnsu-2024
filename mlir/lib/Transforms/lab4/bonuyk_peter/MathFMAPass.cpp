#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

namespace {
class FusedMultiplyAddPass
    : public PassWrapper<FusedMultiplyAddPass, FunctionPass> {
public:
  static constexpr TypeID PassID = MLIR_TYPEID_EXPLICIT(FusedMultiplyAddPass);

  StringRef getArgument() const final { return "bonyuk_fused_multiply_add"; }
  StringRef getDescription() const final {
    return "Merge multiplication and addition into math.fma";
  }

  void runOnFunction() override {
    ConversionTarget target(getContext());
    target.addLegalOp<ModuleOp, FuncOp, ReturnOp, ConstantOp, CmpFOp, CmpIOp,
                      AddFOp, MulFOp>();

    OwningRewritePatternList patterns;
    patterns.insert<FusedMultiplyAddPattern>(&getContext());

    if (failed(applyPartialConversion(getFunction(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

class FusedMultiplyAddPattern : public OpRewritePattern<AddFOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(AddFOp addOp,
                                PatternRewriter &rewriter) const override {
    Value mulOpA, mulOpB;
    if (!matchPattern(addOp.getOperand(0), m_MulFOp(mulOpA, mulOpB)) &&
        !matchPattern(addOp.getOperand(1), m_MulFOp(mulOpA, mulOpB))) {
      return failure();
    }

    auto fmaOp = rewriter.create<math::FMAOp>(addOp.getLoc(), mulOpA, mulOpB,
                                               addOp.getResult());
    rewriter.replaceOp(addOp, fmaOp.getResult());

    return success();
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