#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {

class MyArithCeilDiv
    : public PassWrapper<MyArithCeilDiv, OperationPass<func::FuncOp>> {
public:
  StringRef getArgument() const final { return "BendArithCeilDiv"; }
  StringRef getDescription() const final {
    return "breaks down arith.ceildivui and arith.ceildivsi";
  }
  template <typename ceilop, typename divop> void combineOp(ceilop op) {
    OpBuilder builder(op);
    auto loc = op.getLoc();
    Value a = op.getOperand(0);
    Value b = op.getOperand(1);
    Value minusOne = builder.create<arith::ConstantIntOp>(loc, -1, b.getType());

    Value aPlusB = builder.create<arith::AddIOp>(loc, a, b);
    Value aPlusBMinusOne = builder.create<arith::AddIOp>(loc, aPlusB, minusOne);

    Value result = builder.create<divop>(loc, aPlusBMinusOne, b);

    op.replaceAllUsesWith(result);
    op.erase();
  }

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (auto ceilDivUIOp = dyn_cast<arith::CeilDivUIOp>(op)) {
        combineOp<arith::CeilDivUIOp, arith::DivUIOp>(ceilDivUIOp);
      } else if (auto ceilDivSIOp = dyn_cast<arith::CeilDivSIOp>(op)) {
        combineOp<arith::CeilDivSIOp, arith::DivSIOp>(ceilDivSIOp);
      }
    });
  }
};
} // anonymous namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(MyArithCeilDiv)
MLIR_DEFINE_EXPLICIT_TYPE_ID(MyArithCeilDiv)

PassPluginLibraryInfo getMyArithCeilDivPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "BendArithCeilDiv", LLVM_VERSION_STRING,
          []() { PassRegistration<MyArithCeilDiv>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getMyArithCeilDivPluginInfo();
}
