#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Tools/Plugins/PassPlugin.h"

using namespace mlir;

namespace {

enum class CeilDivType { Signed, Unsigned };

class CeilDivPass
    : public PassWrapper<CeilDivPass, OperationPass<LLVM::LLVMFuncOp>> {
public:
  StringRef getArgument() const final { return "pushkarev_ceildiv"; }
  StringRef getDescription() const final {
    return "breaks arith.ceildivui and arith.ceildivsi operations into arith. "
           "calculations";
  }

  void runOnOperation() override {
    getOperation()->walk([&](Operation *op) {
      if (auto ceilDivUI = dyn_cast<arith::CeilDivUIOp>(op)) {
        transformCeilDiv(ceilDivUI, CeilDivType::Unsigned);
      } else if (auto ceilDivSI = dyn_cast<arith::CeilDivSIOp>(op)) {
        transformCeilDiv(ceilDivSI, CeilDivType::Signed);
      }
    });
  }

private:
  template <typename CeilDivOp>
  void transformCeilDiv(CeilDivOp op, CeilDivType divType) {
    OpBuilder builder(op);
    Location location = op.getLoc();
    Value numerator = op.getLhs();
    Value denominator = op.getRhs();

    Value one =
        builder.create<arith::ConstantIntOp>(location, 1, builder.getI32Type());
    Value sum = builder.create<arith::AddIOp>(location, numerator, denominator);
    Value adjustedSum = builder.create<arith::SubIOp>(location, sum, one);
    Value quotient;

    if (divType == CeilDivType::Signed) {
      quotient =
          builder.create<arith::DivSIOp>(location, adjustedSum, denominator);
    } else {
      quotient =
          builder.create<arith::DivUIOp>(location, adjustedSum, denominator);
    }

    op.replaceAllUsesWith(quotient);
    op.erase();
  }
};

} // anonymous namespace

MLIR_DECLARE_EXPLICIT_TYPE_ID(CeilDivPass)
MLIR_DEFINE_EXPLICIT_TYPE_ID(CeilDivPass)

PassPluginLibraryInfo getCeilDivPassPluginInfo() {
  return {MLIR_PLUGIN_API_VERSION, "pushkarev_ceildiv", LLVM_VERSION_STRING,
          []() { PassRegistration<CeilDivPass>(); }};
}

extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo mlirGetPassPluginInfo() {
  return getCeilDivPassPluginInfo();
}
