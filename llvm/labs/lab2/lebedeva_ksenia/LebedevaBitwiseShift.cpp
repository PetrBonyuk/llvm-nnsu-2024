#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <optional>

using namespace llvm;

class BitwiseShift : public PassInfoMixin<BitwiseShift> {
public:
  PreservedAnalyses run(llvm::Function &func,
                        llvm::FunctionAnalysisManager &aManager) {
    bool changed = false;
    for (auto &basicBlock : func) {
      for (auto InstIt = basicBlock.begin(); InstIt != basicBlock.end();) {
        Instruction &instr = *InstIt++;
        if (instr.getOpcode() != Instruction::Mul) {
          continue;
        }
        auto *Op = dyn_cast<BinaryOperator>(&instr);
        if (!Op) {
          continue;
        }

        Value *leftVal = Op->getOperand(0);
        Value *rightVal = Op->getOperand(1);
        auto leftLog = getLog2(leftVal);
        auto rightLog = getLog2(rightVal);

        if (rightLog < leftLog) {
          std::swap(leftLog, rightLog);
          std::swap(leftVal, rightVal);
        }
        if (rightLog >= 0) {
          IRBuilder<> Builder(Op);
          Value *NewVal = Builder.CreateShl(
              leftVal, ConstantInt::get(Op->getType(), *rightLog));
          instr.replaceAllUsesWith(NewVal);
          InstIt = instr.eraseFromParent();
          changed = true;
        }
      }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }

private:
  std::optional<int> getLog2(llvm::Value *Op) {
    if (auto *CI = dyn_cast<ConstantInt>(Op)) {
      return CI->getValue().exactLogBase2();
    }
    return std::nullopt;
  }
};

bool registerPipeLine(llvm::StringRef Name, llvm::FunctionPassManager &FPM,
                      llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
  if (Name == "lebedeva-bitwise-shift") {
    FPM.addPass(BitwiseShift());
    return true;
  }
  return false;
}

PassPluginLibraryInfo getBitwiseShiftPluginPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "lebedeva-bitwise-shift",
          LLVM_VERSION_STRING, [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(registerPipeLine);
          }};
}

#ifndef LLVM_LEBEDEVABITWISESHIFTPLUGIN_LINK_INTO_TOOLS
extern "C" LLVM_ATTRIBUTE_WEAK PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return getBitwiseShiftPluginPluginInfo();
}
#endif
