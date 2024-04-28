#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include <stack>

namespace {
struct MulToBitShiftUpdated : llvm::PassInfoMixin<MulToBitShiftUpdated> {
public:
  llvm::PreservedAnalyses run(llvm::Function &Func,
                              llvm::FunctionAnalysisManager &FAM) {
    std::vector<llvm::Instruction *> toRemove;

    for (llvm::BasicBlock &BB : Func) {
      for (llvm::Instruction &Inst : BB) {
        if (!llvm::BinaryOperator::classof(&Inst)) {
          continue;
        }
        llvm::BinaryOperator *op = llvm::cast<llvm::BinaryOperator>(&Inst);
        if (op->getOpcode() != llvm::Instruction::BinaryOps::Mul) {
          continue;
        }

        llvm::Value *lhs = op->getOperand(0);
        llvm::Value *rhs = op->getOperand(1);

        int lg1 = getLogBase2(lhs);
        int lg2 = getLogBase2(rhs);
        if (lg1 < lg2) {
          std::swap(lg1, lg2);
          std::swap(lhs, rhs);
        }

        if (lg1 > -1) {
          llvm::Value *lg_val = llvm::ConstantInt::get(
              llvm::IntegerType::get(Func.getContext(), 32),
              llvm::APInt(32, lg1));

          llvm::Value *val = llvm::BinaryOperator::Create(
              llvm::Instruction::Shl, rhs, lg_val, op->getName(), op);

          op->replaceAllUsesWith(val);
          toRemove.push_back(op);
        }
      }
      for (auto *I : toRemove) {
        I->eraseFromParent();
      }
    }

    auto PA = llvm::PreservedAnalyses::all();
    return PA;
  }

private:
  int getLogBase2(llvm::Value *val) {
    if (llvm::ConstantInt *CI = llvm::dyn_cast<llvm::ConstantInt>(val)) {
      return CI->getValue().exactLogBase2();
    }
    if (auto *LI = llvm::dyn_cast<llvm::LoadInst>(val)) {
      llvm::Value *Op = LI->getPointerOperand();
      Op->reverseUseList();
      llvm::StoreInst *StInst = nullptr;
      for (auto *Inst : Op->users()) {
        if (auto *SI = llvm::dyn_cast<llvm::StoreInst>(Inst)) {
          StInst = SI;
        }
        if (Inst == LI) {
          break;
        }
      }
      Op->reverseUseList();
      if (!StInst) {
        return -2;
      }
      if (auto *CI =
              llvm::dyn_cast<llvm::ConstantInt>(StInst->getValueOperand())) {
        return CI->getValue().exactLogBase2();
      }
    }
    return -2;
  }
};
} // namespace

bool registerPlugin(llvm::StringRef Name, llvm::FunctionPassManager &FPM,
                    llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
  if (Name == "vesel-mult-shift") {
    FPM.addPass(MulToBitShiftUpdated());
    return true;
  }
  return false;
}

llvm::PassPluginLibraryInfo getMulToBitShiftPluginInfoUpdated() {
  return {LLVM_PLUGIN_API_VERSION, "MulToBitShiftUpdated", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(registerPlugin);
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getMulToBitShiftPluginInfoUpdated();
}
