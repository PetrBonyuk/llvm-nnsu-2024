#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Value.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Casting.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include <map>
#include <vector>

namespace {
llvm::BasicBlock *splitBlockBefore(llvm::BasicBlock *Old,
                                   llvm::Instruction *SplitPt,
                                   llvm::DomTreeUpdater *DTU = nullptr,
                                   llvm::LoopInfo *LI = nullptr,
                                   llvm::MemorySSAUpdater *MSSAU = nullptr,
                                   const llvm::Twine &BBName = "",
                                   bool Before = true) {
  return llvm::SplitBlock(Old, SplitPt, DTU, LI, MSSAU, BBName, Before);
}

class VyunovMagicInlinePass
    : public llvm::PassInfoMixin<VyunovMagicInlinePass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &Func,
                              llvm::FunctionAnalysisManager &) {
    bool IsTransformed = false;
    std::vector<llvm::Instruction *> InstrDeleteList;
    llvm::LLVMContext &CTX = Func.getContext();
    for (llvm::BasicBlock &Block : Func) {
      for (llvm::Instruction &Instr : Block) {
        llvm::CallInst *CallInstruction =
            llvm::dyn_cast<llvm::CallInst>(&Instr);
        if (!CallInstruction) {
          continue;
        }
        llvm::Function *Called = CallInstruction->getCalledFunction();
        if (Called == &Func || !Called->getReturnType()->isVoidTy() ||
            Called->getFunctionType()->getNumParams() > 0) {
          continue;
        }
        IsTransformed = true;
        std::map<llvm::BasicBlock *, llvm::BasicBlock *> BlockMap;
        llvm::ValueToValueMapTy VMap;
        llvm::BasicBlock *SplittedBlock = splitBlockBefore(&Block, &Instr);
        llvm::BasicBlock *NextBlockPlace = SplittedBlock->getNextNode();
        for (llvm::BasicBlock &ToInlineBasicBlock : *Called) {
          llvm::BasicBlock *TmpBlock =
              llvm::BasicBlock::Create(CTX, "", &Func, NextBlockPlace);
          BlockMap[&ToInlineBasicBlock] = TmpBlock;
          NextBlockPlace = TmpBlock->getNextNode();
        }
        for (llvm::BasicBlock &ToInlineBasicBlock : *Called) {
          for (llvm::Instruction &ToInlineInstr : ToInlineBasicBlock) {
            llvm::BasicBlock *CurrentBlock = BlockMap[&ToInlineBasicBlock];
            if (llvm::isa<llvm::ReturnInst>(&ToInlineInstr)) {
              llvm::BranchInst::Create(&Block)->insertInto(CurrentBlock,
                                                           CurrentBlock->end());
            } else {
              llvm::Instruction *NewInstr = ToInlineInstr.clone();
              if (llvm::isa<llvm::BranchInst>(*NewInstr)) {
                unsigned OpNum = NewInstr->getNumOperands();
                for (unsigned Ind = 0; Ind < OpNum; Ind++) {
                  llvm::Value *CurOperand = NewInstr->getOperand(Ind);
                  if (llvm::isa<llvm::BasicBlock>(*CurOperand)) {
                    NewInstr->setOperand(
                        Ind, llvm::dyn_cast<llvm::Value>(
                                 BlockMap[llvm::dyn_cast<llvm::BasicBlock>(
                                     CurOperand)]));
                  }
                }
              }
              NewInstr->insertInto(CurrentBlock, CurrentBlock->end());
              llvm::RemapInstruction(NewInstr, VMap,
                                     llvm::RF_NoModuleLevelChanges |
                                         llvm::RF_IgnoreMissingLocals);
              VMap[&ToInlineInstr] = NewInstr;
            }
          }
        }
        SplittedBlock->getTerminator()->setOperand(
            0, llvm::dyn_cast<llvm::Value>(BlockMap[&Called->getEntryBlock()]));
        InstrDeleteList.push_back(&Instr);
      }
    }
    for (llvm::Instruction *Instr : InstrDeleteList) {
      Instr->eraseFromParent();
    }
    if (IsTransformed) {
      return llvm::PreservedAnalyses::none();
    }
    return llvm::PreservedAnalyses::all();
  }
};
} // namespace

llvm::PassPluginLibraryInfo getInlineFunctionsVyunovDanilaFI2PluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "VyunovMagicInlinePass", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "vyunov-magic-inline") {
                    PM.addPass(VyunovMagicInlinePass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getInlineFunctionsVyunovDanilaFI2PluginInfo();
}
