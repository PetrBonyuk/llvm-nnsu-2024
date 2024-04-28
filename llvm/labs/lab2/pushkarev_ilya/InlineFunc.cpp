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

// Splits a basic block before the given instruction
llvm::BasicBlock *
splitBlockBefore(llvm::BasicBlock *oldBlock, llvm::Instruction *splitPoint,
                 llvm::DomTreeUpdater *dtu = nullptr,
                 llvm::LoopInfo *li = nullptr,
                 llvm::MemorySSAUpdater *mssaUpdater = nullptr,
                 const llvm::Twine &bbName = "", bool before = true) {
  return llvm::SplitBlock(oldBlock, splitPoint, dtu, li, mssaUpdater, bbName,
                          before);
}

class PushkarevFunctionInliningPass
    : public llvm::PassInfoMixin<PushkarevFunctionInliningPass> {
public:
  llvm::PreservedAnalyses run(llvm::Function &function,
                              llvm::FunctionAnalysisManager &) {
    bool modified = false;
    std::vector<llvm::Instruction *> instructionsToDelete;
    llvm::LLVMContext &context = function.getContext();

    for (llvm::BasicBlock &block : function) {
      for (llvm::Instruction &instruction : block) {

        auto *callInst = llvm::dyn_cast<llvm::CallInst>(&instruction);
        if (!callInst) {
          continue;
        }

        llvm::Function *calledFunction = callInst->getCalledFunction();

        if (calledFunction == &function ||
            !calledFunction->getReturnType()->isVoidTy() ||
            calledFunction->getFunctionType()->getNumParams() > 0) {
          continue;
        }

        modified = true;
        std::map<llvm::BasicBlock *, llvm::BasicBlock *> blockMap;
        llvm::ValueToValueMapTy valueMap;

        llvm::BasicBlock *splitBlock = splitBlockBefore(&block, &instruction);
        llvm::BasicBlock *nextBlock = splitBlock->getNextNode();

        for (llvm::BasicBlock &inlinedBlock : *calledFunction) {
          llvm::BasicBlock *newBlock =
              llvm::BasicBlock::Create(context, "", &function, nextBlock);
          blockMap[&inlinedBlock] = newBlock;
          nextBlock = newBlock->getNextNode();
        }

        for (llvm::BasicBlock &inlinedBlock : *calledFunction) {
          llvm::BasicBlock *currentBlock = blockMap[&inlinedBlock];
          for (llvm::Instruction &inlinedInstruction : inlinedBlock) {
            if (llvm::isa<llvm::ReturnInst>(&inlinedInstruction)) {
              llvm::BranchInst::Create(&block)->insertInto(currentBlock,
                                                           currentBlock->end());
            } else {

              llvm::Instruction *newInstruction = inlinedInstruction.clone();
              if (llvm::isa<llvm::BranchInst>(*newInstruction)) {
                for (unsigned i = 0; i < newInstruction->getNumOperands();
                     ++i) {
                  llvm::Value *operand = newInstruction->getOperand(i);
                  if (auto bb = llvm::dyn_cast<llvm::BasicBlock>(operand)) {
                    newInstruction->setOperand(
                        i, llvm::dyn_cast<llvm::Value>(blockMap[bb]));
                  }
                }
              }
              newInstruction->insertInto(currentBlock, currentBlock->end());
              llvm::RemapInstruction(newInstruction, valueMap,
                                     llvm::RF_NoModuleLevelChanges |
                                         llvm::RF_IgnoreMissingLocals);
              valueMap[&inlinedInstruction] = newInstruction;
            }
          }
        }

        splitBlock->getTerminator()->setOperand(
            0, llvm::dyn_cast<llvm::Value>(
                   blockMap[&calledFunction->getEntryBlock()]));
        instructionsToDelete.push_back(&instruction);
      }
    }

    // delete previous call
    for (llvm::Instruction *instructionToDelete : instructionsToDelete) {
      instructionToDelete->eraseFromParent();
    }

    return modified ? llvm::PreservedAnalyses::none()
                    : llvm::PreservedAnalyses::all();
  }
};
} // namespace

llvm::PassPluginLibraryInfo getPushkarevFunctionInliningPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "PushkarevFunctionInliningPass", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "pushkarev-function-inlining") {
                    PM.addPass(PushkarevFunctionInliningPass());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getPushkarevFunctionInliningPassPluginInfo();
}
