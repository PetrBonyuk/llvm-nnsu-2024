#include "llvm/IR/IRBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Debug.h"

struct LoopPlugin : public llvm::PassInfoMixin<LoopPlugin> {
  llvm::PreservedAnalyses
  run(llvm::Function &Func, llvm::FunctionAnalysisManager &Func_analys_manag) {
    llvm::dbgs() << "Entering run\n";

    llvm::LLVMContext &llvmcont = Func.getContext();
    llvm::Module *par_module = Func.getParent();

    llvm::FunctionType *function_type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(llvmcont), false);

    llvm::LoopAnalysis::Result &LI =
        Func_analys_manag.getResult<llvm::LoopAnalysis>(Func);
    for (auto *Loop : LI) {
      llvm::dbgs() << "Entering loop\n";

      llvm::IRBuilder<> Builder(Loop->getHeader()->getContext());
      llvm::BasicBlock *Header = Loop->getHeader();
      for (auto *const pre_header :
           llvm::children<llvm::Inverse<llvm::BasicBlock *>>(Header)) {
        if (Loop->contains(pre_header) &&
            !LoopCallPresent("loop_start", pre_header)) {
          Builder.SetInsertPoint(pre_header->getTerminator());
          Builder.CreateCall(
              par_module->getOrInsertFunction("loop_start", function_type));
        }
      }

      llvm::SmallVector<llvm::BasicBlock *, 4> llvm_blocks;
      Loop->getExitBlocks(llvm_blocks);
      for (auto *const llvm_block : llvm_blocks) {
        if (!LoopCallPresent("loop_end", llvm_block) &&
            LastBlock(llvm_block, llvm_blocks)) {
          Builder.SetInsertPoint(llvm_block->getFirstNonPHI());
          Builder.CreateCall(
              par_module->getOrInsertFunction("loop_end", function_type));
        }
      }
    }
    return llvm::PreservedAnalyses::all();
  }

  bool LoopCallPresent(const std::string &LoopFuncName, llvm::BasicBlock *BB) {
    llvm::Function *TargetFunc =
        BB->getParent()->getParent()->getFunction(LoopFuncName);
    if (!TargetFunc)
      return false;

    for (auto *U : TargetFunc->users()) {
      if (auto *Inst = llvm::dyn_cast<llvm::Instruction>(U)) {
        if (BB == Inst->getParent()) {
          return true;
        }
      }
    }
    return false;
  }

  bool LastBlock(llvm::BasicBlock *const BasicB,
                 const llvm::SmallVector<llvm::BasicBlock *, 4> &llvm_blocks) {
    auto *Branch = llvm::dyn_cast<llvm::BranchInst>(BasicB->getTerminator());
    if (!Branch || !Branch->isUnconditional())
      return true;
    llvm::BasicBlock *TargetBB = Branch->getSuccessor(0);
    for (llvm::BasicBlock *BBlock : llvm_blocks)
      if (BBlock == TargetBB)
        return false;
    return true;
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "LoopPlugin", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PassBuild) {
            PassBuild.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PassManag,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "bonyuk-loop-plugin") {
                    PassManag.addPass(LoopPlugin());
                    return true;
                  }
                  return false;
                });
          }};
}