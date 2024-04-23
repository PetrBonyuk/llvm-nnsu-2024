#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/GraphTraits.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Pass.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/Compiler.h"

namespace {

void InsertLoopStart(llvm::FunctionCallee LoopStartFunc, llvm::Loop *L,
                     llvm::IRBuilder<> &Builder) {
  auto *Preheader = L->getLoopPreheader();
  if (!Preheader)
    return;
  Builder.SetInsertPoint(Preheader->getTerminator());
  Builder.CreateCall(LoopStartFunc);
}

void InsertLoopEnd(llvm::FunctionCallee LoopEndFunc, llvm::Loop *L,
                   llvm::IRBuilder<> &Builder) {
  llvm::SmallVector<llvm::BasicBlock *, 8> ExitBlocks;
  L->getExitBlocks(ExitBlocks);

  for (auto *ExitBlock : ExitBlocks) {
    if (ExitBlock) {
      Builder.SetInsertPoint(&*(ExitBlock->getFirstInsertionPt()));
      Builder.CreateCall(LoopEndFunc);
    }
  }
}

void ProcessLoop(llvm::Loop *L, llvm::FunctionCallee LoopStartFunc,
                 llvm::FunctionCallee LoopEndFunc, llvm::IRBuilder<> &Builder) {
  InsertLoopStart(LoopStartFunc, L, Builder);
  InsertLoopEnd(LoopEndFunc, L, Builder);
  for (auto *SubLoop : L->getSubLoops()) {
    ProcessLoop(SubLoop, LoopStartFunc, LoopEndFunc, Builder);
  }
}

struct PetrovWrapLoopPlugin : llvm::PassInfoMixin<PetrovWrapLoopPlugin> {
  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &FAM) {
    auto *M = F.getParent();
    auto *Ctx = &F.getContext();
    auto LoopStartFunc = M->getOrInsertFunction(
        "loop_start",
        llvm::FunctionType::get(llvm::Type::getVoidTy(*Ctx), false));
    auto LoopEndFunc = M->getOrInsertFunction(
        "loop_end",
        llvm::FunctionType::get(llvm::Type::getVoidTy(*Ctx), false));

    llvm::IRBuilder<> Builder(*Ctx);
    auto &LInfo = FAM.getResult<llvm::LoopAnalysis>(F);
    for (auto *L : LInfo) {
      ProcessLoop(L, LoopStartFunc, LoopEndFunc, Builder);
    }
    return llvm::PreservedAnalyses::none();
  }
};

} // namespace
llvm::PassPluginLibraryInfo getPetrovWrapLoopPluginPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "PetrovWrapLoopPlugin", LLVM_VERSION_STRING,
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef Name, llvm::FunctionPassManager &PM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) {
                  if (Name == "PetrovWrapLoopPlugin") {
                    PM.addPass(PetrovWrapLoopPlugin());
                    return true;
                  }
                  return false;
                });
          }};
}

extern "C" LLVM_ATTRIBUTE_WEAK llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return getPetrovWrapLoopPluginPluginInfo();
}
