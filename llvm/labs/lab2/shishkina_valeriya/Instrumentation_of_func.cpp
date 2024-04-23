#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct instrFunct : llvm::PassInfoMixin<instrFunct> {
  llvm::PreservedAnalyses run(llvm::Function &func,
                              llvm::FunctionAnalysisManager &) {
    llvm::LLVMContext &context = func.getContext();
    llvm::IRBuilder<> build(context);
    llvm::Module *module = func.getParent();
    bool start = false;
    bool end = false;

    llvm::FunctionType *type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
    llvm::FunctionCallee instrStart =
        (*module).getOrInsertFunction("instrument_start", type);
    llvm::FunctionCallee instrEnd =
        (*module).getOrInsertFunction("instrument_end", type);

    for (auto &block : func) {
      for (auto &instruction : block) {
        if (llvm::isa<llvm::CallInst>(&instruction)) {
          llvm::CallInst *callInst = llvm::cast<llvm::CallInst>(&instruction);
          if (callInst->getCalledFunction() == instrStart.getCallee()) {
            start = true;
          } else if (callInst->getCalledFunction() == instrEnd.getCallee()) {
            end = true;
          }
        }
      }
    }

    if (!start) {
      build.SetInsertPoint(&func.getEntryBlock().front());
      build.CreateCall(instrStart);
    }
    if (!end) {
      for (llvm::BasicBlock &BB : func) {
        if (llvm::dyn_cast<llvm::ReturnInst>(BB.getTerminator())) {
          build.SetInsertPoint(BB.getTerminator());
          build.CreateCall(instrEnd);
        }
      }
    }

    return llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "instrumentation_functions", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instrumentation-functions") {
                    FPM.addPass(instrFunct{});
                    return true;
                  }
                  return false;
                });
          }};
}
