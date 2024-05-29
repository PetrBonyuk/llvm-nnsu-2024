#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct InstrumentFunction : llvm::PassInfoMixin<InstrumentFunction> {
  llvm::PreservedAnalyses run(llvm::Function &function,
                              llvm::FunctionAnalysisManager &) {
    llvm::LLVMContext &con = function.getContext();
    llvm::IRBuilder<> build(con);
    llvm::Module *mod = function.getParent();
    llvm::FunctionType *type =
        llvm::FunctionType::get(llvm::Type::getVoidTy(con), false);
    llvm::FunctionCallee f_start =
        mod->getOrInsertFunction("instrument_start", type);
    llvm::FunctionCallee f_end =
        mod->getOrInsertFunction("instrument_end", type);
    bool is_start = false;
    bool is_end = false;
    for (auto &bb : function) {
      for (auto &instr : bb) {
        if (llvm::isa<llvm::CallInst>(&instr)) {
          llvm::CallInst *ci = llvm::cast<llvm::CallInst>(&instr);
          if (ci->getCalledFunction() == f_start.getCallee()) {
            is_start = true;
          } else if (ci->getCalledFunction() == f_end.getCallee()) {
            is_end = true;
          }
        }
      }
    }
    if (!is_start) {
      llvm::Instruction *i = &function.front().front();
      build.SetInsertPoint(i);
      build.CreateCall(f_start);
    }
    if (!is_end) {
      for (auto &bb : function) {
        if (llvm::isa<llvm::ReturnInst>(bb.getTerminator())) {
          build.SetInsertPoint(bb.getTerminator());
          build.CreateCall(f_end);
        }
      }
    }
    return llvm::PreservedAnalyses::all();
  }

  static bool require() { return true; }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "InstrumentFunction", "0.1",
          [](llvm::PassBuilder &pb) {
            pb.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &fpm,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instrument_function") {
                    fpm.addPass(InstrumentFunction{});
                    return true;
                  }
                  return false;
                });
          }};
}