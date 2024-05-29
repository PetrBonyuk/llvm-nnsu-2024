#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

struct InstrumentFunctions : llvm::PassInfoMixin<InstrumentFunctions> {
  llvm::PreservedAnalyses run(llvm::Function &func,
                              llvm::FunctionAnalysisManager &) {
    llvm::LLVMContext &context = func.getContext();
    llvm::IRBuilder<> builder(context);
    auto module = func.getParent();

    llvm::FunctionType *funcType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
    llvm::FunctionCallee instrStartFunc =
        module->getOrInsertFunction("instrument_start", funcType);
    llvm::FunctionCallee instEndFunc =
        module->getOrInsertFunction("instrument_end", funcType);

    bool startInserted = false;
    bool endInserted = false;

    llvm::Function *func_f_f =
        llvm::dyn_cast<llvm::Function>(instrStartFunc.getCallee());
    llvm::Function *func_l_f =
        llvm::dyn_cast<llvm::Function>(instEndFunc.getCallee());

    // Проверка, добавлен ли уже вызов instrument_start
    for (auto *U : func_f_f->users()) {
      if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(U)) {
        if (callInst->getFunction() == &func) {
          startInserted = true;
          break;
        }
      }
    }

    // Проверка, добавлен ли уже вызов instrument_end
    for (auto *U : func_l_f->users()) {
      if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(U)) {
        if (callInst->getFunction() == &func) {
          endInserted = true;
          break;
        }
      }
    }

    if (!startInserted) {
      builder.SetInsertPoint(&func.getEntryBlock().front());
      builder.CreateCall(instrStartFunc);
    }

    if (!endInserted) {
      for (auto &block : func) {
        if (llvm::isa<llvm::ReturnInst>(block.getTerminator())) {
          builder.SetInsertPoint(block.getTerminator());
          builder.CreateCall(instEndFunc);
        }
      }
    }

    return llvm::PreservedAnalyses::all();
  }
};

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "instr_func", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instr_func") {
                    FPM.addPass(InstrumentFunctions{});
                    return true;
                  }
                  return false;
                });
          }};
}
