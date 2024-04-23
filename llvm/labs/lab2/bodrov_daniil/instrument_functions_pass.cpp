#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/raw_ostream.h"

namespace {
struct InstrumentFunctionsPass : llvm::PassInfoMixin<InstrumentFunctionsPass> {

  // Function to check for existing instrument calls in the function
  bool checkForInstrumentCalls(llvm::Function &F, llvm::FunctionCallee &func) {
    for (auto *user : func.getCallee()->users()) {
      if (auto *callInst = llvm::dyn_cast<llvm::CallInst>(user)) {
        if (callInst->getParent()->getParent() == &F) {
          return true;
        }
      }
    }
    return false;
  }

  // Function to insert instrument function call if not already inserted
  void insertInstrumentCall(llvm::Function &F, llvm::FunctionCallee &func,
                            llvm::IRBuilder<> &builder, bool &inserted) {
    if (!inserted) {
      builder.SetInsertPoint(&F.getEntryBlock().front());
      builder.CreateCall(func);
      inserted = true;
    }
  }

  // Function to find the last return instruction within a function
  llvm::ReturnInst *findLastReturnInst(llvm::Function &F) {
    llvm::ReturnInst *lastReturnInst = nullptr;
    for (llvm::BasicBlock &BB : F) {
      llvm::Instruction *terminator = BB.getTerminator();
      if (llvm::isa<llvm::ReturnInst>(terminator)) {
        lastReturnInst = llvm::cast<llvm::ReturnInst>(terminator);
      }
    }
    return lastReturnInst;
  }

  llvm::PreservedAnalyses run(llvm::Function &F,
                              llvm::FunctionAnalysisManager &) {
    llvm::LLVMContext &context = F.getContext();
    llvm::IRBuilder<> builder(context);
    llvm::Module *module = F.getParent();

    // Get the instrument_start() and instrument_end() functions
    llvm::FunctionType *funcType =
        llvm::FunctionType::get(llvm::Type::getVoidTy(context), false);
    llvm::FunctionCallee startFunc =
        module->getOrInsertFunction("instrument_start", funcType);
    llvm::FunctionCallee endFunc =
        module->getOrInsertFunction("instrument_end", funcType);

    bool startInserted = checkForInstrumentCalls(F, startFunc);
    bool endInserted = checkForInstrumentCalls(F, endFunc);

    // Insert instrument_start() if not already inserted
    insertInstrumentCall(F, startFunc, builder, startInserted);

    // Insert instrument_end() if not already inserted
    if (!endInserted) {
      llvm::ReturnInst *lastReturnInst = findLastReturnInst(F);

      if (lastReturnInst) {
        builder.SetInsertPoint(lastReturnInst);
        builder.CreateCall(endFunc);
        endInserted = true;
      }
    }

    return llvm::PreservedAnalyses::all();
  }

  static bool isRequired() { return true; }
};
} // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "InstrumentFunctionsPass", "0.1",
          [](llvm::PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](llvm::StringRef name, llvm::FunctionPassManager &FPM,
                   llvm::ArrayRef<llvm::PassBuilder::PipelineElement>) -> bool {
                  if (name == "instrument_functions_pass") {
                    FPM.addPass(InstrumentFunctionsPass{});
                    return true;
                  }
                  return false;
                });
          }};
}
