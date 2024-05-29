#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <string>

using namespace llvm;

namespace {

struct CustomInliningPass : public PassInfoMixin<CustomInliningPass> {
  PreservedAnalyses run(Function &F, FunctionAnalysisManager &AM) {
    std::vector<CallInst *> callsToInline = findCallsToInline(F);

    if (callsToInline.empty())
      return PreservedAnalyses::all();

    processCallSites(F, callsToInline);

    return PreservedAnalyses::none();
  }

  static bool isRequired() { return true; }

private:
  std::vector<CallInst *> findCallsToInline(Function &F) {
    std::vector<CallInst *> callsToInline{};
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          if (Function *Callee = CI->getCalledFunction()) {
            if (Callee->arg_size() == 0 &&
                Callee->getReturnType()->isVoidTy()) {
              callsToInline.push_back(CI);
            }
          }
        }
      }
    }
    return callsToInline;
  }

  void processCallSites(Function &F, std::vector<CallInst *> &callsToInline) {
    size_t counter_splited = 0;
    size_t counter_inlined = 0;

    for (CallInst *CI : callsToInline) {
      BasicBlock *InsertBB = CI->getParent();
      Instruction *InsertPt = CI->getNextNode();

      DenseMap<BasicBlock *, BasicBlock *> BlockMap;
      Function *Callee = CI->getCalledFunction();
      ValueToValueMapTy VMap;

      BasicBlock *SplitBB =
          splitBlockAtCallSite(InsertBB, InsertPt, counter_splited);
      createNewBlocksForFunction(F, Callee, BlockMap, counter_inlined);
      updateTerminatorOfInsertBB(InsertBB, Callee, BlockMap);

      copyInstructionsFromCallee(F, Callee, BlockMap, VMap);
      updateBranchInstructions(F, BlockMap);
      remapInstructions(Callee, BlockMap, VMap);

      handleReturnInstructions(F, SplitBB);
      CI->eraseFromParent();
      moveSplitBBAfterNextBlock(F, SplitBB);
    }
  }

  BasicBlock *splitBlockAtCallSite(BasicBlock *InsertBB, Instruction *InsertPt,
                                   size_t &counter_splited) {
    return InsertBB->splitBasicBlock(InsertPt,
                                     InsertBB->getName() + ".splited." +
                                         std::to_string(counter_splited++));
  }

  void
  createNewBlocksForFunction(Function &F, Function *Callee,
                             DenseMap<BasicBlock *, BasicBlock *> &BlockMap,
                             size_t &counter_inlined) {
    for (BasicBlock &CalleeBB : *Callee) {
      BasicBlock *NewBB = BasicBlock::Create(
          F.getContext(),
          CalleeBB.getName() + ".inlined." + std::to_string(counter_inlined),
          &F);
      BlockMap[&CalleeBB] = NewBB;
    }
    counter_inlined++;
  }

  void
  updateTerminatorOfInsertBB(BasicBlock *InsertBB, Function *Callee,
                             DenseMap<BasicBlock *, BasicBlock *> &BlockMap) {
    InsertBB->getTerminator()->setSuccessor(0,
                                            BlockMap[&Callee->getEntryBlock()]);
  }

  void
  copyInstructionsFromCallee(Function &F, Function *Callee,
                             DenseMap<BasicBlock *, BasicBlock *> &BlockMap,
                             ValueToValueMapTy &VMap) {
    for (BasicBlock &CalleeBB : *Callee) {
      BasicBlock *NewBB = BlockMap[&CalleeBB];
      for (Instruction &Inst : CalleeBB) {
        IRBuilder<> Builder(NewBB);
        Instruction *NewInst = Inst.clone();
        Builder.Insert(NewInst);
        VMap[&Inst] = NewInst;
      }
    }
  }

  void
  updateBranchInstructions(Function &F,
                           DenseMap<BasicBlock *, BasicBlock *> &BlockMap) {
    for (BasicBlock &CalleeBB : F) {
      for (Instruction &Branch : CalleeBB) {
        if (auto *BI = dyn_cast<BranchInst>(&Branch)) {
          for (size_t i = 0, e = BI->getNumSuccessors(); i != e; ++i) {
            BasicBlock *Successor = BI->getSuccessor(i);
            if (BlockMap[Successor] != nullptr) {
              BI->setSuccessor(i, BlockMap[Successor]);
            }
          }
        }
      }
    }
  }

  void remapInstructions(Function *Callee,
                         DenseMap<BasicBlock *, BasicBlock *> &BlockMap,
                         ValueToValueMapTy &VMap) {
    for (BasicBlock &CalleeBB : *Callee) {
      BasicBlock *NewBB = BlockMap[&CalleeBB];
      for (Instruction &Inst : *NewBB) {
        RemapInstruction(&Inst, VMap, RF_IgnoreMissingLocals, nullptr, nullptr);
      }
    }
  }

  void handleReturnInstructions(Function &F, BasicBlock *SplitBB) {
    IRBuilder<> Builder(F.getContext());
    for (BasicBlock &BB : F) {
      SmallVector<ReturnInst *, 16> ReturnInstructions;
      for (Instruction &I : BB) {
        if (auto *RI = dyn_cast<ReturnInst>(&I)) {
          ReturnInstructions.push_back(RI);
        }
      }

      for (ReturnInst *RI : ReturnInstructions) {
        if (RI->getParent() != SplitBB) {
          RI->eraseFromParent();
          Builder.SetInsertPoint(&BB);
          Builder.CreateBr(SplitBB);
        }
      }
    }
  }

  void moveSplitBBAfterNextBlock(Function &F, BasicBlock *SplitBB) {
    Function::iterator it = F.begin();
    while (it != F.end() && &*it != SplitBB) {
      ++it;
    }
    Function::iterator next_it{};
    while (it != F.end() && std::next(it) != F.end()) {
      next_it = std::next(it++);
    }
    if (it != F.end())
      SplitBB->moveAfter(&*next_it);
  }
};

} // end anonymous namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "AkopyanInliningPass", "v0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "akopyan-inlining") {
                    FPM.addPass(CustomInliningPass());
                    return true;
                  }
                  return false;
                });
          }};
}