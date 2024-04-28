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

    Function *Callee{};
    std::vector<CallInst *> callsToInline{};
    for (auto &BB : F) {
      for (auto &I : BB) {
        if (auto *CI = dyn_cast<CallInst>(&I)) {
          Callee = CI->getCalledFunction();
          if (Callee && Callee->arg_size() == 0 &&
              Callee->getReturnType()->isVoidTy()) {
            callsToInline.push_back(CI);
          }
        }
      }
    }

    if (callsToInline.empty())
      return PreservedAnalyses::all();

    // Process each call site
    size_t counter_splited = 0;
    size_t counter_inlined = 0;

    for (CallInst *CI : callsToInline) {
      BasicBlock *InsertBB = CI->getParent();
      Instruction *InsertPt = CI->getNextNode();

      // Create a map for block correspondence
      DenseMap<BasicBlock *, BasicBlock *> BlockMap;

      // Split the calling function at the call site (use ".splited." for
      // clarity)
      BasicBlock *SplitBB = InsertBB->splitBasicBlock(
          InsertPt, InsertBB->getName() + ".splited." +
                        std::to_string(counter_splited++));

      // Create new blocks corresponding to empty function's blocks
      for (BasicBlock &CalleeBB : *Callee) {
        BasicBlock *NewBB = BasicBlock::Create(
            F.getContext(),
            CalleeBB.getName() + ".inlined." + std::to_string(counter_inlined),
            &F);
        BlockMap[&CalleeBB] = NewBB;
      }
      InsertBB->getTerminator()->setSuccessor(
          0, BlockMap[&Callee->getEntryBlock()]);

      counter_inlined++;

      // Copy instructions from corresponding blocks
      ValueToValueMapTy VMap;
      for (BasicBlock &CalleeBB : *Callee) {
        BasicBlock *NewBB = BlockMap[&CalleeBB];
        for (Instruction &Inst : CalleeBB) {
          IRBuilder<> Builder(
              NewBB); // Create IRBuilder with specified basic block
          Instruction *NewInst = Inst.clone(); // Clone the instruction
          Builder.Insert(NewInst); // Add the cloned instruction to the end of
                                   // the basic block
          VMap[&Inst] = NewInst;
        }
      }

      // Update branch instructions to target correct blocks
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

      for (BasicBlock &CalleeBB : *Callee) {
        BasicBlock *NewBB = BlockMap[&CalleeBB];
        for (Instruction &Inst : *NewBB) {
          RemapInstruction(&Inst, VMap, RF_IgnoreMissingLocals, nullptr,
                           nullptr);
        }
      }

      // Create IRBuilder and set its context
      IRBuilder<> Builder(F.getContext());

      // Traversing basic blocks in Caller
      for (BasicBlock &BB : F) {
        // Creating a copy of the instruction list of ReturnInst type for the
        // current basic block
        SmallVector<ReturnInst *, 16> ReturnInstructions;
        for (Instruction &I : BB) {
          if (auto *RI = dyn_cast<ReturnInst>(&I)) {
            ReturnInstructions.push_back(RI);
          }
        }

        for (ReturnInst *RI : ReturnInstructions) {
          if (RI->getParent() != SplitBB) {
            RI->eraseFromParent();

            // Creating an unconditional branch instruction using IRBuilder
            Builder.SetInsertPoint(
                &BB); // Setting the insertion point to the current basic block
            Builder.CreateBr(
                SplitBB); // Creating an unconditional branch to SplitBB
          }
        }
      }

      // Remove the call instruction
      CI->eraseFromParent();

      // Create an iterator for the basic block SplitBB
      Function::iterator it = F.begin();
      // Iterate through all basic blocks until we find SplitBB
      while (&*it != SplitBB && it != F.end()) {
        ++it;
      }
      Function::iterator next_it{};
      // If we found SplitBB and it's not the last block
      while (it != F.end() && std::next(it) != F.end()) {
        // Create a new iterator for the next block
        next_it = std::next(it++);

        // Move SplitBB after the next block
      }
      if (it != F.end())
        SplitBB->moveAfter(&*next_it);
    }

    return PreservedAnalyses::none();
  }
  static bool isRequired() { return true; }
};

} // end anonymous namespace

extern "C" ::llvm::PassPluginLibraryInfo LLVM_ATTRIBUTE_WEAK
llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "CustomInliningPass", "v0.1",
          [](PassBuilder &PB) {
            PB.registerPipelineParsingCallback(
                [](StringRef Name, FunctionPassManager &FPM,
                   ArrayRef<PassBuilder::PipelineElement>) {
                  if (Name == "custom-inlining") {
                    FPM.addPass(CustomInliningPass());
                    return true;
                  }
                  return false;
                });
          }};
}
