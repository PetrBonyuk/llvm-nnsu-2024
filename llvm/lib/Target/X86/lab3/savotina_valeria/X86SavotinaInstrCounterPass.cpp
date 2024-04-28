#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;

class X86SavotinaInstrCounterPass : public MachineFunctionPass {
public:
  static inline char ID = 0;

  X86SavotinaInstrCounterPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &machFunc) override {
    const TargetInstrInfo *tgtInstrInfo = nullptr;
    DebugLoc dbgLoc;

    if (!machFunc.empty() &&
        machFunc.front().begin() != machFunc.front().end()) {
      tgtInstrInfo = machFunc.getSubtarget().getInstrInfo();
      dbgLoc = machFunc.front().begin()->getDebugLoc();

      if (tgtInstrInfo) {
        for (auto BBIter = machFunc.begin(); BBIter != machFunc.end();
             ++BBIter) {
          MachineBasicBlock &machBasicBlock = *BBIter;

          if (!machBasicBlock.empty()) {
            unsigned instrCount = 0;
            for (auto InstrIter = machBasicBlock.begin();
                 InstrIter != machBasicBlock.end(); ++InstrIter) {
              instrCount++;
            }

            BuildMI(machBasicBlock, machBasicBlock.getFirstTerminator(), dbgLoc,
                    tgtInstrInfo->get(X86::ADD64ri32))
                .addImm(instrCount)
                .addExternalSymbol("ic");
          }
        }
      }
    }

    return true;
  }
};

static RegisterPass<X86SavotinaInstrCounterPass>
    X("x86-savotina-instr-counter",
      "This pass let to instrCount the X86 machine instructions' number", false,
      false);