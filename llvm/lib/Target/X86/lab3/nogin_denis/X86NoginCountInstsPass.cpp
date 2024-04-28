#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

class X86NoginCountInstsPass : public MachineFunctionPass {
public:
  static inline char ID = 0;

  X86NoginCountInstsPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MachFunc) override {
    DebugLoc DebugLocation = MachFunc.front().begin()->getDebugLoc();
    const TargetInstrInfo *InstrInfo = MachFunc.getSubtarget().getInstrInfo();

    for (auto &BasicBlock : MachFunc) {
      size_t Count = 0;
      for (auto &Instr : BasicBlock) {
        Count++;
      }

      BuildMI(BasicBlock, BasicBlock.getFirstTerminator(), DebugLocation,
              InstrInfo->get(X86::ADD64ri32))
          .addImm(Count)
          .addExternalSymbol("ic");
    }

    return true;
  }
};

static RegisterPass<X86NoginCountInstsPass>
    X("x86-nogin-count-insts",
      "A pass counting the number of X86 machine instructions", false, false);