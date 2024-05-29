#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

class X86KostanyanPass : public MachineFunctionPass {
public:
  static inline char ID = 0;

  X86KostanyanPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MFunc) override {
    DebugLoc DLocation = MFunc.front().begin()->getDebugLoc();
    const TargetInstrInfo *InstrInfo = MFunc.getSubtarget().getInstrInfo();
    const TargetRegisterInfo *RegInfo = MFunc.getSubtarget().getRegisterInfo();

    for (auto &BB : MFunc) {
      size_t Count = std::distance(BB.begin(), BB.end());

      const TargetRegisterClass *RC = RegInfo->getRegClass(X86::GR64RegClassID);
      Register TmpReg = MFunc.getRegInfo().createVirtualRegister(RC);

      BuildMI(BB, BB.getFirstTerminator(), DLocation,
              InstrInfo->get(X86::MOV64rm), TmpReg)
          .addExternalSymbol("ic");

      BuildMI(BB, BB.getFirstTerminator(), DLocation,
              InstrInfo->get(X86::ADD64ri32), TmpReg)
          .addReg(TmpReg)
          .addImm(Count);

      BuildMI(BB, BB.getFirstTerminator(), DLocation,
              InstrInfo->get(X86::MOV64mr))
          .addExternalSymbol("ic")
          .addReg(TmpReg);
    }

    return true;
  }
};

static RegisterPass<X86KostanyanPass>
    X("x86-kostanyan-count-inst",
      "Pass that counts the number of machine instructions", false, false);
