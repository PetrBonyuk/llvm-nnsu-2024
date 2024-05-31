#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

class X86TravinCountInstPass : public MachineFunctionPass {
public:
  static inline char ID = 0;

  X86TravinCountInstPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MachFunc) override {
    DebugLoc DebugLocation = MachFunc.front().begin()->getDebugLoc();
    const TargetInstrInfo *InstrInfo = MachFunc.getSubtarget().getInstrInfo();
    const TargetRegisterInfo *RegInfo =
        MachFunc.getSubtarget().getRegisterInfo();

    for (auto &BasicBlock : MachFunc) {
      size_t Count = std::distance(BasicBlock.begin(), BasicBlock.end());

      const TargetRegisterClass *RC = RegInfo->getRegClass(X86::GR64RegClassID);
      Register TmpReg = MachFunc.getRegInfo().createVirtualRegister(RC);

      BuildMI(BasicBlock, BasicBlock.getFirstTerminator(), DebugLocation,
              InstrInfo->get(X86::MOV64rm), TmpReg)
          .addExternalSymbol("ic");

      BuildMI(BasicBlock, BasicBlock.getFirstTerminator(), DebugLocation,
              InstrInfo->get(X86::ADD64ri32), TmpReg)
          .addReg(TmpReg)
          .addImm(Count);

      BuildMI(BasicBlock, BasicBlock.getFirstTerminator(), DebugLocation,
              InstrInfo->get(X86::MOV64mr))
          .addExternalSymbol("ic")
          .addReg(TmpReg);
    }

    return true;
  }
};

static RegisterPass<X86TravinCountInstPass>
    X("x86-travin-count-inst", "Pass counting number of machine instructions",
      false, false);