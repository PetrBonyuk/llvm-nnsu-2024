#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"
#include <vector>

using namespace llvm;

namespace {
class X86BendPass : public MachineFunctionPass {
public:
  static char ID;

  X86BendPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return "X86 Bend Pass"; }
};
} // namespace

char X86BendPass::ID = 0;

bool X86BendPass::runOnMachineFunction(MachineFunction &MF) {
  bool Modified = false;
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

  for (auto &MBB : MF) {
    std::vector<llvm::MachineBasicBlock::iterator> mulToErase;
    llvm::Register ar2;
    MachineInstr *AI;
    for (auto I = MBB.begin(); I != MBB.end(); I++) {
      if (I->getOpcode() == X86::MULPDrr) {
        const llvm::Register mr0 = I->getOperand(0).getReg();
        bool Found = false;
        for (auto J = I; J != MBB.end(); J++) {
          if (J->getOpcode() == X86::ADDPDrr) {
            llvm::Register ar1 = J->getOperand(1).getReg();
            ar2 = J->getOperand(2).getReg();
            if (mr0 != ar1 && mr0 != ar2)
              continue;
            if (Found) {
              Found = false;
              break;
            }
            if (mr0 == ar2) {
              ar2 = ar1;
            }
            AI = &(*J);
            Found = true;
          }
        }
        if (!Found)
          continue;

        auto &MI = *I;
        MIMetadata MIMD(*AI);
        MachineInstrBuilder MIB =
            BuildMI(MBB, *AI, MIMD, TII->get(X86::VFMADD213PDr));
        MIB.addReg(AI->getOperand(0).getReg(), RegState::Define);
        MIB.addReg(MI.getOperand(1).getReg());
        MIB.addReg(MI.getOperand(2).getReg());
        MIB.addReg(ar2);
        AI->eraseFromParent();
        mulToErase.emplace_back(I);
        Modified = true;
      }
    }
    for (auto &mul : mulToErase) {
      (*mul).eraseFromParent();
    }
  }

  return Modified;
}

static RegisterPass<X86BendPass> X("x86-bend-pass", "X86 Bend Pass", false,
                                   false);
