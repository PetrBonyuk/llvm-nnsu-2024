#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include <vector>

using namespace llvm;

class X86SoloninkoOptsPass : public llvm::MachineFunctionPass {
public:
  static inline char ID = 0;

  X86SoloninkoOptsPass() : llvm::MachineFunctionPass(ID) {}

private:
  void buildMI(llvm::MachineBasicBlock &MachineBlock, MachineInstr *Mul,
               const llvm::TargetInstrInfo *TII, MachineInstr *Add) {
    llvm::MachineInstrBuilder MIB =
        BuildMI(*Mul->getParent(), *Mul, Mul->getDebugLoc(),
                TII->get(llvm::X86::VFMADD213PDr));
    MIB.addReg(Add->getOperand(0).getReg(), llvm::RegState::Define);
    MIB.addReg(Mul->getOperand(1).getReg());
    MIB.addReg(Mul->getOperand(2).getReg());
    MIB.addReg(Add->getOperand(2).getReg());
  }

  bool runOnMachineFunction(llvm::MachineFunction &MF) override {
    const llvm::TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
    std::vector<MachineInstr *> vecMI;
    MachineInstr *Mul;
    MachineInstr *Add;
    for (llvm::MachineBasicBlock &MachineBlock : MF) {

      for (llvm::MachineBasicBlock::iterator MBIter = MachineBlock.begin();
           MBIter != MachineBlock.end(); ++MBIter) {
        if (MBIter->getOpcode() != llvm::X86::MULPDrr) {
          continue;
        }
        Mul = &(*MBIter);
        MBIter++;
        for (llvm::MachineBasicBlock::iterator MBIterN = MBIter;
             MBIterN != MachineBlock.end(); MBIterN++) {
          if (MBIterN->getOpcode() == llvm::X86::ADDPDrr) {
            Add = &(*MBIterN);

            if (Mul->getOperand(0).getReg() == Add->getOperand(1).getReg()) {
              vecMI.emplace_back(Add);
              vecMI.emplace_back(Mul);

              buildMI(MachineBlock, Mul, TII, Add);

              break;
            }
          }
        }
      }
    }
    for (llvm::MachineInstr *instr : vecMI) {
      instr->eraseFromParent();
    }
    return true;
  }
};

static RegisterPass<X86SoloninkoOptsPass> X("x86-soloninko-lab3",
                                            "X86 Soloninko Pass", false, false);
