#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"
#include <utility>
#include <vector>

using namespace llvm;

namespace {
class X86SadikovPassFMA : public MachineFunctionPass {
public:
  static char ID;

  X86SadikovPassFMA() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    bool instructions_changed = false;

    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

    std::vector<std::pair<MachineInstr *, MachineInstr *>> pairs;

    for (MachineBasicBlock &BB : MF) {
      for (auto MI = BB.begin(); MI != BB.end(); ++MI) {
        MachineInstr *mulInstr = &(*MI);
        if (mulInstr->getOpcode() == X86::MULPDrr ||
            mulInstr->getOpcode() == X86::MULPDrm) {
          MachineInstr *addInstr = nullptr;
          for (auto MI_2 = std::next(MI); MI_2 != BB.end(); ++MI_2) {
            if (MI_2->readsRegister(mulInstr->getOperand(0).getReg())) {
              if ((MI_2->getOpcode() == X86::ADDPDrr ||
                   MI_2->getOpcode() == X86::ADDPDrm) &&
                  addInstr == nullptr &&
                  mulInstr->getOperand(0).getReg() ==
                      MI_2->getOperand(1).getReg()) {
                addInstr = &(*MI_2);
              } else {
                addInstr = nullptr;
                break;
              }
            }
          }
          if (addInstr) {
            if (addInstr->getOperand(1).getReg() !=
                addInstr->getOperand(2).getReg()) {
              pairs.emplace_back(mulInstr, addInstr);
            }
          }
        }
      }
    }

    for (auto [mulInstr, addInstr] : pairs) {
      MIMetadata MIMD(*mulInstr);
      MachineBasicBlock &BB = *mulInstr->getParent();

      BuildMI(BB, mulInstr, MIMD, TII->get(X86::VFMADD213PDr),
              addInstr->getOperand(0).getReg())
          .addReg(mulInstr->getOperand(1).getReg())
          .addReg(mulInstr->getOperand(2).getReg())
          .addReg(addInstr->getOperand(2).getReg());

      mulInstr->eraseFromParent();
      addInstr->eraseFromParent();

      instructions_changed = true;
    }

    return instructions_changed;
  }
};

char X86SadikovPassFMA::ID = 0;

} // namespace

static RegisterPass<X86SadikovPassFMA> X("x86-sadikov-pass-fma",
                                         "X86 Sadikov Pass FMA", false, false);
