#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"
#include <algorithm>

using namespace llvm;

namespace {
class X86PolozovReplaceMulAddToFMA : public MachineFunctionPass {
public:
  static char ID;
  X86PolozovReplaceMulAddToFMA() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
    const X86InstrInfo &TII = *STI.getInstrInfo();

    bool Changed = false;
    std::vector<MachineInstr *> Need_Del;

    for (MachineBasicBlock &MBB : MF) {

      auto isUsed = [&](MachineInstr *instr, MachineInstr *mult) {
        for (auto &reg : instr->all_uses()) {
          if (reg.getReg() == mult->getOperand(0).getReg()) {
            return false;
          }
        }
        return true;
      };

      for (auto iterator_for_mult = MBB.begin(); iterator_for_mult != MBB.end();
           ++iterator_for_mult) {
        if (iterator_for_mult->getOpcode() != X86::MULPDrr) {
          continue;
        }
        auto mult = &(*iterator_for_mult);
        bool Can = true;
        std::vector<MachineInstr *> add_instructions;

        auto Check = [&](MachineInstr *add, auto it_begin) {
          ++it_begin;

          for (it_begin; &(*it_begin) != add; ++it_begin) {
            auto instr = &(*it_begin);
            if (std::find(Need_Del.begin(), Need_Del.end(), instr) !=
                Need_Del.end()) {
              continue;
            }
            if (std::find(add_instructions.begin(), add_instructions.end(),
                          instr) != add_instructions.end()) {
              continue;
            }
            for (auto &reg : instr->all_uses()) {
              if (reg.getReg() == add->getOperand(1).getReg() ||
                  reg.getReg() == add->getOperand(2).getReg()) {
                return false;
              }
            }
          }
          return true;
        };

        auto iterator_for_add = std::next(iterator_for_mult);
        for (; iterator_for_add != MBB.end(); ++iterator_for_add) {
          if (iterator_for_add->getOpcode() != X86::ADDPDrr) {
            if (!isUsed(&(*iterator_for_add), mult)) {
              Can = false;
              break;
            }
            continue;
          }
          auto add = &(*iterator_for_add);
          if (std::find(Need_Del.begin(), Need_Del.end(), add) !=
              Need_Del.end()) {
            if (!isUsed(&(*iterator_for_add), mult)) {
              Can = false;
              break;
            }
            continue;
          }
          if (iterator_for_add->getOperand(1).getReg() ==
                  mult->getOperand(0).getReg() ||
              iterator_for_add->getOperand(2).getReg() ==
                  mult->getOperand(0).getReg()) {
            add_instructions.push_back(add);
          }
        }
        for (auto add : add_instructions) {
          if (!Check(add, iterator_for_mult)) {
            Can = false;
          }
        }
        if (Can) {
          Need_Del.push_back(mult);
          for (auto add : add_instructions) {
            Need_Del.push_back(add);
            MIMetadata MIMD(*iterator_for_mult);
            BuildMI(MBB, add, MIMD, TII.get(X86::VFMADD213PDr),
                    add->getOperand(0).getReg())
                .addReg(mult->getOperand(1).getReg())
                .addReg(mult->getOperand(2).getReg())
                .addReg(add->getOperand(2).getReg());
          }
        }
      }
    }

    for (auto &MI : Need_Del) {
      MI->eraseFromParent();
    }

    return Changed;
  }
};
} // end anonymous namespace

char X86PolozovReplaceMulAddToFMA::ID = 0;

static RegisterPass<X86PolozovReplaceMulAddToFMA>
    X("x86-polozov-replace-mul-add-to-fma",
      "X86 replace mul and add to fma pass", false, false);