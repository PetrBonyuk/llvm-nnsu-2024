#include <algorithm>

#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"

using namespace llvm;

namespace {
class X86VolodinEMulAddPass : public MachineFunctionPass {
public:
  static char ID;
  X86VolodinEMulAddPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    const X86Subtarget &STI = MF.getSubtarget<X86Subtarget>();
    const X86InstrInfo &TII = *STI.getInstrInfo();

    bool Changed = false;
    std::vector<MachineInstr *> MIvector;

    for (MachineBasicBlock &MBB : MF) {
      for (auto iterator = MBB.begin(); iterator != MBB.end(); ++iterator) {
        if (iterator->getOpcode() != X86::MULPDrr) {
          continue;
        }
        auto multiplicaton = &(*iterator);
        for (auto iter = std::next(iterator); iter != MBB.end(); ++iter) {
          if (iter->getOpcode() != X86::ADDPDrr) {
            continue;
          }
          if (multiplicaton->getOperand(0).getReg() !=
                  iter->getOperand(1).getReg() &&
              multiplicaton->getOperand(0).getReg() !=
                  iter->getOperand(2).getReg()) {
            continue;
          }
          auto addition = &(*iter);
          if (findInstruction(MIvector, addition)) {
            continue;
          }
          bool used = false;
          for (auto iter2 = std::next(iterator); iter2 != iter; ++iter2) {
            auto instr = &(*iter2);
            for (auto &reg : instr->all_uses()) {
              if (reg.getReg() != addition->getOperand(1).getReg() &&
                  reg.getReg() != addition->getOperand(2).getReg()) {
                continue;
              }
              if (findInstruction(MIvector, instr)) {
                continue;
              }
              used = true;
              break;
            }
            if (used)
              break;
          }
          if (used)
            continue;
          MIMetadata MIMD(*iterator);
          MachineInstrBuilder MIB =
              BuildMI(MBB, multiplicaton, MIMD, TII.get(X86::VFMADD213PDr));
          MIB.addReg(addition->getOperand(0).getReg(), RegState::Define);
          MIB.addReg(multiplicaton->getOperand(1).getReg());
          MIB.addReg(multiplicaton->getOperand(2).getReg());
          if (multiplicaton->getOperand(0).getReg() ==
              iter->getOperand(1).getReg()) {
            MIB.addReg(addition->getOperand(2).getReg());
          } else if (multiplicaton->getOperand(0).getReg() ==
                     iter->getOperand(2).getReg()) {
            MIB.addReg(addition->getOperand(1).getReg());
          }
          if (!findInstruction(MIvector, multiplicaton)) {
            MIvector.push_back(multiplicaton);
          }
          if (!findInstruction(MIvector, addition)) {
            MIvector.push_back(addition);
          }
          if (multiplicaton->getOperand(0).getReg() ==
              addition->getOperand(0).getReg()) {
            break;
          }
          Changed = true;
        }
      }
    }
    for (auto &MI : MIvector) {
      MI->eraseFromParent();
    }
    return Changed;
  }

private:
  bool findInstruction(std::vector<MachineInstr *> &MIvector,
                       MachineInstr *instruction) {
    auto result{std::find(begin(MIvector), end(MIvector), instruction)};
    return (result != end(MIvector));
  }
};
} // end anonymous namespace

char X86VolodinEMulAddPass::ID = 0;

static RegisterPass<X86VolodinEMulAddPass>
    X("x86-volodin-replace-mul-add", "X86 replace mul and add with muladd pass",
      false, false);