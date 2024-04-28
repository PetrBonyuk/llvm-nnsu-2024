#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include <utility>
#include <vector>

using namespace llvm;

namespace {
class X86MulAddPass : public MachineFunctionPass {
public:
  static char ID;
  X86MulAddPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
};

char X86MulAddPass::ID = 0;

bool X86MulAddPass::runOnMachineFunction(MachineFunction &machineFunc) {
  const TargetInstrInfo *instrInfo = machineFunc.getSubtarget().getInstrInfo();
  bool changed = false;

  std::vector<std::pair<MachineInstr *, MachineInstr *>> deletedInstructions;

  for (auto &block : machineFunc) {
    MachineInstr *mulInstr = nullptr;
    MachineInstr *addInstr = nullptr;

    for (auto op = block.begin(); op != block.end(); ++op) {
      if (op->getOpcode() == X86::MULPDrr) {
        mulInstr = &(*op);

        for (auto opNext = std::next(op); opNext != block.end(); ++opNext) {
          if (opNext->getOpcode() == X86::ADDPDrr) {
            addInstr = &(*opNext);

            if (mulInstr->getOperand(0).getReg() ==
                addInstr->getOperand(1).getReg()) {
              deletedInstructions.emplace_back(mulInstr, addInstr);
              changed = true;
              break;
            }
          } else if (opNext->definesRegister(mulInstr->getOperand(0).getReg()))
            break;
        }
      }
    }
  }

  for (auto &[mulInstr, addInstr] : deletedInstructions) {
    MachineInstrBuilder MIB =
        BuildMI(*mulInstr->getParent(), *mulInstr, mulInstr->getDebugLoc(),
                instrInfo->get(X86::VFMADD213PDZ128r));

    MIB.addReg(addInstr->getOperand(0).getReg(), RegState::Define);
    MIB.addReg(mulInstr->getOperand(1).getReg());
    MIB.addReg(mulInstr->getOperand(2).getReg());
    MIB.addReg(addInstr->getOperand(2).getReg());

    mulInstr->eraseFromParent();
    addInstr->eraseFromParent();
  }

  return changed;
}
} // namespace

static RegisterPass<X86MulAddPass> X("x86-muladd", "X86 muladd pass", false,
                                     false);
