#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include <utility>
#include <vector>

using namespace llvm;

namespace {
class X86KozyrevaPass : public MachineFunctionPass {
public:
  static char ID;
  X86KozyrevaPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
};

char X86KozyrevaPass::ID = 0;

bool X86KozyrevaPass::runOnMachineFunction(MachineFunction &pFunction) {
  const TargetInstrInfo *instrInfo = pFunction.getSubtarget().getInstrInfo();
  bool isModified = false;
  std::vector<std::pair<MachineInstr *, MachineInstr *>> toDelete;

  for (auto &basicBlock : pFunction) {
    MachineInstr *mulInstruction = nullptr;
    MachineInstr *addInstruction = nullptr;

    for (auto &instr : basicBlock) {
      if (instr.getOpcode() == X86::MULPDrr) {
        mulInstruction = &instr;

        for (auto next = std::next(instr.getIterator());
             next != basicBlock.end(); ++next) {
          if (next->getOpcode() == X86::ADDPDrr) {
            addInstruction = &*next;
            if (mulInstruction->getOperand(0).getReg() ==
                addInstruction->getOperand(1).getReg()) {
              toDelete.emplace_back(mulInstruction, addInstruction);
              isModified = true;
              break;
            }
          } else if (next->definesRegister(
                         mulInstruction->getOperand(0).getReg())) {
            break;
          }
        }
      }
    }
  }

  for (const auto &[mulInstr, addInstr] : toDelete) {
    MachineInstrBuilder builder =
        BuildMI(*mulInstr->getParent(), *mulInstr, mulInstr->getDebugLoc(),
                instrInfo->get(X86::VFMADD213PDZ128r));
    builder.addReg(addInstr->getOperand(0).getReg(), RegState::Define);
    builder.addReg(mulInstr->getOperand(1).getReg());
    builder.addReg(mulInstr->getOperand(2).getReg());
    builder.addReg(addInstr->getOperand(2).getReg());
    mulInstr->eraseFromParent();
    addInstr->eraseFromParent();
  }

  return isModified;
}
} // namespace

static RegisterPass<X86KozyrevaPass> X("x86-kozyreva-pass", "X86 Kozyreva Pass",
                                       false, false);
