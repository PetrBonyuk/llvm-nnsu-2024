#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"

using namespace llvm;

#define X86_MULADD_PASS_NAME "X86 Petrov muladd pass"

namespace {
class X86PetrovMulAddPass : public MachineFunctionPass {
public:
  static char ID;

  X86PetrovMulAddPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override;

  StringRef getPassName() const override { return X86_MULADD_PASS_NAME; }
};

char X86PetrovMulAddPass::ID = 0;

bool X86PetrovMulAddPass::runOnMachineFunction(llvm::MachineFunction &MF) {
  const llvm::TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  bool Changed = false;

  for (auto &MBB : MF) {
    for (auto I = MBB.begin(); I != MBB.end();) {
      if (I->getOpcode() != llvm::X86::MULPDrr) {
        ++I;
        continue;
      }
      auto MulInstr = I;
      llvm::Register MulDestReg = MulInstr->getOperand(0).getReg();

      ++I;
      for (; I != MBB.end(); ++I) {
        if (I->getOpcode() == llvm::X86::ADDPDrr &&
            (I->getOperand(1).getReg() == MulDestReg ||
             I->getOperand(2).getReg() == MulDestReg)) {

          auto MIB =
              BuildMI(MBB, I, I->getDebugLoc(),
                      TII->get(llvm::X86::VFMADD213PDr))
                  .addReg(I->getOperand(0).getReg(), llvm::RegState::Define)
                  .addReg(MulInstr->getOperand(1).getReg())
                  .addReg(MulInstr->getOperand(2).getReg())
                  .addReg((I->getOperand(1).getReg() == MulDestReg)
                              ? I->getOperand(2).getReg()
                              : I->getOperand(1).getReg());

          I = MBB.erase(I);
          MBB.erase(MulInstr);

          Changed = true;
          break;
        }
      }
      if (!Changed && I != MBB.end()) {
        ++I;
      }
    }
  }
  return Changed;
}

} // namespace

static RegisterPass<X86PetrovMulAddPass> X("x86-petrov-muladd",
                                           X86_MULADD_PASS_NAME, false, false);
