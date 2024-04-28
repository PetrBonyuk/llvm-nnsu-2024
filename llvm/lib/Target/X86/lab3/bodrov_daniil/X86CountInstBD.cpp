#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"

using namespace llvm;

#define MY_COUNTINSTRUCTIONS_PASS_NAME "x86-count-machine-instructions"
#define MY_COUNTINSTRUCTIONS_PASS_DESC "X86 Count Machine Instructions Pass"

namespace {
class X86BodrovCountInstructionsPass : public MachineFunctionPass {
public:
  static char ID;
  X86BodrovCountInstructionsPass() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
};

char X86BodrovCountInstructionsPass::ID = 0;

bool X86BodrovCountInstructionsPass::runOnMachineFunction(MachineFunction &MF) {
  // Get the global variable to store the counter
  Module *M = MF.getFunction().getParent();
  GlobalVariable *CounterVar = M->getGlobalVariable("ic");
  if (!CounterVar) {
    // If global variable doesn't exist, create it
    CounterVar =
        new GlobalVariable(*M, IntegerType::get(M->getContext(), 64), false,
                           GlobalValue::ExternalLinkage, nullptr, "ic");
  }

  // Get the target instruction info
  const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
  const TargetRegisterInfo *TRI = MF.getSubtarget().getRegisterInfo();

  DebugLoc DL3 = MF.front().begin()->getDebugLoc();

  // Iterate over all basic blocks in the function
  for (auto &MBB : MF) {
    unsigned InstructionCount = std::distance(MBB.begin(), MBB.end());
    auto InsertPt = MBB.getFirstTerminator();

    for (auto &MI : MBB) {
      // Check if the instruction modifies or reads the EFLAGS register
      if (MI.modifiesRegister(X86::EFLAGS, TRI) ||
          MI.readsRegister(X86::EFLAGS, TRI)) {
        if (MI.isCompare()) {
          InsertPt = &MI;
        }
      }
    }

    // Update the counter
    BuildMI(MBB, InsertPt, DL3, TII->get(X86::ADD64mi32))
        .addReg(0)
        .addImm(1)
        .addReg(0)
        .addGlobalAddress(CounterVar)
        .addReg(0)
        .addImm(InstructionCount);
  }

  return true;
}

} // namespace

static RegisterPass<X86BodrovCountInstructionsPass>
    X(MY_COUNTINSTRUCTIONS_PASS_NAME, MY_COUNTINSTRUCTIONS_PASS_DESC, false,
      false);