#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"

using namespace llvm;

#define PASS_NAME "x86-kulaev-increm-counter-pass"

namespace {
class X86KulaevIncremCounterPass : public MachineFunctionPass {
public:
  static char ID;
  X86KulaevIncremCounterPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MachineFunctionRunning) override;

private:
  void updateCount(DebugLoc dl, MachineFunction &mf, const TargetInstrInfo *ti);
};
} // namespace

bool X86KulaevIncremCounterPass::runOnMachineFunction(
    MachineFunction &MachineFunctionRunning) {
  const TargetInstrInfo *TargetInstructionInfo =
      MachineFunctionRunning.getSubtarget().getInstrInfo();
  DebugLoc DL3 = MachineFunctionRunning.front().begin()->getDebugLoc();

  updateCount(DL3, MachineFunctionRunning, TargetInstructionInfo);

  return true;
}

void X86KulaevIncremCounterPass::updateCount(DebugLoc dl, MachineFunction &mf,
                                             const TargetInstrInfo *ti) {
  for (auto &MBasicBlock : mf) {
    unsigned count = 0;
    for (auto &MInstruction : MBasicBlock) {
      ++count;
    }

    // updating the counter
    BuildMI(MBasicBlock, MBasicBlock.getFirstTerminator(), dl,
            ti->get(X86::ADD64ri32))
        .addImm(count)
        .addExternalSymbol("ic");
  }
}

char X86KulaevIncremCounterPass::ID = 0;

static RegisterPass<X86KulaevIncremCounterPass>
    X(PASS_NAME, "Сounts the number of executed instructions in our function",
      false, false);