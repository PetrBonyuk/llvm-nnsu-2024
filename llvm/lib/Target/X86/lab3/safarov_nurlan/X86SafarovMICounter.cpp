#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

using namespace llvm;

namespace {

class X86SafarovMICounter : public MachineFunctionPass {
public:
  static char ID;
  X86SafarovMICounter() : MachineFunctionPass(ID) {}
  bool
  runOnMachineFunction(llvm::MachineFunction &objectMachineFunction) override {
    auto *globalVariable =
        objectMachineFunction.getFunction().getParent()->getNamedGlobal("ic");
    if (!globalVariable) {
      return false;
    }

    auto debugLocation = objectMachineFunction.front().begin()->getDebugLoc();
    auto *targetInstructionInfo =
        objectMachineFunction.getSubtarget().getInstrInfo();

    for (auto &machineBasicBlock : objectMachineFunction) {
      int cnt =
          std::distance(machineBasicBlock.begin(), machineBasicBlock.end());
      auto position = machineBasicBlock.getFirstTerminator();
      if (position != machineBasicBlock.end() &&
          position != machineBasicBlock.begin() &&
          position->getOpcode() >= X86::JCC_1 &&
          position->getOpcode() <= X86::JCC_4) {
        --position;
      }
      BuildMI(machineBasicBlock, position, debugLocation,
              targetInstructionInfo->get(X86::ADD64mi32))
          .addReg(0)
          .addImm(1)
          .addReg(0)
          .addGlobalAddress(globalVariable)
          .addReg(0)
          .addImm(cnt);
    }
    return true;
  }
};
} // namespace

char X86SafarovMICounter::ID = 0;
static RegisterPass<X86SafarovMICounter>
    X("x86-safarov-mi-counter",
      "Counter of machine instructions executed during the execution of "
      "functions",
      false, false);