#include "X86.h"
#include "X86InstrInfo.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
using namespace llvm;
namespace {

class X86KosarevInstCounter : public MachineFunctionPass {
public:
  static char ID;
  X86KosarevInstCounter() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(llvm::MachineFunction &MF) override;
};
} // namespace

char X86KosarevInstCounter::ID = 0;

bool X86KosarevInstCounter::runOnMachineFunction(llvm::MachineFunction &MF) {
  // Get the global variable 'ic'
  auto *GV = MF.getFunction().getParent()->getNamedGlobal("ic");
  if (!GV) {
    // If 'ic' does not exist, create it
    LLVMContext &context = MF.getFunction().getParent()->getContext();
    GV = new GlobalVariable(*MF.getFunction().getParent(),
                            IntegerType::get(context, 64), false,
                            GlobalValue::ExternalLinkage, nullptr, "ic");
    GV->setAlignment(Align(8));
    // If the global variable 'ic' cannot be created, return false
    if (!GV) {
      return false;
    }
  }
  auto DL = MF.front().begin()->getDebugLoc();
  auto *TII = MF.getSubtarget().getInstrInfo();
  for (auto &MBB : MF) {
    // Count the number of instructions in the basic block
    int Counter = std::distance(MBB.begin(), MBB.end());
    auto Place = MBB.getFirstTerminator();
    if (Place != MBB.end() && Place != MBB.begin() &&
        Place->getOpcode() >= X86::JCC_1 && Place->getOpcode() <= X86::JCC_4) {
      --Place;
    }
    // Create a new instruction that adds the Counter value to the global
    // variable 'ic'
    BuildMI(MBB, Place, DL, TII->get(X86::ADD64mi32))
        .addReg(0)
        .addImm(1)
        .addReg(0)
        .addGlobalAddress(GV)
        .addReg(0)
        .addImm(Counter);
  }
  return true;
}

static RegisterPass<X86KosarevInstCounter>
    X("x86-kosarev-instruction-counter", "Instruction counter", false, false);