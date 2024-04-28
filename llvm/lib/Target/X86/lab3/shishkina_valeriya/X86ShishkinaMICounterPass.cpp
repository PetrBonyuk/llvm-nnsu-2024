#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {
class X86ShishkinaMICounterPass : public MachineFunctionPass {
public:
  static char ID;

  X86ShishkinaMICounterPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &func) override {
    const TargetInstrInfo *targInstInf = func.getSubtarget().getInstrInfo();
    DebugLoc debugLocation = func.front().begin()->getDebugLoc();

    Module &mod = *func.getFunction().getParent();
    GlobalVariable *gVar = mod.getGlobalVariable("ic");

    if (!gVar) {
      LLVMContext &context = mod.getContext();
      gVar = new GlobalVariable(mod, IntegerType::get(context, 64), false,
                                GlobalValue::ExternalLinkage, nullptr, "ic");
    }

    for (auto &basicBl : func) {
      unsigned count = 0;
      for (auto &instr : basicBl) {
        if (!instr.isDebugInstr())
          ++count;
      }

      BuildMI(basicBl, basicBl.getFirstTerminator(), debugLocation,
              targInstInf->get(X86::ADD64ri32))
          .addGlobalAddress(gVar, 0, X86II::MO_NO_FLAG)
          .addImm(count);
    }

    return true;
  }
};
} // end anonymous namespace

char X86ShishkinaMICounterPass::ID = 0;
static RegisterPass<X86ShishkinaMICounterPass>
    X("x86-shishkina-mi-counter",
      "X86 Count number of machine instructions pass", false, false);
