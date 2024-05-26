#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"

using namespace llvm;

namespace {
	class X86BonyukPass : public MachineFunctionPass {
	public:
		static char ID;
		X86BonyukPass() : MachineFunctionPass(ID) {}

		bool runOnMachineFunction(MachineFunction &MF) override {
			const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();
			Module *M = MF.getFunction().getParent();
			LLVMContext &Ctx = M->getContext();
				// Create a global variable 'ic' to store the instruction count
				GlobalVariable *GVar = new GlobalVariable(*M, Type::getInt64Ty(Ctx), false, GlobalValue::ExternalLinkage, nullptr, "ic");
			GVar->setInitializer(ConstantInt::get(Type::getInt64Ty(Ctx), 0));

			for (auto &MBB : MF) {
				for (auto &MI : MBB) {
					// Increment the global variable 'ic' for each instruction
					BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(X86::ADD64ri8), X86::RAX)
						.addReg(X86::RAX)
						.addImm(1);
					BuildMI(MBB, MI, MI.getDebugLoc(), TII->get(X86::MOV64mr))
						.addReg(X86::RIP)
						.addImm(0)
						.addReg(0)
						.addGlobalAddress(GVar)
						.addReg(0)
						.addReg(X86::RAX);
				}
			}

			return true;
		}
	};

	char X86BonyukPass::ID = 0;
} // namespace

static RegisterPass<X86BonyukPass> X("x86-bonyuk-pass", "X86 Bonyuk Pass", false, false);