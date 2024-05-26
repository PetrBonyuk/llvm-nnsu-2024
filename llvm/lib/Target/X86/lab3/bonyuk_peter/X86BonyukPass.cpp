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
			GlobalVariable *GVar = M->getGlobalVariable("ic");

			if (!GVar) {
				GVar = new GlobalVariable(*M, Type::getInt64Ty(Ctx), false,
					GlobalValue::ExternalLinkage, nullptr, "ic");
				GVar->setInitializer(ConstantInt::get(Type::getInt64Ty(Ctx), 0));
			}

			for (auto &MBB : MF) {
				unsigned count = 0;
				for (auto &MI : MBB) {
					if (!MI.isDebugInstr())
						++count;
				}

				BuildMI(MBB, MBB.getFirstTerminator(), MI.getDebugLoc(),
					TII->get(X86::ADD64ri32))
					.addGlobalAddress(GVar, 0, X86II::MO_NO_FLAG)
					.addImm(count);
			}

			return true;
		}
	};

	char X86BonyukPass::ID = 0;
} // namespace

static RegisterPass<X86BonyukPass> X("x86-bonyuk-pass", "X86 Bonyuk Pass", false, false);