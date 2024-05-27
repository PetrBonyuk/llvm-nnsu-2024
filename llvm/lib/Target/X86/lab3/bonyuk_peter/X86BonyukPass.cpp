#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Passes/PassPlugin.h"

using namespace llvm;

namespace {
	class X86BonyukPass : public MachineFunctionPass {
	public:
		static char ID;
		X86BonyukPass() : MachineFunctionPass(ID) {}

		bool runOnMachineFunction(MachineFunction &MF) override {
			const TargetInstrInfo *TII= MF.getSubtarget().getInstrInfo();
			DebugLoc debugLocation = MF.front().begin()->getDebugLoc();

			Module &M = *MF.getFunction().getParent();
			GlobalVariable *GVar = M.getGlobalVariable("ic");

			if (!gVar) {
				LLVMContext &context = M.getContext();
				GVar = new GlobalVariable(M, IntegerType::get(context, 64), false,
					GlobalValue::ExternalLinkage, nullptr, "ic");
			}

			for (auto &MBB : MF) {
				unsigned count = 0;
				for (auto &MI : MBB) {
					if (!MI.isDebugInstr())
						++count;
				}

				BuildMI(MBB, MBB.getFirstTerminator(), debugLocation,
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