#include "X86.h"
#include "X86InstrInfo.h"
#include "X86Subtarget.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/Register.h"
#include <map>

using namespace llvm;

namespace {

class X86KulikovFMAPass : public MachineFunctionPass {
public:
  static char ID;
  X86KulikovFMAPass() : MachineFunctionPass(ID) {}

  bool runOnMachineFunction(MachineFunction &MF) override {
    bool Modified = false;
    const TargetInstrInfo *TII = MF.getSubtarget().getInstrInfo();

    for (auto &MBB : MF) {
      std::map<llvm::Register, std::pair<MachineInstr *, MachineInstr *>>
          useMap;
      for (auto I = MBB.begin(); I != MBB.end(); I++) {
        if (I->getOpcode() == X86::MULPDrr) {
          useMap.insert({I->getOperand(0).getReg(), {&(*I), nullptr}});
        } else {
          for (unsigned i = 1; i < I->getNumOperands(); i++) {
            llvm::MachineOperand &MO = I->getOperand(i);
            if (MO.isReg() && MO.getReg().isVirtual()) {
              auto r = MO.getReg();
              auto fnd = useMap.find(r);
              if (fnd != useMap.end()) {
                if (fnd->second.second) {
                  useMap.erase(r);
                } else {
                  useMap.insert_or_assign(
                      r, std::make_pair(fnd->second.first, &(*I)));
                }
              }
            }
          }
        }
      }
      for (auto u : useMap) {
        if (u.second.second && u.second.second->getOpcode() == X86::ADDPDrr) {
          auto &MulInstr = *u.second.first; // a = b * c;
          auto &AddInstr =
              *u.second
                   .second; // d = e(a) + f | d = e + f(a);  => d = b * c + e

          auto a = MulInstr.getOperand(0).getReg();
          auto b = MulInstr.getOperand(1).getReg();
          auto c = MulInstr.getOperand(2).getReg();
          auto d = AddInstr.getOperand(0).getReg();
          auto e = AddInstr.getOperand(1).getReg();
          auto f = AddInstr.getOperand(2).getReg();
          if (e == a) {
            e = f;
          }

          MIMetadata MIMD(AddInstr);
          BuildMI(MBB, AddInstr, MIMD,
                  TII->get(X86::VFMADD213PDr)) // or X86::VFMADDPD4rr
              .addReg(d, RegState::Define)
              .addReg(b)
              .addReg(c)
              .addReg(e);
          MulInstr.eraseFromParent();
          AddInstr.eraseFromParent();
          Modified = true;
        }
      }
    }

    return Modified;
  }
};

} // end of anonymous namespace

char X86KulikovFMAPass::ID = 0;
static RegisterPass<X86KulikovFMAPass> X("x86-kulikov-fma", "X86 Kulikov FMA",
                                         false, false);
