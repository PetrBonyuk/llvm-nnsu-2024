#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceLocation.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/ADT/StringRef.h"

namespace {

class AlwaysInlineVisitor
    : public clang::RecursiveASTVisitor<AlwaysInlineVisitor> {
public:
  bool VisitStmt(clang::Stmt *St) {
    if (clang::isa<clang::IfStmt>(St) || clang::isa<clang::WhileStmt>(St) ||
        clang::isa<clang::ForStmt>(St) || clang::isa<clang::DoStmt>(St) ||
        clang::isa<clang::SwitchStmt>(St)) {
      hasCondition = true;
    }
    return true;
  }

  bool hasCondition = false;
};

class AlwaysInlineConsumer : public clang::ASTConsumer {
public:
  bool HandleTopLevelDecl(clang::DeclGroupRef DeclGroup) override {
    for (clang::Decl *Decl : DeclGroup) {
      if (auto FuncDecl = clang::dyn_cast<clang::FunctionDecl>(Decl)) {
        if (FuncDecl->getAttr<clang::AlwaysInlineAttr>()) {
          continue;
        }
        clang::Stmt *Body = FuncDecl->getBody();
        if (Body != nullptr) {
          AlwaysInlineVisitor Visitor;
          Visitor.TraverseStmt(Body);
          if (!Visitor.hasCondition) {
            clang::SourceLocation Location(
                FuncDecl->getSourceRange().getBegin());
            clang::SourceRange Range(Location);
            FuncDecl->addAttr(clang::AlwaysInlineAttr::Create(
                FuncDecl->getASTContext(), Range));
          }
        }
      }
    }
    return true;
  }
};

class AlwaysInlinePlugin : public clang::PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<AlwaysInlineConsumer>();
  }
  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &Args) override {
    return true;
  }
};

} // namespace

static clang::FrontendPluginRegistry::Add<AlwaysInlinePlugin>
    X("add-always-inline", "Add the always_inline attribute to all functions "
                           "that don't have conditions");