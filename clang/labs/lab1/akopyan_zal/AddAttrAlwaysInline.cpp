#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

namespace {
class AlwaysInlineVisitor
    : public clang::RecursiveASTVisitor<AlwaysInlineVisitor> {
public:
  explicit AlwaysInlineVisitor(clang::ASTContext *context_)
      : context(context_) {}
  bool VisitFunctionDecl(clang::FunctionDecl *decl) {
    if (decl->isFunctionOrFunctionTemplate() &&
        !decl->hasAttr<clang::AlwaysInlineAttr>()) {
      auto body = decl->getBody();
      if (body && !containsConditionalStatement(body)) {
        auto loc = decl->getSourceRange();
        if (loc.isValid()) {
          decl->addAttr(
              clang::AlwaysInlineAttr::Create(*context, loc.getBegin()));
        }
      }
    }
    return true;
  }

private:
  bool containsConditionalStatement(clang::Stmt *S) {
    if (!S) {
      return false;
    }

    if (llvm::isa<clang::IfStmt>(S) || llvm::isa<clang::SwitchStmt>(S) ||
        llvm::isa<clang::WhileStmt>(S) || llvm::isa<clang::DoStmt>(S) ||
        llvm::isa<clang::ForStmt>(S)) {
      return true;
    }

    for (clang::Stmt *Child : S->children()) {
      if (containsConditionalStatement(Child)) {
        return true;
      }
    }

    return false;
  }

private:
  clang::ASTContext *context;
};

class AlwaysInlineConsumer : public clang::ASTConsumer {
public:
  explicit AlwaysInlineConsumer(clang::ASTContext *сontext)
      : visitor(сontext) {}

  bool HandleTopLevelDecl(clang::DeclGroupRef groupDecl) override {
    for (clang::Decl *Decl : groupDecl) {
      visitor.TraverseDecl(Decl);
    }
    return true;
  }

private:
  AlwaysInlineVisitor visitor;
};

class AlwaysInlinePlugin final : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci, llvm::StringRef) override {
    return std::make_unique<AlwaysInlineConsumer>(&ci.getASTContext());
  }

  bool ParseArgs(const clang::CompilerInstance &ci,
                 const std::vector<std::string> &args) override {
    for (const auto &arg : args) {
      if (arg == "--help") {
        llvm::outs() << "Applies the always_inline attribute to functions that "
                        "don't contain conditional statements\n";
      }
    }
    return true;
  }
};
} // namespace

static clang::FrontendPluginRegistry::Add<AlwaysInlinePlugin>
    X("always-inline", "Applies the always_inline attribute to functions that "
                       "don't contain conditional statements");
