#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

namespace {
class AddAttrAlwaysInlineVisitor final
    : public clang::RecursiveASTVisitor<AddAttrAlwaysInlineVisitor> {
public:
  explicit AddAttrAlwaysInlineVisitor(clang::ASTContext *context_)
      : context(context_) {}
  bool VisitFunctionDecl(clang::FunctionDecl *decl) {
    if (!decl->isFunctionOrFunctionTemplate())
      return true;
    if (decl->hasAttr<clang::AlwaysInlineAttr>())
      return true;
    if (auto body = decl->getBody()) {
      if (findСonditionalStatement(body))
        return true;
      if (auto loc = decl->getSourceRange(); loc.isValid())
        decl->addAttr(
            clang::AlwaysInlineAttr::Create(*context, loc.getBegin()));
    }
    return true;
  }

private:
  bool findСonditionalStatement(clang::Stmt *statement) {
    if (!statement)
      return false;

    if (clang::isa<clang::IfStmt>(statement) ||
        clang::isa<clang::SwitchStmt>(statement) ||
        clang::isa<clang::DoStmt>(statement) ||
        clang::isa<clang::WhileStmt>(statement) ||
        clang::isa<clang::ForStmt>(statement))
      return true;

    if (auto parent = clang::dyn_cast<clang::CompoundStmt>(statement)) {
      for (auto child : parent->body()) {
        if (findСonditionalStatement(child))
          return true;
      }
    }
    return false;
  }

private:
  clang::ASTContext *context;
};

class AddAttrAlwaysInlineConsumer final : public clang::ASTConsumer {
public:
  explicit AddAttrAlwaysInlineConsumer(clang::ASTContext *сontext)
      : visitor(сontext) {}

  bool HandleTopLevelDecl(clang::DeclGroupRef groupDecl) override {
    for (clang::Decl *Decl : groupDecl) {
      visitor.TraverseDecl(Decl);
    }
    return true;
  }

private:
  AddAttrAlwaysInlineVisitor visitor;
};

class AddAttrAlwaysInlineAction final : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci, llvm::StringRef) override {
    return std::make_unique<AddAttrAlwaysInlineConsumer>(&ci.getASTContext());
  }

  bool ParseArgs(const clang::CompilerInstance &ci,
                 const std::vector<std::string> &args) override {
    for (const auto &arg : args) {
      if (arg == "--help") {
        llvm::outs() << "Adds the always_inline attribute to functions without "
                        "conditions\n";
      }
    }
    return true;
  }
};
} // namespace

static clang::FrontendPluginRegistry::Add<AddAttrAlwaysInlineAction>
    X("add_attr_always_inline_plugin",
      "Adds the always_inline attribute to functions without conditions");
