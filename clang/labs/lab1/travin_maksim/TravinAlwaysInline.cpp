#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/AST/Stmt.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

class InlineVisitor : public clang::RecursiveASTVisitor<InlineVisitor> {
private:
  clang::ASTContext *Context;
  bool HasStatement;

public:
  explicit InlineVisitor(clang::ASTContext *Context)
      : Context(Context), HasStatement(false) {}

  bool VisitFunctionDecl(clang::FunctionDecl *Func) {
    HasStatement = false;
    TraverseStmt(Func->getBody());
    if (Func->hasAttr<clang::AlwaysInlineAttr>()) {
      return true;
    }
    if (!HasStatement) {
      clang::SourceRange Range = Func->getSourceRange();
      Func->addAttr(clang::AlwaysInlineAttr::Create(*Context, Range));
    }
    return true;
  }

  bool VisitStmt(clang::Stmt *stmt) {
    if (clang::isa<clang::IfStmt>(stmt) || clang::isa<clang::WhileStmt>(stmt) ||
        clang::isa<clang::ForStmt>(stmt) || clang::isa<clang::DoStmt>(stmt) ||
        clang::isa<clang::SwitchStmt>(stmt)) {
      HasStatement = true;
    }
    return true;
  }
};

class InlineConsumer : public clang::ASTConsumer {
private:
  InlineVisitor Visitor;

public:
  InlineConsumer(clang::ASTContext *Context) : Visitor(Context) {}

  bool HandleTopLevelDecl(clang::DeclGroupRef Group) override {
    for (clang::Decl *Decl : Group) {
      Visitor.TraverseDecl(Decl);
    }
    return true;
  }
};

class InlinePlugin : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &Compiler,
                    llvm::StringRef) override {
    return std::make_unique<InlineConsumer>(&Compiler.getASTContext());
  }
  bool ParseArgs(const clang::CompilerInstance &Compiler,
                 const std::vector<std::string> &Args) override {
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<InlinePlugin>
    X("add-always-inline",
      "adds always_inline to functions without if statements.");