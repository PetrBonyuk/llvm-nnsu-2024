#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class DeprecFuncVisitor : public RecursiveASTVisitor<DeprecFuncVisitor> {
private:
  ASTContext *ast_Context;

public:
  explicit DeprecFuncVisitor(ASTContext *ast_Context)
      : ast_Context(ast_Context) {}

  bool VisitFunctionDecl(FunctionDecl *Funct) {
    if (Funct->getNameInfo().getAsString().find("deprecated") !=
        std::string::npos) {
      DiagnosticsEngine &Diags = ast_Context->getDiagnostics();
      size_t CustomDiagID = Diags.getCustomDiagID(
          DiagnosticsEngine::Warning,
          "The function name contains the word 'deprecated'");
      Diags.Report(Funct->getLocation(), CustomDiagID)
          << Funct->getNameInfo().getAsString();
    }
    return true;
  }
};

class DeprecFuncConsumer : public clang::ASTConsumer {
private:
  CompilerInstance &Instance;

public:
  explicit DeprecFuncConsumer(CompilerInstance &CI) : Instance(CI) {}

  void HandleTranslationUnit(ASTContext &ast_Context) override {
    DeprecFuncVisitor Visitor(&Instance.getASTContext());
    Visitor.TraverseDecl(ast_Context.getTranslationUnitDecl());
  }
};

class DeprecFuncPlugin : public PluginASTAction {
protected:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<DeprecFuncConsumer>(Compiler);
  }

  bool ParseArgs(const CompilerInstance &Compiler,
                 const std::vector<std::string> &Args) override {
    return true;
  }
};

static FrontendPluginRegistry::Add<DeprecFuncPlugin> X("deprecated-warning",
                                                       "deprecated warning");