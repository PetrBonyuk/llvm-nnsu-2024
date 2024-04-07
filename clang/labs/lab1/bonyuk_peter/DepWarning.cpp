#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"

using namespace clang;

class DeprecFuncVisitor : public RecursiveASTVisitor<DeprecFuncVisitor> {
private:
  ASTContext *astContext;

public:
  explicit DeprecFuncVisitor(ASTContext *astContext) : astContext(astContext) {}

  bool FunctionDeclVisit(FunctionDecl *Funct) {
    if (Funct->getNameInfo().getAsString().find("deprecated") !=
      std::string::npos) {
      DiagnosticsEngine &Diags = astContext->getDiagnostics();
      size_t CustomDiagID =
        Diags.getCustomDiagID(DiagnosticsEngine::Warning,
          "The 'deprecated' is in the function name");
        Diags.Report(Funct->getLocation(), CustomDiagID)
        << Funct->getNameInfo().getAsString();
    }
    return true;
  }
};

class DeprecFuncConsumer : public ASTConsumer {
private:
  CompilerInstance &Instance;

public:
  explicit DeprecFuncConsumer(CompilerInstance &CI) : Instance(CI) {}

  void HandleTranslationUnit(ASTContext &astContext) override {
    DeprecFuncVisitor Visitor(&Instance.getASTContext());
    Visitor.TraverseDecl(astContext.getTranslationUnitDecl());
  }
};


class DeprecFuncPlugin : public PluginASTAction {
protected:
  std::unique_ptr<ASTConsumer>
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