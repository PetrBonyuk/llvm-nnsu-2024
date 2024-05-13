#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include <algorithm>
#include <cctype>

using namespace clang;

class DeprecFuncVisitor : public RecursiveASTVisitor<DeprecFuncVisitor> {
private:
  ASTContext *ast_Context;
  bool CaseInsensitive;
  bool HelpRequested;

public:
  explicit DeprecFuncVisitor(ASTContext *ast_Context, bool CaseInsensitive,
                             bool HelpRequested)
      : ast_Context(ast_Context), CaseInsensitive(CaseInsensitive),
        HelpRequested(HelpRequested) {}

  bool VisitFunctionDecl(FunctionDecl *Funct) {
    if (HelpRequested) {
      return true;
    }

    std::string FuncName = Funct->getNameInfo().getAsString();
    std::string Deprecated = "deprecated";
    if (CaseInsensitive) {
      std::transform(FuncName.begin(), FuncName.end(), FuncName.begin(),
                     [](unsigned char c) { return std::tolower(c); });
    }

    if (FuncName.find(Deprecated) != std::string::npos) {
      DiagnosticsEngine &Diags = ast_Context->getDiagnostics();
      unsigned CustomDiagID = Diags.getCustomDiagID(
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
  bool CaseInsensitive;
  bool HelpRequested;

public:
  explicit DeprecFuncConsumer(CompilerInstance &CI, bool CaseInsensitive,
                              bool HelpRequested)
      : Instance(CI), CaseInsensitive(CaseInsensitive),
        HelpRequested(HelpRequested) {}

  void HandleTranslationUnit(ASTContext &ast_Context) override {
    DeprecFuncVisitor Visitor(&Instance.getASTContext(), CaseInsensitive,
                              HelpRequested);
    Visitor.TraverseDecl(ast_Context.getTranslationUnitDecl());
  }
};

class DeprecFuncPlugin : public PluginASTAction {
protected:
  bool CaseInsensitive = false;
  bool HelpRequested = false;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(CompilerInstance &Compiler,
                    llvm::StringRef InFile) override {
    return std::make_unique<DeprecFuncConsumer>(Compiler, CaseInsensitive,
                                                HelpRequested);
  }

  bool ParseArgs(const CompilerInstance &Compiler,
                 const std::vector<std::string> &Args) override {
    for (const auto &Arg : Args) {
      if (Arg == "help") {
        llvm::errs() << "DeprecFuncPlugin: Checks for deprecated functions in "
                        "the code.\n";
        HelpRequested = true;
        return true;
      } else if (Arg == "case-insensitive") {
        CaseInsensitive = true;
      }
    }
    return true;
  }
};

static FrontendPluginRegistry::Add<DeprecFuncPlugin> X("deprecated-warning",
                                                       "deprecated warning");