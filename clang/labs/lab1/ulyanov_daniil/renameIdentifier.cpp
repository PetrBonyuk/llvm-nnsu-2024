#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/Rewrite/Core/Rewriter.h"

enum class IdType { Var, Func, Class };

class RenameVisitor : public clang::RecursiveASTVisitor<RenameVisitor> {
public:
  explicit RenameVisitor(clang::Rewriter rewriter, IdType type,
                         clang::StringRef first_name,
                         clang::StringRef second_name)
      : rewriter(rewriter), type(type), first_name(first_name),
        second_name(second_name) {}

  bool VisitFunctionDecl(clang::FunctionDecl *func) {
    if (type == IdType::Func && func->getName() == first_name) {
      rewriter.ReplaceText(func->getNameInfo().getSourceRange(), second_name);
    }
    return true;
  }

  bool VisitCallExpr(clang::CallExpr *call) {
    if (type == IdType::Func) {
      clang::FunctionDecl *callee = call->getDirectCallee();
      if (callee && callee->getName() == first_name) {
        rewriter.ReplaceText(call->getCallee()->getSourceRange(), second_name);
      }
    }
    return true;
  }

  bool VisitVarDecl(clang::VarDecl *var) {
    if (type == IdType::Var && var->getName() == first_name) {
      rewriter.ReplaceText(var->getLocation(), first_name.size(), second_name);
    }
    if (type == IdType::Class &&
        var->getType().getAsString() == first_name + " *") {
      rewriter.ReplaceText(var->getTypeSourceInfo()->getTypeLoc().getBeginLoc(),
                           first_name.size(), second_name);
    }
    if (type == IdType::Class && var->getType().getAsString() == first_name) {
      rewriter.ReplaceText(
          var->getTypeSourceInfo()->getTypeLoc().getSourceRange(), second_name);
    }
    return true;
  }

  bool VisitDeclRefExpr(clang::DeclRefExpr *expr) {
    clang::VarDecl *var = clang::dyn_cast<clang::VarDecl>(expr->getDecl());
    if (type == IdType::Var && var && var->getName() == first_name) {
      rewriter.ReplaceText(expr->getSourceRange(), second_name);
    }
    return true;
  }

  bool VisitCXXRecordDecl(clang::CXXRecordDecl *record) {
    if (type == IdType::Class && record->getName() == first_name) {
      rewriter.ReplaceText(record->getLocation(), second_name);
      const auto *destructor = record->getDestructor();
      if (destructor) {
        rewriter.ReplaceText(destructor->getLocation(), first_name.size() + 1,
                             '~' + second_name);
      }
    }
    return true;
  }

  bool VisitCXXConstructorDecl(clang::CXXConstructorDecl *constructor) {
    if (type == IdType::Class) {
      if (constructor->getNameAsString() == first_name) {
        rewriter.ReplaceText(constructor->getLocation(), first_name.size(),
                             second_name);
      }
    }
    return true;
  }

  bool VisitCXXNewExpr(clang::CXXNewExpr *newExpr) {
    if (type == IdType::Class) {
      if (newExpr->getConstructExpr()->getType().getAsString() == first_name) {
        rewriter.ReplaceText(newExpr->getExprLoc(), first_name.size() + 4,
                             "new " + second_name);
      }
    }
    return true;
  }

  bool save_changes() { return rewriter.overwriteChangedFiles(); }

private:
  clang::Rewriter rewriter;
  IdType type;
  std::string first_name;
  std::string second_name;
};

class RenameASTConsumer : public clang::ASTConsumer {
public:
  explicit RenameASTConsumer(clang::CompilerInstance &CI, IdType type,
                             clang::StringRef first_name,
                             clang::StringRef second_name)
      : Visitor(clang::Rewriter(CI.getSourceManager(), CI.getLangOpts()), type,
                first_name, second_name) {}

  void HandleTranslationUnit(clang::ASTContext &context) override {
    Visitor.TraverseDecl(context.getTranslationUnitDecl());
    if (Visitor.save_changes()) {
      llvm::errs() << "error with saving file!\n";
    }
  }

private:
  RenameVisitor Visitor;
};

class RenamePlugin : public clang::PluginASTAction {
public:
  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &CI,
                    clang::StringRef InFile) override {
    return std::make_unique<RenameASTConsumer>(CI, type, first_name,
                                               second_name);
  }

protected:
  bool ParseArgs(const clang::CompilerInstance &CI,
                 const std::vector<std::string> &args) override {
    std::vector<std::pair<std::string, std::string>> params = {
        {"type=", ""}, {"first-name=", ""}, {"second-name=", ""}};

    if (!args.empty() && args[0] == "help") {
      llvm::errs()
          << "Specify three required arguments:\n"
             "-plugin-arg-rename type=[\"variable\", \"function\", \"class\"]\n"
             "-plugin-arg-rename first-name=\"Current identifier name\"\n"
             "-plugin-arg-rename second-name=\"New identifier name\"\n";
      return true;
    }

    if (args.size() < 3) {
      PrintParamsError(CI);
      return false;
    }

    for (const auto &arg : args) {
      bool is_found = false;
      for (auto &param : params) {
        if (arg.find(param.first) == 0 && param.second.empty()) {
          param.second = arg.substr(param.first.size());
          is_found = true;
          break;
        }
      }
      if (!is_found) {
        PrintParamsError(CI);
        return false;
      }
    }

    std::vector<std::pair<std::string, IdType>> id_type = {
        {"variable", IdType::Var},
        {"function", IdType::Func},
        {"class", IdType::Class}};
    size_t i;
    for (i = 0; i < id_type.size(); i++) {
      if (params[0].second == id_type[i].first) {
        type = id_type[i].second;
        break;
      }
    }
    if (i == id_type.size()) {
      PrintParamsError(CI);
      return false;
    }
    first_name = params[1].second;
    second_name = params[2].second;
    return true;
  }

  void PrintParamsError(const clang::CompilerInstance &CI) {
    clang::DiagnosticsEngine &D = CI.getDiagnostics();

    D.Report(
        D.getCustomDiagID(clang::DiagnosticsEngine::Error,
                          "Invalid arguments\n"
                          "Specify \"-plugin-arg-rename help\" for usage\n"));
  }

private:
  IdType type;
  std::string first_name;
  std::string second_name;
};

static clang::FrontendPluginRegistry::Add<RenamePlugin>
    X("rename", "Rename variable, function or class");