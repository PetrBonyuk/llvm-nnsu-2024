#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "llvm/Support/raw_ostream.h"

class ClassPrinter : public clang::RecursiveASTVisitor<ClassPrinter> {
public:
  bool VisitCXXRecordDecl(clang::CXXRecordDecl *declaration) {
    if (declaration->isClass() || declaration->isStruct()) {
      printClassName(declaration);
      printClassFields(declaration);
      llvm::outs() << "\n";
    }
    return true;
  }

private:
  void printClassName(clang::CXXRecordDecl *declaration) {
    llvm::outs() << declaration->getNameAsString() << "\n";
  }

  void printClassFields(clang::CXXRecordDecl *declaration) {
    for (auto field_member : declaration->fields()) {
      llvm::outs() << "|_" << field_member->getNameAsString() << "\n";
    }
  }
};

class ClassPrinterASTConsumer : public clang::ASTConsumer {
public:
  ClassPrinter Visitor;
  void HandleTranslationUnit(clang::ASTContext &Context) override {
    Visitor.TraverseDecl(Context.getTranslationUnitDecl());
  }
};

class ClassPrinterPluginAction : public clang::PluginASTAction {
public:
  bool helpFlag = false;

  std::unique_ptr<clang::ASTConsumer>
  CreateASTConsumer(clang::CompilerInstance &ci, llvm::StringRef) override {
    if (helpFlag) {
      return nullptr;
    }
    return std::make_unique<ClassPrinterASTConsumer>();
  }

  bool ParseArgs(const clang::CompilerInstance &ci,
                 const std::vector<std::string> &Args) override {
    for (const auto &arg : Args) {
      if (arg == "--help") {
        llvm::outs() << "This plugin traverses the Abstract Syntax Tree (AST) "
                        "of a codebase and prints the name and fields of each "
                        "class it encounters\n";
        helpFlag = true;
      }
    }
    return true;
  }
};

static clang::FrontendPluginRegistry::Add<ClassPrinterPluginAction>
    X("classprinter", "Prints all members of the class");
