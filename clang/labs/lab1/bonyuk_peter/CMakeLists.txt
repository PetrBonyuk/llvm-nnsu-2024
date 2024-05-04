add_llvm_library(depWarningPluginBonyuk MODULE DepWarning.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
      Support
  )
  clang_target_link_libraries(depWarningPluginBonyuk PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "depWarningPluginBonyuk" ${CLANG_TEST_DEPS} PARENT_SCOPE)