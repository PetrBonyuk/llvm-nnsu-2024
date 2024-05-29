add_llvm_library(IsaevAlwaysInlinePlugin MODULE AlwaysInline.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(IsaevAlwaysInlinePlugin PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "IsaevAlwaysInlinePlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)