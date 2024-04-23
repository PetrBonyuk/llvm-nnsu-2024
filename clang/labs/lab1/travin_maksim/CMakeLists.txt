add_llvm_library(TravinAlwaysInlinePlugin MODULE TravinAlwaysInline.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(TravinAlwaysInlinePlugin PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "TravinAlwaysInlinePlugin" ${CLANG_TEST_DEPS} PARENT_SCOPE)