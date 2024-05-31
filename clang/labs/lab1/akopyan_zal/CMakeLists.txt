add_llvm_library(AlwaysInlinePluginAkopyan MODULE AddAttrAlwaysInline.cpp PLUGIN_TOOL clang)

if(WIN32 OR CYGWIN)
  set(LLVM_LINK_COMPONENTS
    Support
  )
  clang_target_link_libraries(AlwaysInlinePluginAkopyan PRIVATE
    clangAST
    clangBasic
    clangFrontend
    )
endif()

set(CLANG_TEST_DEPS "AlwaysInlinePluginAkopyan" ${CLANG_TEST_DEPS} PARENT_SCOPE)
