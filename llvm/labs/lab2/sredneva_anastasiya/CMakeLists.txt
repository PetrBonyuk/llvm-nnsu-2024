set(Plugin_Name "sredneva_anastasiya_loop_wrapper_pass_plugin")

if (NOT WIN32 AND NOT CYGWIN)

    file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h *.hpp)
    add_llvm_pass_plugin( ${Plugin_Name} ${ALL_SOURCE_FILES}
    DEPENDS intrinsics_gen BUILDTREE_ONLY)
    set(LLVM_TEST_DEPENDS ${Plugin_Name} ${LLVM_TEST_DEPENDS} PARENT_SCOPE)

endif()
