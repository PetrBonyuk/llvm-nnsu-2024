if (NOT WIN32 AND NOT CYGWIN)
  set(PluginName VeselovBitwiseShiftMultiplier)
  
  file(GLOB_RECURSE ALL_SOURCE_FILES *.cpp *.h)
  add_llvm_pass_plugin(${PluginName} 
  ${ALL_SOURCE_FILES}  )

  set(LLVM_TEST_DEPENDS ${PluginName} ${LLVM_TEST_DEPENDS} PARENT_SCOPE)
endif()
