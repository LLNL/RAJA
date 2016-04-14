include_directories(.)


add_subdirectory(Kripke)
add_subdirectory(RAJA)


#if(ENABLE_CUDA)
#	cuda_add_library(cukripkelib STATIC ${CUDA_KRIPKE_SOURCE})

#  list(APPEND KRIPKE_LIBS cukripkelib)
#  list(APPEND KRIPKE_LIBS ${CUDA_cublas_LIBRARY} ${CUDA_npps_LIBRARY} ${CUDA_TOOLKIT_ROOT_DIR}/lib64/libnvToolsExt.so)

#endif()

set(KRIPKE_LIBS ${KRIPKE_LIBS} PARENT_SCOPE)

