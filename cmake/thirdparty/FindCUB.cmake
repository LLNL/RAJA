include (FindPackageHandleStandardArgs)

find_path(CUB_INCLUDE_DIRS
  NAMES cub/cub.cuh
  HINTS 
    ${CUB_DIR}/
    ${CUB_DIR}/include)

find_package_handle_standard_args(
  CUB
  DEFAULT_MSG
  CUB_INCLUDE_DIRS)

if (CUB_INCLUDE_DIRS)
  set(CUB_FOUND True)
else ()
  set(CUB_FOUND False)
endif()
