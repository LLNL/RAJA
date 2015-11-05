add_definitions(-DDOUBLE -DDO_MPI)

set_source_files_properties( CoMD_info.h PROPERTIES GENERATED true)
add_custom_command ( OUTPUT CoMD_info.h 
  COMMAND ./generate_info_header CoMD-local ${CMAKE_CXX_COMPILER}" "
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

add_executable(coMD
  cmdLineParser.c
  CoMD.c
  decomposition.c
  eam.c
  haloExchange.c
  initAtoms.c
  linkCells.c
  ljForce.c
  mycommand.c
  parallel.c
  performanceTimers.c
  random.c
  timestep.c
  yamlOutput.c)
