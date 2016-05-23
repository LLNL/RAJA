include_directories(..)

add_definitions(-DRAJA_ENABLE_NESTED)

set(KRIPKE_SRC 
    Directions.cpp  
    Grid.cpp  
    Input_Variables.cpp
    Kernel.cpp
    Layout.cpp
    Subdomain.cpp  
    Sweep_Solver.cpp
    ParallelComm.cpp  
    Timing.cpp
      
    Kernel/Kernel_3d_GDZ.cpp
    Kernel/Kernel_3d_DGZ.cpp
    Kernel/Kernel_3d_ZDG.cpp
    Kernel/Kernel_3d_DZG.cpp
    Kernel/Kernel_3d_ZGD.cpp
    Kernel/Kernel_3d_GZD.cpp
    
    ParallelComm/BlockJacobiComm.cpp
    ParallelComm/SweepComm.cpp 
    
    Test/TestKernels.cpp)

if(RAJA_ENABLE_CUDA)
  cuda_add_library(lib_kripke_raja ${KRIPKE_SRC})  
else()
  add_library(lib_kripke_raja ${KRIPKE_SRC})  
endif()  
                                                                                                          
list(APPEND KRIPKE_LIBS lib_kripke_raja) 


# Propagate to parent directory
set(KRIPKE_LIBS ${KRIPKE_LIBS} PARENT_SCOPE) 
 
