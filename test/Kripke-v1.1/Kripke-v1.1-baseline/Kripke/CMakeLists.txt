include_directories(..)

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
  
  Test/TestKernels.cpp
)

if(RAJA_ENABLE_CUDA)
  cuda_add_library(lib_kripke ${KRIPKE_SRC})  
else()
  add_library(lib_kripke ${KRIPKE_SRC})  
endif() 
                                                                                                          
set(KRIPKE_LIBS ${KRIPKE_LIBS} lib_kripke PARENT_SCOPE) 
