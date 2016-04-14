include_directories(..)

add_library(lib_kripke  
  Directions.cpp  
  Grid.cpp  
  Input_Variables.cpp
  Kernel.cpp
  Layout.cpp
  Subdomain.cpp  
  Sweep_Solver.cpp
  ParallelComm.cpp  
  Timing.cpp
  #example.cpp  
    
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
                                                                                                          
list(APPEND KRIPKE_LIBS lib_kripke) 


if(ENABLE_CUDA)
	cuda_add_library(cukripke
	  Kernel.cu
	)

	list(APPEND KRIPKE_LIBS cukripke) 

endif()


# Propagate to parent directory
set(KRIPKE_LIBS ${KRIPKE_LIBS} PARENT_SCOPE) 
 
