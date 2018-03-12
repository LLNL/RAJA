//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef EXAMPLES_MEMORYMANAGER_HPP
#define EXAMPLES_MEMORYMANAGER_HPP

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

/*
  As RAJA does not manage memory we include a general purpose memory
  manager which may be used to perform c++ style allocation/deallocation 
  or allocate/deallocate CUDA unified memory. The type of memory allocated 
  is dependent on how RAJA was configured.  
*/
namespace memoryManager{

  template <typename T>
  T *allocate(RAJA::Index_type size)
  {
    T *ptr;
#if defined(RAJA_ENABLE_CUDA)
    cudaMallocManaged((void **)&ptr, sizeof(T) * size, cudaMemAttachGlobal);
#else
    ptr = new T[size];
#endif
    return ptr;
  }
  
  template <typename T>
  void deallocate(T *&ptr)
  {
    if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
      cudaFree(ptr);
#else
      delete[] ptr;
#endif
      ptr = nullptr;
    }    
  }
  
};
#endif
