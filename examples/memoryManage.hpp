#ifndef __MEMORYMANAGER_HPP__
#define __MEMORYMANAGER_HPP__

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

/*
  Developer must manage memory
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
