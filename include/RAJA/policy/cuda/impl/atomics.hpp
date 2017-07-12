#ifndef RAJA_cuda_atomics_HPP
#define RAJA_cuda_atomics_HPP

namespace cuda
{

namespace atomics
{

/*!
 ******************************************************************************
 *
 * \brief Generics of atomic update methods used in reduction variables.
 *
 * The generic version just wraps the nvidia cuda atomics.
 * Specializations implement other atomics using atomic CAS.
 *
 ******************************************************************************
 */
template <typename T>
__device__ inline T min(T *address, T value)
{
  return atomicMin(address, value);
}
///
template <typename T>
__device__ inline T max(T *address, T value)
{
  return atomicMax(address, value);
}
///
template <typename T>
__device__ inline T add(T *address, T value)
{
  return atomicAdd(address, value);
}

//
// Three different variants of min/max reductions can be run by choosing
// one of these macros. Only one should be defined!!!
//
//#define RAJA_USE_ATOMIC_ONE
//#define RAJA_USE_NO_ATOMICS
#define RAJA_USE_ATOMIC_TWO

#include "RAJA/policy/cuda/impl/atomics_v1.hpp"
#include "RAJA/policy/cuda/impl/atomics_v2.hpp"

#if !defined(RAJA_USE_NO_ATOMICS) && !defined(RAJA_USE_CUDA_ATOMIC)
#error "one of the options for using/not using atomics must be specified"
#endif

}  // end namespace atomics

}  // end namespace cuda

#endif
