/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA resource definitions.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_resource_HPP
#define RAJA_resource_HPP

#include "camp/resource.hpp"
#include "RAJA/policy/cuda/policy.hpp"
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/sequential/policy.hpp"

namespace RAJA
{

  namespace resources
  {
  using namespace camp::resources;



  template<typename T>
  T raja_get(Resource &res)
  {
    if (!res.try_get<T>()) RAJA_ABORT_OR_THROW("Execution architecture incompatible with resource.");

    return res.get<T>();
  }

  namespace detail
  {
#if defined(RAJA_ENABLE_CUDA)
    // Non templated inline function so as not to generate duplicate objects for cuda resource.
    RAJA_INLINE Cuda get_cuda_default(){
      static Cuda r = Cuda::get_default();
      return r;
    }
#endif
#if defined(RAJA_ENABLE_HIP)
    // Non templated inline function so as not to generate duplicate objects for cuda resource.
    RAJA_INLINE Hip get_hip_default(){
      static Hip r = Hip::get_default();
      return r;
    }
#endif
  }

  template<typename e>
  struct get_default_resource_s{
    using type = Host;
  };


#if defined(RAJA_ENABLE_CUDA)
  template<size_t BlockSize, bool Async>
  struct get_default_resource_s<cuda_exec<BlockSize, Async>>{
    using type = Cuda;
  };

  template<typename ISetIter, size_t BlockSize, bool Async>
  struct get_default_resource_s<ExecPolicy<ISetIter, cuda_exec<BlockSize, Async>>>{
    using type = Cuda;
  };

  template<size_t BlockSize, bool Async>
  RAJA_INLINE Cuda get_default_resource(cuda_exec<BlockSize, Async>){
    //std::cout<<"Get defualt cuda_exec\n";
    return detail::get_cuda_default(); 
  }
  template<size_t BlockSize, bool Async>
  RAJA_INLINE Cuda get_default_resource(ExecPolicy<seq_exec,cuda_exec<BlockSize, Async>>){
    //std::cout<<"Get defualt cuda_exec\n";
    return detail::get_cuda_default(); 
  }
#endif

#if defined(RAJA_ENABLE_HIP)
  template<size_t BlockSize, bool Async>
  RAJA_INLINE Hip get_default_resource(hip_exec<BlockSize, Async>){
    //std::cout<<"Get defualt hip_exec\n";
    return detail::get_hip_default(); 
  }
  template<size_t BlockSize, bool Async>
  RAJA_INLINE Hip get_default_resource(ExecPolicy<seq_exec,hip_exec<BlockSize, Async>>){
    //std::cout<<"Get defualt hip_exec\n";
    return detail::get_hip_default(); 
  }
#endif

  template <typename SELECTOR, typename... POLICIES>
  RAJA_INLINE Host get_default_resource(RAJA::policy::multi::MultiPolicy<SELECTOR, POLICIES...>){
    //std::cout<<"get default MultPolicy\n";
    return Host::get_default();
  }

  // Temporary to catch all of the other policies.
  template<typename EXEC_POL>
  RAJA_INLINE Host get_default_resource(EXEC_POL){
    //std::cout<<"Get defualt EXEC_POL\n";
    return Host::get_default();
  }

  }
  namespace type_traits
  {
    template <typename T> struct is_resource : std::false_type {};
    template <> struct is_resource<resources::Host> : std::true_type {};
#if defined(RAJA_ENABLE_CUDA)
    template <> struct is_resource<resources::Cuda> : std::true_type {};
#endif
#if defined(RAJA_ENABLE_HIP)
    template <> struct is_resource<resources::Hip> : std::true_type {};
#endif
  }
}  // end namespace RAJA

#endif //RAJA_resources_HPP#
