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
    // Non templated inline function so as not to generate duplicate objects for cuda resource.
    RAJA_INLINE Resource get_cuda_default(){
      static Resource r = Resource{Cuda::get_default()};
      return r;
    }
  }

  RAJA_INLINE Resource get_default_resource(seq_exec){
    std::cout<<"Get defualt seq_exec\n";
    return Resource{Host::get_default()};
  }

  template<size_t BlockSize, bool Async>
  RAJA_INLINE Resource get_default_resource(cuda_exec<BlockSize, Async>){
    std::cout<<"Get defualt cuda_exec\n";
    return detail::get_cuda_default(); 
  }
  template<size_t BlockSize, bool Async>
  RAJA_INLINE Resource get_default_resource(ExecPolicy<seq_exec,cuda_exec<BlockSize, Async>>){
    std::cout<<"Get defualt cuda_exec\n";
    return detail::get_cuda_default(); 
  }

  template <typename SELECTOR, typename... POLICIES>
  RAJA_INLINE Resource get_default_resource(RAJA::policy::multi::MultiPolicy<SELECTOR, POLICIES...> &p){
    std::cout<<"get default MultPolicy\n";
    return Resource{Host::get_default()};
  }

  // Temporary to catch all of the other policies.
  template<typename EXEC_POL>
  RAJA_INLINE Resource get_default_resource(EXEC_POL){
    std::cout<<"Get defualt EXEC_POL\n";
    return Resource{Host::get_default()};
  }

  }
  namespace type_traits
  {
    template <typename T>
    struct is_resource
        : ::std::is_same<camp::resources::Resource, typename std::decay<T>::type> {
    };
  }
}  // end namespace RAJA

#endif //RAJA_resources_HPP#
