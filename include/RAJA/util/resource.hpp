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
#if defined(RAJA_CUDA_ACTIVE)
#include "RAJA/policy/cuda/policy.hpp"
#endif
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/sequential/policy.hpp"

namespace RAJA
{

  namespace resources
  {
  using namespace camp::resources;

  template<typename e>
  struct get_resource{
    using type = camp::resources::Host;
  };

  template<typename ExecPol>
  constexpr auto get_default_resource() -> typename get_resource<ExecPol>::type {
    return get_resource<ExecPol>::type::get_default();
  }

#if defined(RAJA_CUDA_ACTIVE)
  template<size_t BlockSize, bool Async>
  struct get_resource<cuda_exec<BlockSize, Async>>{
    using type = Cuda;
  };

  template<typename ISetIter, size_t BlockSize, bool Async>
  struct get_resource<ExecPolicy<ISetIter, cuda_exec<BlockSize, Async>>>{
    using type = Cuda;
  };
#endif

#if defined(RAJA_ENABLE_HIP)
  template<size_t BlockSize, bool Async>
  struct get_resource<hip_exec<BlockSize, Async>>{
    using type = Hip;
  };

  template<typename ISetIter, size_t BlockSize, bool Async>
  struct get_resource<ExecPolicy<ISetIter, hip_exec<BlockSize, Async>>>{
    using type = Hip;
  };
#endif
  } // end namespace resources

  namespace type_traits
  {
    template <typename T> struct is_resource : std::false_type {};
    template <> struct is_resource<resources::Host> : std::true_type {};
#if defined(RAJA_CUDA_ACTIVE)
    template <> struct is_resource<resources::Cuda> : std::true_type {};
#endif
#if defined(RAJA_ENABLE_HIP)
    template <> struct is_resource<resources::Hip> : std::true_type {};
#endif
  } // end namespace type_traits

}  // end namespace RAJA

#endif //RAJA_resources_HPP#
