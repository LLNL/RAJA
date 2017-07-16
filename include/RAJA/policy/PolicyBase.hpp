/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for basic RAJA policy mechanics.
 *
 ******************************************************************************
 */

#ifndef RAJA_POLICYBASE_HPP
#define RAJA_POLICYBASE_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstddef>
#include "RAJA/util/concepts.hpp"
#include "RAJA/util/type_traits.hpp"

namespace RAJA
{

enum class Policy { undefined, sequential, simd, openmp, target_openmp, cuda };

enum class Pattern { undefined, forall, reduce, taskgraph };

enum class Launch { undefined, sync, async };

enum class Platform { undefined = 0, host = 1, cuda = 2, omp_target = 4 };

struct PolicyBase {
};

template <Policy Policy_,
          Pattern Pattern_,
          Launch Launch_,
          Platform Platform_,
          typename... Traits>
struct PolicyBaseT : PolicyBase {
  static constexpr Policy policy = Policy_;
  static constexpr Pattern pattern = Pattern_;
  static constexpr Launch launch = Launch_;
  static constexpr Platform platform = Platform_;
};

template <typename Inner>
struct wrapper {
  using inner = Inner;
};

namespace reduce
{

struct ordered {
};

}  // end namespace wrapper


template <Policy Pol, Pattern Pat, typename... Args>
using make_policy_pattern_t =
    PolicyBaseT<Pol, Pat, Launch::undefined, Platform::undefined, Args...>;


template <Policy Policy_,
          Pattern Pattern_,
          Launch Launch_,
          Platform Platform_,
          typename... Args>
using make_policy_pattern_launch_platform_t =
    PolicyBaseT<Policy_, Pattern_, Launch_, Platform_, Args...>;

template <Policy Policy_, Pattern Pattern_, Launch Launch_, typename... Args>
using make_policy_pattern_launch_t =
    PolicyBaseT<Policy_, Pattern_, Launch_, Platform::undefined, Args...>;


namespace concepts
{

namespace impl
{
template <typename T>
using decay_t = typename std::decay<T>::type;
}

template <typename Pol>
using ExecutionPolicy = DefineConcept(
    has_type<RAJA::Policy>(decay<decltype(decay<Pol>::policy)>()),
    has_type<RAJA::Pattern>(decay<decltype(decay<Pol>::pattern)>()),
    has_type<RAJA::Launch>(decay<decltype(decay<Pol>::launch)>()),
    has_type<RAJA::Platform>(decay<decltype(decay<Pol>::platform)>()));

template <typename Pol>
using SequentialPolicy =
    DefineConcept(ExecutionPolicy<Pol>(),
                  conforms<bool_<Pol::policy == Policy::sequential>>());

template <typename Pol>
using OpenMPPolicy =
    DefineConcept(ExecutionPolicy<Pol>(),
                  conforms<bool_<Pol::policy == Policy::openmp>>());

template <typename Pol>
using TargetOpenMPPolicy =
    DefineConcept(ExecutionPolicy<Pol>(),
                  conforms<bool_<Pol::policy == Policy::target_openmp>>());

template <typename Pol>
using CudaPolicy =
    DefineConcept(ExecutionPolicy<Pol>(),
                  conforms<bool_<Pol::policy == Policy::cuda>>());

template <typename Pol>
using SimdPolicy =
    DefineConcept(ExecutionPolicy<Pol>(),
                  conforms<bool_<Pol::policy == Policy::simd>>());

}  // end namespace concepts

namespace type_traits
{

DefineTypeTraitFromConcept(is_execution_policy,
                           RAJA::concepts::ExecutionPolicy);
DefineTypeTraitFromConcept(is_sequential_policy,
                           RAJA::concepts::SequentialPolicy);
DefineTypeTraitFromConcept(is_openmp_policy, RAJA::concepts::OpenMPPolicy);
DefineTypeTraitFromConcept(is_target_openmp_policy,
                           RAJA::concepts::TargetOpenMPPolicy);
DefineTypeTraitFromConcept(is_cuda_policy, RAJA::concepts::CudaPolicy);
DefineTypeTraitFromConcept(is_simd_policy, RAJA::concepts::SimdPolicy);

}  // end namespace type_traits

}  // end namespace RAJA

#endif /* RAJA_POLICYBASE_HPP */
