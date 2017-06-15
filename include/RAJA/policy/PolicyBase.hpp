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

#include <stddef.h>

namespace RAJA
{

enum class Policy { undefined, sequential, simd, openmp, cuda, cilk };

enum class Launch { undefined, sync, async };

enum class Pattern {
  undefined,
  forall,
  reduce,
  taskgraph,
};

struct PolicyBase {
};

template <Policy P = Policy::undefined,
          Launch L = Launch::undefined,
          Pattern Pat = Pattern::undefined>
struct PolicyBaseT : public PolicyBase {
  static constexpr Policy policy = P;
  static constexpr Launch launch = L;
  static constexpr Pattern pattern = Pat;
};

template <typename Inner, typename... T>
struct WrapperPolicy : public Inner {
  using inner = Inner;
};

// "makers"

template <typename Inner, typename... T>
struct wrap : public WrapperPolicy<Inner, T...> {
};

template <Policy Pol, Launch L, Pattern P>
struct make_policy_launch_pattern : public PolicyBaseT<Pol, L, P> {
};

template <Policy P>
struct make_policy
    : public PolicyBaseT<P, Launch::undefined, Pattern::undefined> {
};

template <Launch L>
struct make_launch
    : public PolicyBaseT<Policy::undefined, L, Pattern::undefined> {
};

template <Pattern P>
struct make_pattern
    : public PolicyBaseT<Policy::undefined, Launch::undefined, P> {
};

template <Policy Pol, Launch L>
struct make_policy_launch : public PolicyBaseT<Pol, L, Pattern::undefined> {
};

template <Policy Pol, Pattern P>
struct make_policy_pattern : public PolicyBaseT<Pol, Launch::undefined, P> {
};

template <Launch L, Pattern P>
struct make_launch_pattern : public PolicyBaseT<Policy::undefined, L, P> {
};

}  // end namespace RAJA

#endif /* RAJA_POLICYBASE_HPP */
