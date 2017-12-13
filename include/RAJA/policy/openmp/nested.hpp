/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
 *
 ******************************************************************************
 */

#ifndef RAJA_policy_openmp_nested_HPP
#define RAJA_policy_openmp_nested_HPP

#include "RAJA/config.hpp"
#include "RAJA/pattern/nested.hpp"

#if defined(RAJA_ENABLE_OPENMP)

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

#include <cassert>
#include <climits>

#include "RAJA/RAJA.hpp"
#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

//#include "RAJA/policy/openmp/MemUtils_OPENMP.hpp"
#include "RAJA/policy/openmp/policy.hpp"

//#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{
namespace nested
{
  
  struct omp_parallel_collapse_exec {
  };

  template <typename... FOR>
  using OmpParallelCollapse = Collapse<omp_parallel_collapse_exec, FOR...>;
  
  
  // TODO, check that FT... are openmp policies
  template <typename FT0, typename FT1>
  struct Executor<Collapse<omp_parallel_collapse_exec, FT0, FT1> > {
    
    static_assert(std::is_base_of<internal::ForBase, FT0>::value,
                  "Only For-based policies should get here");
    static_assert(std::is_base_of<internal::ForBase, FT1>::value,
                  "Only For-based policies should get here");
    template <typename WrappedBody>
    void operator()(Collapse<omp_parallel_collapse_exec, FT0, FT1> const &, WrappedBody const &wrap)
    {

      auto b0 = std::begin(camp::get<FT0::index_val>(wrap.data.st));
      auto b1 = std::begin(camp::get<FT1::index_val>(wrap.data.st));
      auto e0 = std::end(camp::get<FT0::index_val>(wrap.data.st));
      auto e1 = std::end(camp::get<FT1::index_val>(wrap.data.st));

      auto l0 = std::distance(b0,e0);
      auto l1 = std::distance(b1,e1);     

#pragma omp parallel
      {
        auto privatizer = RAJA::nested::thread_privatize(wrap);
        auto private_wrap = privatizer.get_priv();
        
#if !defined(RAJA_COMPILER_MSVC)
#pragma omp for nowait collapse(2)
#else
#pragma omp for nowait
#endif
        for (auto i0 = (decltype(l0))0; i0 < l0; ++i0){
          for (auto i1 = (decltype(l1))0; i1 < l1; ++i1){
            private_wrap.data.template assign_index<FT0::index_val>(b0[i0]);
            private_wrap.data.template assign_index<FT1::index_val>(b1[i1]);
            private_wrap();
          }
        }
        
      }
       
    }
   
  };


}  // namespace nested
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_OPENMP guard

#endif  // closing endif for header file include guard
