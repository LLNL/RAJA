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
// For additional details, please also read raja/README-license.txt.
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
 
#ifndef RAJA_forallN_openmp_HXX__
#define RAJA_forallN_openmp_HXX__

#include<RAJA/config.hxx>
#include<RAJA/int_datatypes.hxx>

namespace RAJA {



/******************************************************************
 *  ForallN CUDA policies
 ******************************************************************/

struct ForallN_OMP_Parallel_Tag {};
template<typename NEXT=Execute>
struct OMP_Parallel {
  // Identify this policy
  typedef ForallN_OMP_Parallel_Tag PolicyTag;

  // The next nested-loop execution policy
  typedef NEXT NextPolicy;
};



/******************************************************************
 *  ForallN collapse nowait policies
 ******************************************************************/

struct omp_collapse_nowait_exec {};

template<typename ... PREST>
struct ForallN_Executor<
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment>,
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment>,
  PREST... > 
{
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset0;  
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset1;  
 
  typedef ForallN_Executor<PREST...> NextExec;
  NextExec next_exec;
  
  RAJA_INLINE
  constexpr
  ForallN_Executor(
    ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &iset0_,  
    ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &iset1_,
    PREST const &... prest) 
    :  iset0(iset0_), iset1(iset1_), next_exec(prest...) 
  { }

  template<typename BODY>
  RAJA_INLINE
  void operator()(BODY body) const {
    
    int begin_i = iset0.getBegin();
    int begin_j = iset1.getBegin();
    int end_i = iset0.getEnd();
    int end_j = iset1.getEnd();
    
    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);
    
#pragma omp for nowait collapse(2)
    for(int i = begin_i;i < end_i;++ i){
      for(int j = begin_j;j < end_j;++ j){
        outer(i,j);
      }
    }
  }
};

template<typename ... PREST>
struct ForallN_Executor<
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment>,
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment>,
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment>,
  PREST... > 
{

  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset0;  
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset1;  
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset2;  
  
  typedef ForallN_Executor<PREST...> NextExec;
  NextExec next_exec;
  
  RAJA_INLINE
  constexpr
  ForallN_Executor(
    ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &iset0_,  
    ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &iset1_,  
    ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &iset2_,
    PREST ... prest) 
    :  iset0(iset0_), iset1(iset1_), iset2(iset2_), next_exec(prest...)
  { }

  template<typename BODY>
  RAJA_INLINE
  void operator()(BODY body) const {
    
    int begin_i = iset0.getBegin();
    int begin_j = iset1.getBegin();
    int begin_k = iset2.getBegin();
    int end_i = iset0.getEnd();
    int end_j = iset1.getEnd();
    int end_k = iset2.getEnd();
    
    ForallN_PeelOuter<NextExec, BODY> outer(next_exec, body);
    
#pragma omp for nowait collapse(3)
    for(int i = begin_i;i < end_i;++ i){
      for(int j = begin_j;j < end_j;++ j){
        for(int k = begin_k;k < end_k;++ k){
          outer(i,j,k);    
        }
      }
    }
  
  }
};

/******************************************************************
 *  forallN_policy(), OpenMP Parallel Region execution
 ******************************************************************/

/*!
 * \brief Tiling policy front-end function.
 */
template<typename POLICY, typename BODY, typename ... PARGS>
RAJA_INLINE void forallN_policy(ForallN_OMP_Parallel_Tag, BODY body, PARGS ... pargs){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

#pragma omp parallel
  {
    forallN_policy<NextPolicy>(NextPolicyTag(), body, pargs...);
  }
}



} // namespace RAJA
  
#endif

