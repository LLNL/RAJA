/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */
  
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
  typedef ForallN_Executor<PREST...> NextExec;
  
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset0;  
  ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> iset1;  
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
  
  RAJA_INLINE
  constexpr
  ForallN_Executor(
    ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &iset0_,  
    ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &iset1_,  
    ForallN_PolicyPair<omp_collapse_nowait_exec, RangeSegment> const &iset2_,
    PREST ... prest) 
    :  iset0(iset0_), iset1(iset1_), iset2(iset2_) 
  { }

  template<typename BODY, typename ... ARGS>
  RAJA_INLINE
  void operator()(BODY body, ARGS ... args) const {
    
    int begin_i = iset0.getBegin();
    int begin_j = iset1.getBegin();
    int begin_k = iset2.getBegin();
    int end_i = iset0.getEnd();
    int end_j = iset1.getEnd();
    int end_k = iset2.getEnd();
    
#pragma omp for nowait collapse(3)
    for(int i = begin_i;i < end_i;++ i){
      for(int j = begin_j;j < end_j;++ j){
        for(int k = begin_k;k < end_k;++ k){
          ForallN_BindFirstArg<BODY> bind_i(body, i);
          ForallN_BindFirstArg<decltype(bind_i)> bind_j(bind_i, j);
          ForallN_BindFirstArg<decltype(bind_j)> bind_k(bind_j, k);
          
          bind_k(args...);    
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

