//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
#ifndef RAJA_policy_openmp_kernel_For_HPP
#define RAJA_policy_openmp_kernel_For_HPP

#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA {
namespace internal {

template <camp::idx_t ArgumentId, typename Data, typename... EnclosedStmts>
struct OpenMPTargetForWrapper : public GenericWrapperBase {
  using data_t = camp::decay<Data>;

  data_t data;

  RAJA_INLINE
  constexpr explicit OpenMPTargetForWrapper(data_t &d) : 
    data{d}  {}

  RAJA_INLINE
  void exec() { execute_statement_list<camp::list<EnclosedStmts...>>(data); }

  template <typename InIndexType>
  RAJA_INLINE void operator()(InIndexType i)
  {
    data.template assign_offset<ArgumentId>(i);
    exec();
  }
};

template <camp::idx_t ArgumentId,
          int N,
          typename... EnclosedStmts>
struct StatementExecutor<statement::For<ArgumentId, omp_target_parallel_for_exec<N>, EnclosedStmts...>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    OpenMPTargetForWrapper<ArgumentId, Data, EnclosedStmts...> for_wrapper(data);

    auto len = segment_length<ArgumentId>(data);
    using len_t = decltype(len);

    forall_impl(omp_target_parallel_for_exec<N>{}, TypedRangeSegment<len_t>(0, len), for_wrapper);
  }
};

template <camp::idx_t Arg0, camp::idx_t Arg1, typename... EnclosedStmts>
struct StatementExecutor<statement::Collapse<omp_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1>,
                                             EnclosedStmts...>> {


  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    OpenMPTargetForWrapper<ArgumentId, Data, EnclosedStmts...> for_wrapper(data);

    auto l0 = segment_length<Arg0>(data);
    auto l1 = segment_length<Arg1>(data);

#pragma omp target teams distribute parallel for schedule(static, 1) \
    map(to : for_wrapper) collapse(2)
      for (auto i0 = (decltype(l0))0; i0 < l0; ++i0) {
        for (auto i1 = (decltype(l1))0; i1 < l1; ++i1) {
          private_data.template assign_offset<Arg0>(i0);
          private_data.template assign_offset<Arg1>(i1);
          execute_statement_list<camp::list<EnclosedStmts...>>(for_wrapper);
        }
      }
    }
  }
};

}
}

#endif // RAJA_policy_openmp_kernel_For_HPP
