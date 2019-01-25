//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
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
#ifndef RAJA_policy_openmp_target_kernel_For_HPP
#define RAJA_policy_openmp_target_kernel_For_HPP

#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA {
namespace internal {

template <camp::idx_t ArgumentId, typename Data, typename... EnclosedStmts>
struct OpenMPTargetForWrapper : public GenericWrapperBase 
{
  using data_t = camp::decay<Data>;

  data_t data;

  /*! 
   * \brief Deferences data so that it can be mapped to the device
   */
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
struct StatementExecutor<statement::For<ArgumentId, omp_target_parallel_for_exec<N>, EnclosedStmts...>> 
{

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    OpenMPTargetForWrapper<ArgumentId, Data, EnclosedStmts...> for_wrapper(data);

    auto len = segment_length<ArgumentId>(data);
    using len_t = decltype(len);

    forall_impl(omp_target_parallel_for_exec<N>{}, TypedRangeSegment<len_t>(0, len), for_wrapper);
  }
};

template <camp::idx_t ArgumentId,
          typename... EnclosedStmts>
struct StatementExecutor<statement::For<ArgumentId, seq_exec, EnclosedStmts...>>
{

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    auto len = segment_length<ArgumentId>(data);

    for (int i = 0; i < len; ++i) {
      data.template assign_offset<ArgumentId>(i);

      // execute enclosed statements
      //enclosed_stmts.exec(data);
      execute_statement_list<camp::list<EnclosedStmts...>>(data);
    }

  }
};

}
}

#endif // RAJA_policy_openmp_kernel_For_HPP
