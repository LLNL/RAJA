//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_openmp_target_kernel_For_HPP
#define RAJA_policy_openmp_target_kernel_For_HPP

#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA {
namespace internal {

template <camp::idx_t ArgumentId, typename Data, typename Types, typename... EnclosedStmts>
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
  void exec() { execute_statement_list<camp::list<EnclosedStmts...>, Types>(data); }

  template <typename InIndexType>
  RAJA_INLINE void operator()(InIndexType i)
  {
    data.template assign_offset<ArgumentId>(i);
    exec();
  }
};

template <camp::idx_t ArgumentId,
          int N,
          typename... EnclosedStmts,
          typename Types>
struct StatementExecutor<statement::For<ArgumentId, omp_target_parallel_for_exec<N>, EnclosedStmts...>, Types>
{

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    // Set the argument type for this loop
    using NewTypes = setSegmentTypeFromData<Types, ArgumentId, Data>;

    OpenMPTargetForWrapper<ArgumentId, Data, NewTypes, EnclosedStmts...> for_wrapper(data);

    auto len = segment_length<ArgumentId>(data);
    using len_t = decltype(len);

    auto r = resources::Omp::get_default();
    forall_impl(r, omp_target_parallel_for_exec<N>{}, TypedRangeSegment<len_t>(0, len), for_wrapper, RAJA::expt::get_empty_forall_param_pack());
  }
};



}
}

#endif // RAJA_policy_openmp_kernel_For_HPP
