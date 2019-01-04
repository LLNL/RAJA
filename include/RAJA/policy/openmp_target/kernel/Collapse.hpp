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
#ifndef RAJA_policy_openmp_target_kernel_Collapse_HPP
#define RAJA_policy_openmp_target_kernel_Collapse_HPP

#include "RAJA/pattern/kernel/internal.hpp"

namespace RAJA {
namespace internal {

template <camp::idx_t Arg0, camp::idx_t Arg1, typename... EnclosedStmts>
struct StatementExecutor<statement::Collapse<omp_target_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1>,
                                             EnclosedStmts...>> 
{
  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    using data_t = camp::decay<Data>;
    data_t private_data{data};

    auto l0 = segment_length<Arg0>(data);
    auto l1 = segment_length<Arg1>(data);

#pragma omp target teams distribute parallel for schedule(static, 1) \
    map(to : private_data) collapse(2)
      for (auto i0 = (decltype(l0))0; i0 < l0; ++i0) {
        for (auto i1 = (decltype(l1))0; i1 < l1; ++i1) {
          private_data.template assign_offset<Arg0>(i0);
          private_data.template assign_offset<Arg1>(i1);
          execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
        }
      }
    }
};

template <camp::idx_t Arg0, camp::idx_t Arg1, camp::idx_t Arg2, typename... EnclosedStmts>
struct StatementExecutor<statement::Collapse<omp_target_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1, Arg2>,
                                             EnclosedStmts...>> 
{
  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    using data_t = camp::decay<Data>;
    data_t private_data{data};

    auto l0 = segment_length<Arg0>(data);
    auto l1 = segment_length<Arg1>(data);
    auto l2 = segment_length<Arg2>(data);

#pragma omp target teams distribute parallel for schedule(static, 1) \
    map(to : private_data) collapse(3)
      for (auto i0 = (decltype(l0))0; i0 < l0; ++i0) {
        for (auto i1 = (decltype(l1))0; i1 < l1; ++i1) {
          for (auto i2 = (decltype(l2))0; i2 < l2; ++i2) {
            private_data.template assign_offset<Arg0>(i0);
            private_data.template assign_offset<Arg1>(i1);
            private_data.template assign_offset<Arg2>(i2);
            execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
          }
        }
      }
    }
};

template <camp::idx_t Arg0, camp::idx_t Arg1, camp::idx_t Arg2, camp::idx_t Arg3, typename... EnclosedStmts>
struct StatementExecutor<statement::Collapse<omp_target_parallel_collapse_exec,
                                             ArgList<Arg0, Arg1, Arg2, Arg3>,
                                             EnclosedStmts...>> 
{
  template <typename Data>
  static RAJA_INLINE void exec(Data&& data)
  {
    using data_t = camp::decay<Data>;
    data_t private_data{data};

    auto l0 = segment_length<Arg0>(data);
    auto l1 = segment_length<Arg1>(data);
    auto l2 = segment_length<Arg2>(data);
    auto l3 = segment_length<Arg3>(data);

#pragma omp target teams distribute parallel for schedule(static, 1) \
    map(to : private_data) collapse(4)
      for (auto i0 = (decltype(l0))0; i0 < l0; ++i0) {
        for (auto i1 = (decltype(l1))0; i1 < l1; ++i1) {
          for (auto i2 = (decltype(l2))0; i2 < l2; ++i2) {
            for (auto i3 = (decltype(l3))0; i3 < l3; ++i3) {
              private_data.template assign_offset<Arg0>(i0);
              private_data.template assign_offset<Arg1>(i1);
              private_data.template assign_offset<Arg2>(i2);
              private_data.template assign_offset<Arg3>(i2);
              execute_statement_list<camp::list<EnclosedStmts...>>(private_data);
            }
          }
        }
      }
    }
};

}
}

#endif // RAJA_policy_openmp_target_kernel_Collapse_HPP
