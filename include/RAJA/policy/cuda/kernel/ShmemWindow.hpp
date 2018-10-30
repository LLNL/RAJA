/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for CUDA shared memory window executors.
 *
 ******************************************************************************
 */

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


#ifndef RAJA_policy_cuda_kernel_ShmemWindow_HPP
#define RAJA_policy_cuda_kernel_ShmemWindow_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"
#include "RAJA/util/ShmemTile.hpp"

#include "RAJA/pattern/kernel/ShmemWindow.hpp"
#include "RAJA/policy/cuda/kernel/internal.hpp"

namespace RAJA
{


namespace internal
{

template<typename pol>
struct checkPol{};

//Create shared memory window version 2.0
template <typename Data, camp::idx_t... Indices, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<Data, statement::CreateShmem2<camp::idx_seq<Indices...>, EnclosedStmts...>, IndexCalc>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;

  IndexCalc index_calc;

  //CUDA thread private memory
  template<camp::idx_t Pos>
  void RAJA_INLINE __device__ initMem(Data &data, int num_logical_blocks, int block_carry, checkPol<RAJA::cuda_priv_mem> ) {

    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::type;
    varType SharedM;
    camp::get<Pos>(data.param_tuple).SharedMem = &SharedM;
    inspect(data, num_logical_blocks, block_carry);
  }

  //CUDA thread private memory
  template<camp::idx_t Pos, camp::idx_t... others>
  void RAJA_INLINE __device__ initMem(Data &data, int num_logical_blocks, int block_carry, checkPol<RAJA::cuda_priv_mem> ) {

    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::type;
    varType SharedM;
    camp::get<Pos>(data.param_tuple).SharedMem = &SharedM;
    inspect<others...>(data, num_logical_blocks, block_carry);
  }

  template<camp::idx_t Pos>
  void RAJA_INLINE __device__ initMem(Data &data, int num_logical_blocks, int block_carry, checkPol<RAJA::cuda_shared_mem> ) {

    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::type;
    __shared__ varType SharedM;
    camp::get<Pos>(data.param_tuple).SharedMem = &SharedM;
    inspect(data, num_logical_blocks, block_carry);
  }

  template<camp::idx_t Pos, camp::idx_t... others>
  void RAJA_INLINE __device__ initMem(Data &data, int num_logical_blocks, int block_carry, checkPol<RAJA::cuda_shared_mem> ) {

    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::type;
    __shared__ varType SharedM;
    camp::get<Pos>(data.param_tuple).SharedMem = &SharedM;
    inspect<others...>(data, num_logical_blocks, block_carry);
  }

  template<camp::idx_t Pos, camp::idx_t... others>
  void RAJA_INLINE __device__ inspect(Data &data, int num_logical_blocks, int block_carry)
  {

    using pol_t = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::pol_t;
    initMem<Pos,others...>(data,num_logical_blocks, block_carry, checkPol<pol_t>{});

  }

  void RAJA_INLINE __device__ inspect(Data &data, int num_logical_blocks, int block_carry)
  {
    enclosed_stmts.exec(data, num_logical_blocks, block_carry);
  }


  RAJA_INLINE __device__ void exec(Data &data,
                                   int num_logical_blocks,
                                   int block_carry)
  {

    //Intialize shared/thread private memory + launch loops
    inspect<Indices...>(data, num_logical_blocks, block_carry);

    //Intialize shared/thread private memory + launch loops
    //CreateVarExec<Indices...>{}(data, num_logical_blocks, block_carry, enclosed_stmts);
    //setPtrToNull
    //TODO
  }

  RAJA_INLINE RAJA_HOST_DEVICE void initBlocks(Data &data,
                                     int num_logical_blocks,
                                     int block_stride)
  {
    enclosed_stmts.initBlocks(data, num_logical_blocks, block_stride);
  }


  RAJA_INLINE RAJA_DEVICE void initThread(Data &data)
  {
    enclosed_stmts.initThread(data);
  }

  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical)
  {

    return enclosed_stmts.calculateDimensions(data, max_physical);
  }

};



//Create Shared memory window - Version 1 TO BE DELETED....
template <typename Data, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::CreateShmem<EnclosedStmts...>,
                             IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;

  IndexCalc index_calc;

  template<camp::idx_t Pos>
  void RAJA_INLINE __device__ createShared(Data &data, int num_logical_blocks, int block_carry, int_<Pos>)
  {

    using varType = typename camp::tuple_element_t<Pos-1, typename camp::decay<Data>::param_tuple_t>::type;
    __shared__ varType SharedM;
    camp::get<Pos-1>(data.param_tuple).SharedMem = &SharedM;
    createShared(data, num_logical_blocks, block_carry, int_<Pos-1>());
  }

  void RAJA_INLINE __device__ createShared(Data &data, int num_logical_blocks, int block_carry, int_<static_cast<camp::idx_t>(1)>)
  {

    using varType = typename camp::tuple_element_t<0, typename camp::decay<Data>::param_tuple_t>::type;
    __shared__ varType SharedM;
    camp::get<0>(data.param_tuple).SharedMem = &SharedM;
    enclosed_stmts.exec(data, num_logical_blocks, block_carry);
  }

  RAJA_INLINE __device__ void exec(Data &data,
                              int num_logical_blocks,
                              int block_carry)
  {

    //if(threadIdx.x==0) printf("created shared memory! \n");

    const camp::idx_t N = camp::tuple_size<typename camp::decay<Data>::param_tuple_t>::value;
    createShared(data, num_logical_blocks, block_carry, int_<N>());

    //Do we need this?
    //RAJA::internal::shmem_set_windows(data.param_tuple,data.get_minimum_index_tuple());

  }

  RAJA_INLINE RAJA_HOST_DEVICE void initBlocks(Data &data,
                                     int num_logical_blocks,
                                     int block_stride)
  {
    enclosed_stmts.initBlocks(data, num_logical_blocks, block_stride);
  }


  RAJA_INLINE RAJA_DEVICE void initThread(Data &data)
  {
    enclosed_stmts.initThread(data);
  }

  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical)
  {

    return enclosed_stmts.calculateDimensions(data, max_physical);
  }

};




//Set Shared memory window
template <typename Data, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<Data,
                             statement::SetShmemWindow<EnclosedStmts...>,
                             IndexCalc> {

  using stmt_list_t = StatementList<EnclosedStmts...>;

  using enclosed_stmts_t =
      CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;

  IndexCalc index_calc;


  RAJA_INLINE __device__ void exec(Data &data,
                              int num_logical_blocks,
                              int block_carry)
  {

    // Call setWindow on all of our shmem objects
    RAJA::internal::shmem_set_windows(data.param_tuple,
                                      data.get_minimum_index_tuple());

    // execute enclosed statements
    enclosed_stmts.exec(data, num_logical_blocks, block_carry);
  }


  inline RAJA_HOST_DEVICE void initBlocks(Data &data,
                                          int num_logical_blocks,
                                          int block_stride)
  {
    enclosed_stmts.initBlocks(data, num_logical_blocks, block_stride);
  }


  RAJA_INLINE RAJA_DEVICE void initThread(Data &data)
  {
    enclosed_stmts.initThread(data);
  }


  RAJA_INLINE
  LaunchDim calculateDimensions(Data const &data, LaunchDim const &max_physical)
  {

    return enclosed_stmts.calculateDimensions(data, max_physical);
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif
