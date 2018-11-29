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

//Intialize thread shared array
template <typename Data, camp::idx_t... Indices, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<Data, statement::InitLocalMem<RAJA::cuda_shared_mem, camp::idx_seq<Indices...>, EnclosedStmts...>, IndexCalc>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;
  
  using enclosed_stmts_t =
    CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;
  
  IndexCalc index_calc;
  
  //Launch loops
  template<camp::idx_t Pos>
  void RAJA_INLINE RAJA_DEVICE initMem(Data &data, int num_logical_blocks, int block_carry)
  {
    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::element_t;
    const camp::idx_t NoElem = camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::NoElem;
    
    __shared__ varType Array[NoElem];
    camp::get<Pos>(data.param_tuple).m_arrayPtr = Array;

    enclosed_stmts.exec(data, num_logical_blocks, block_carry);
  }
  
  //Intialize local array
  //Identifies type + number of elements needed
  template<camp::idx_t Pos, camp::idx_t... others>
  void RAJA_INLINE RAJA_DEVICE initMem(Data &data, int num_logical_blocks, int block_carry)
  {
    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::element_t;
    const camp::idx_t NoElem = camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::NoElem;
    
    __shared__ varType Array[NoElem];
    camp::get<Pos>(data.param_tuple).m_arrayPtr = Array;
    initMem<others...>(data, num_logical_blocks, block_carry);
  }

  //Set pointer to null base case
  template<camp::idx_t Pos>
  void RAJA_INLINE RAJA_DEVICE setPtrToNull(Data &data, int num_logical_blocks, int block_carry)
  {

    camp::get<Pos>(data.param_tuple).m_arrayPtr = nullptr;
  }


  //Set pointer to null recursive case
  template<camp::idx_t Pos, camp::idx_t... others>
  void RAJA_INLINE RAJA_DEVICE setPtrToNull(Data &data, int num_logical_blocks, int block_carry)
  {

    camp::get<Pos>(data.param_tuple).m_arrayPtr = nullptr;
    setPtrToNull<others...>(data, num_logical_blocks, block_carry);
  }


  RAJA_INLINE RAJA_DEVICE void exec(Data &data,
                                   int num_logical_blocks,
                                   int block_carry)
  {
    
    //Intialize scoped arrays + launch loops
    initMem<Indices...>(data, num_logical_blocks, block_carry);
    
    //set pointers in scoped arrays to null
    setPtrToNull<Indices...>(data, num_logical_blocks, block_carry);
  }

  RAJA_INLINE RAJA_DEVICE void initBlocks(Data &data,
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

//Intialize thread private array
template <typename Data, camp::idx_t... Indices, typename... EnclosedStmts, typename IndexCalc>
struct CudaStatementExecutor<Data, statement::InitLocalMem<RAJA::cuda_thread_mem, camp::idx_seq<Indices...>, EnclosedStmts...>, IndexCalc>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;
  
  using enclosed_stmts_t =
    CudaStatementListExecutor<Data, stmt_list_t, IndexCalc>;
  enclosed_stmts_t enclosed_stmts;
  
  IndexCalc index_calc;
  
  //Launch loops
  template<camp::idx_t Pos>
  void RAJA_INLINE RAJA_DEVICE initMem(Data &data, int num_logical_blocks, int block_carry)
  {
    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::element_t;
    const camp::idx_t NoElem = camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::NoElem;
    
    varType Array[NoElem];
    camp::get<Pos>(data.param_tuple).m_arrayPtr = Array;

    enclosed_stmts.exec(data, num_logical_blocks, block_carry);
  }
  
  //Intialize local array
  //Identifies type + number of elements needed
  template<camp::idx_t Pos, camp::idx_t... others>
  void RAJA_INLINE RAJA_DEVICE initMem(Data &data, int num_logical_blocks, int block_carry)
  {
    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::element_t;
    const camp::idx_t NoElem = camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::NoElem;
    
    varType Array[NoElem];
    camp::get<Pos>(data.param_tuple).m_arrayPtr = Array;
    initMem<others...>(data, num_logical_blocks, block_carry);
  }

  //Set pointer to null base case
  template<camp::idx_t Pos>
  void RAJA_INLINE RAJA_DEVICE setPtrToNull(Data &data, int num_logical_blocks, int block_carry)
  {

    camp::get<Pos>(data.param_tuple).m_arrayPtr = nullptr;
  }


  //Set pointer to null recursive case
  template<camp::idx_t Pos, camp::idx_t... others>
  void RAJA_INLINE RAJA_DEVICE setPtrToNull(Data &data, int num_logical_blocks, int block_carry)
  {

    camp::get<Pos>(data.param_tuple).m_arrayPtr = nullptr;
    setPtrToNull<others...>(data, num_logical_blocks, block_carry);
  }


  RAJA_INLINE RAJA_DEVICE void exec(Data &data,
                                   int num_logical_blocks,
                                   int block_carry)
  {
    
    //Intialize scoped arrays + launch loops
    initMem<Indices...>(data, num_logical_blocks, block_carry);
    
    //set pointers in scoped arrays to null
    setPtrToNull<Indices...>(data, num_logical_blocks, block_carry);
  }

  RAJA_INLINE RAJA_DEVICE void initBlocks(Data &data,
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


  RAJA_INLINE RAJA_DEVICE void exec(Data &data,
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
