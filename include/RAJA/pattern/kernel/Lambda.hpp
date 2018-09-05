/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for kernel lambda executor.
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


#ifndef RAJA_pattern_kernel_Lambda_HPP
#define RAJA_pattern_kernel_Lambda_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "camp/camp.hpp"
#include "camp/concepts.hpp"
#include "camp/tuple.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/internal.hpp"

const int TILE_DIM = 2;

template<typename Type>
struct PtrWrapper
{ 
  Type * myData;
};


//myShared memory
struct sharedMem{
  int array[TILE_DIM][TILE_DIM] = {{0}};
  sharedMem(){ printf("Created RAJA Shared Memory \n");};
  int &operator()(int row, int col) 
  { 
    //printf("row = %d col = %d, data = %d \n",row, col, array[row][col]);   
    return array[row][col];
  };
};

namespace RAJA
{

namespace statement
{

/*!
 * A RAJA::kernel statement that invokes a lambda function.
 *
 * The lambda is specified by its index in the sequence of lambda arguments
 * to a RAJA::kernel method.
 *
 * for example:
 * RAJA::kernel<exec_pol>(make_tuple{s0, s1, s2}, lambda0, lambda1);
 *
 */
template <camp::idx_t BodyIdx>
struct Lambda : internal::Statement<camp::nil> {
  const static camp::idx_t loop_body_index = BodyIdx;
};

//This statement will create shared memory
struct CreateShmem : public internal::Statement<camp::nil> {
};

}  // end namespace statement

namespace internal
{

template <camp::idx_t LoopIndex>
struct StatementExecutor<statement::Lambda<LoopIndex>> {

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {
    invoke_lambda<LoopIndex>(std::forward<Data>(data));
  }
};

template<>
struct StatementExecutor<statement::CreateShmem>{

  template<typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    printf("Creating RAJA shared memory.... \n");
    sharedMem rajaShared;

    camp::get<0>(data.param_tuple).myData = &rajaShared;
    
    #pragma omp barrier

  }
};


}  // namespace internal

}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
