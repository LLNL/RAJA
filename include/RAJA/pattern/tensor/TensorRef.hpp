/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining SIMD/SIMT register operations.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_tensor_tensorref_HPP
#define RAJA_pattern_tensor_tensorref_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"


namespace RAJA
{

  namespace internal
  {

  namespace ET
  {
    enum TensorTileSize
    {
      TENSOR_PARTIAL,  // the tile is a full TensorRegister
      TENSOR_FULL,     // the tile is a partial TensorRegister
      TENSOR_MULTIPLE  // the tile is multiple TennsorRegisters
    };

    template<typename INDEX_TYPE, TensorTileSize TENSOR_SIZE, camp::idx_t NUM_DIMS>
    struct TensorTile
    {
        using index_type = INDEX_TYPE;
        index_type m_begin[NUM_DIMS];
        index_type m_size[NUM_DIMS];

        static constexpr TensorTileSize s_tensor_size = TENSOR_SIZE;


        RAJA_HOST_DEVICE
        RAJA_INLINE
        void print() const {
          printf("TensorTile: dims=%d, m_begin=[",  (int)NUM_DIMS);

          for(camp::idx_t i = 0;i < NUM_DIMS;++ i){
            printf("%ld ", (long)m_begin[i]);
          }

          printf("], m_size=[");

          for(camp::idx_t i = 0;i < NUM_DIMS;++ i){
            printf("%ld ", (long)m_size[i]);
          }

          printf("]\n");
        }
    };



    template<typename TENSOR_REG_TYPE, typename POINTER_TYPE, typename INDEX_TYPE, TensorTileSize TENSOR_SIZE, camp::idx_t NUM_DIMS, camp::idx_t STRIDE_ONE_DIM = -1>
    struct TensorRef
    {
        using self_type = TensorRef<TENSOR_REG_TYPE, POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE, NUM_DIMS, STRIDE_ONE_DIM>;
        using tile_type = TensorTile<INDEX_TYPE, TENSOR_SIZE, NUM_DIMS>;

        using tensor_type = TENSOR_REG_TYPE;
        using pointer_type = POINTER_TYPE;
        using index_type = INDEX_TYPE;
        static constexpr camp::idx_t s_stride_one_dim = STRIDE_ONE_DIM;

        pointer_type m_pointer;
        index_type m_stride[NUM_DIMS];
        tile_type m_tile;

        RAJA_HOST_DEVICE
        RAJA_INLINE
        void print() const {
          printf("TensorRef: dims=%d, m_pointer=%p, m_stride=[", (int)NUM_DIMS, m_pointer);

          for(camp::idx_t i = 0;i < NUM_DIMS;++ i){
            printf("%ld ", (long)m_stride[i]);
          }

          printf("]\n");

          m_tile.print();
        }

    };



  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
