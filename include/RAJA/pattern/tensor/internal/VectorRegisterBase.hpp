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

#ifndef RAJA_pattern_tensor_VectorRegisterBase_HPP
#define RAJA_pattern_tensor_VectorRegisterBase_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/internal/TensorRegisterBase.hpp"
#include "RAJA/pattern/tensor/MatrixRegister.hpp"
#include "RAJA/pattern/tensor/stats.hpp"


namespace RAJA
{


namespace internal {

  /*!
   * This provides common functionality that is special to 1D (Vector) Tensors
   */
  template<typename Derived>
  class VectorRegisterBase;

  template<typename REGISTER_POLICY, typename T, camp::idx_t SIZE, camp::idx_t ... VAL_SEQ>
  class VectorRegisterBase<TensorRegister<REGISTER_POLICY, T, VectorLayout, camp::idx_seq<SIZE>, camp::idx_seq<VAL_SEQ...>>> :
    public TensorRegisterBase<TensorRegister<REGISTER_POLICY, T, VectorLayout, camp::idx_seq<SIZE>, camp::idx_seq<VAL_SEQ...>>>
  {
    public:
      using self_type = TensorRegister<REGISTER_POLICY, T, VectorLayout, camp::idx_seq<SIZE>, camp::idx_seq<VAL_SEQ...>>;
      using base_type = TensorRegisterBase<TensorRegister<REGISTER_POLICY, T, VectorLayout, camp::idx_seq<SIZE>, camp::idx_seq<VAL_SEQ...>>>;
      using element_type = camp::decay<T>;
      using layout_type = TensorLayout<0>;


      static constexpr camp::idx_t s_num_elem = sizeof...(VAL_SEQ);

    private:

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type *getThis(){
        return static_cast<self_type *>(this);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      self_type const *getThis() const{
        return static_cast<self_type const *>(this);
      }

    public:

      /*!
       * Provide left vector-matrix multiply for operator* between
       * this vector and a matrix
       */
      template<typename T2, typename L, typename RP>
      self_type
      operator*(MatrixRegister<T2, L, RP> const &y) const
      {
        return y.left_vector_multiply(*getThis());
      }

      // make sure our overloaded operator* doesn't hide our base class
      // implementation
      using base_type::operator*;


      /*!
       * @brief Performs load specified by TensorRef object.
       */
      template<typename POINTER_TYPE, typename INDEX_TYPE, internal::TensorTileSize TENSOR_SIZE, camp::idx_t STRIDE_ONE_DIM>
      RAJA_INLINE
      self_type &load_ref(internal::TensorRef<POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE, 1, STRIDE_ONE_DIM> const &ref){

        auto ptr = ref.m_pointer + ref.m_tile.m_begin[0]*ref.m_stride[0];

        // check for packed data
        if(STRIDE_ONE_DIM == 0){
          // full vector?
          if(TENSOR_SIZE == internal::TENSOR_FULL){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_packed ++;
#endif
            getThis()->load_packed(ptr);
          }
          // partial
          else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_packed_n ++;
#endif
            getThis()->load_packed_n(ptr, ref.m_tile.m_size[0]);
          }

        }
        // strided data
        else
        {
          // full vector?
          if(TENSOR_SIZE == internal::TENSOR_FULL){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided ++;
#endif
            getThis()->load_strided(ptr, ref.m_stride[0]);
          }
          // partial
          else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided_n ++;
#endif
            getThis()->load_strided_n(ptr, ref.m_stride[0], ref.m_tile.m_size[0]);
          }
        }
        return *getThis();
      }


      /*!
       * @brief Performs load specified by TensorRef object.
       */
      template<typename POINTER_TYPE, typename INDEX_TYPE, internal::TensorTileSize TENSOR_SIZE, camp::idx_t STRIDE_ONE_DIM>
      RAJA_INLINE
      self_type const &store_ref(internal::TensorRef<POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE, 1, STRIDE_ONE_DIM> const &ref) const {

        auto ptr = ref.m_pointer + ref.m_tile.m_begin[0]*ref.m_stride[0];

        // check for packed data
        if(STRIDE_ONE_DIM == 0){
          // full vector?
          if(TENSOR_SIZE == internal::TENSOR_FULL){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_packed ++;
#endif
            getThis()->store_packed(ptr);
          }
          // partial
          else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_packed_n ++;
#endif
            getThis()->store_packed_n(ptr, ref.m_tile.m_size[0]);
          }

        }
        // strided data
        else
        {
          // full vector?
          if(TENSOR_SIZE == internal::TENSOR_FULL){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_strided ++;
#endif
            getThis()->store_strided(ptr, ref.m_stride[0]);
          }
          // partial
          else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_strided_n ++;
#endif
            getThis()->store_strided_n(ptr, ref.m_stride[0], ref.m_tile.m_size[0]);
          }
        }
        return *getThis();
      }






      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      element_type dot(self_type const &x) const
      {
        return getThis()->multiply(x).sum();
      }


      /*!
       * Provides vector-level building block for matrix transpose operations.
       *
       * This is a non-optimized reference version which will be used if
       * no architecture specialized version is supplied
       *
       * This is a permute-and-shuffle left operation
       *
       *           X=   x0  x1  x2  x3  x4  x5  x6  x7...
       *           Y=   y0  y1  y2  y3  y4  y5  y6  y7...
       *
       *  lvl=0    Z=   x0  y0  x2  y2  x4  y4  x6  y6...
       *  lvl=1    Z=   x0  x1  y0  y1  x4  x5  y4  y5...
       *  lvl=2    Z=   x0  x1  x2  x3  y0  y1  y2  y3...
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type transpose_shuffle_left(camp::idx_t lvl, self_type const &y) const
      {
        auto const &x = *getThis();

        self_type z;

        for(camp::idx_t i = 0;i < s_num_elem;++ i){

          // extract value x or y
          camp::idx_t xy_select = (i >> lvl) & 0x1;


          z.set(xy_select == 0 ? x.get(i) : y.get(i - (1<<lvl)), i);
        }

        return z;
      }


      /*!
       * Provides vector-level building block for matrix transpose operations.
       *
       * This is a non-optimized reference version which will be used if
       * no architecture specialized version is supplied
       *
       * This is a permute-and-shuffle right operation
       *
       *           X=   x0  x1  x2  x3  x4  x5  x6  x7...
       *           Y=   y0  y1  y2  y3  y4  y5  y6  y7...
       *
       *  lvl=0    Z=   x1  y1  x3  y3  x5  y5  x7  y7...
       *  lvl=1    Z=   x2  x3  y2  y3  x6  x7  y6  y7...
       *  lvl=2    Z=   x4  x5  x6  x7  y4  y5  y6  y7...
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type transpose_shuffle_right(int lvl, self_type const &y) const
      {
        auto const &x = *getThis();

        self_type z;

        camp::idx_t i0 = 1<<lvl;

        for(camp::idx_t i = 0;i < s_num_elem;++ i){

          // extract value x or y
          camp::idx_t xy_select = (i >> lvl) & 0x1;

          z.set(xy_select == 0 ? x.get(i0 + i) : y.get(i0 + i - (1<<lvl)), i);
        }

        return z;
      }

  };

} //namespace internal


}  // namespace RAJA


// Bring in the register policy file so we get the default register type
// and all of the register traits setup
#include "RAJA/policy/tensor/arch.hpp"


#endif
