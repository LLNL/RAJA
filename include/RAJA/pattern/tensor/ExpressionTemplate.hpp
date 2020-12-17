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

#ifndef RAJA_pattern_tensor_expression_template_HPP
#define RAJA_pattern_tensor_expression_template_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/TensorRef.hpp"


namespace RAJA
{

  namespace internal
  {

  namespace ET
  {

    template<typename TENSOR_REGISTER_TYPE, typename REF_TYPE>
    class TensorLoadStore;

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorMultiply;

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorAdd;

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorSubtract;

    template<typename DERIVED_TYPE>
    class TensorExpressionBase  {
      public:
        using self_type = DERIVED_TYPE;

      private:

        RAJA_INLINE
        RAJA_HOST_DEVICE
        self_type *getThis(){
          return static_cast<self_type*>(this);
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        constexpr
        self_type const *getThis() const {
          return static_cast<self_type const*>(this);
        }

      public:

        template<typename RHS>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorMultiply<self_type, RHS> operator*(RHS const &rhs) const {
          return TensorMultiply<self_type, RHS>(*getThis(), rhs);
        }

        template<typename RHS>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorAdd<self_type, RHS> operator+(RHS const &rhs) const {
          return TensorAdd<self_type, RHS>(*getThis(), rhs);
        }

        template<typename RHS>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorAdd<self_type, RHS> operator-(RHS const &rhs) const {
          return TensorSubtract<self_type, RHS>(*getThis(), rhs);
        }


    };




    template<typename TENSOR_REGISTER_TYPE, typename REF_TYPE>
    class TensorLoadStore : public TensorExpressionBase<TensorLoadStore<TENSOR_REGISTER_TYPE, REF_TYPE>> {
      public:
        using self_type = TensorLoadStore<TENSOR_REGISTER_TYPE, REF_TYPE>;
        using tensor_register_type = TENSOR_REGISTER_TYPE;
        using element_type = typename TENSOR_REGISTER_TYPE::element_type;
        using index_type = typename REF_TYPE::index_type;
        using ref_type = REF_TYPE;
        using tile_type = typename REF_TYPE::tile_type;
        using result_type = TENSOR_REGISTER_TYPE;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        constexpr
        TensorLoadStore(ref_type const &ref) : m_ref(ref)
        {
        }

        TensorLoadStore(self_type const &rhs) = default;


        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &operator=(self_type const &rhs)
        {
          store(rhs);
          return *this;
        }

        template<typename RHS>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &operator=(RHS const &rhs)
        {
          store(rhs);
          return *this;
        }


        template<typename RHS>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &operator+=(RHS const &rhs)
        {
          store(TensorAdd<self_type, RHS>(*this, rhs) );
          return *this;
        }

        template<typename RHS>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &operator-=(RHS const &rhs)
        {
          store(TensorSubtract<self_type, RHS>(*this, rhs) );
          return *this;
        }

        template<typename RHS>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type operator*=(RHS const &rhs)
        {
          store(TensorMultiply<self_type, RHS>(*this, rhs) );
          return *this;
        }


        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval_full(tile_type const &tile) const {
          auto ptr = m_ref.m_pointer +
                     tile.m_begin[0]*m_ref.m_stride[0] +
                     tile.m_begin[1]*m_ref.m_stride[1];

          result_type x;
          x.load_strided(ptr,
                         m_ref.m_stride[0],
                         m_ref.m_stride[1]);

          return x;
        }



        RAJA_INLINE
        RAJA_HOST_DEVICE
        constexpr
        index_type getDimSize(index_type dim) const {
          return m_ref.m_tile.m_size[dim];
        }

        template<typename RHS>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        void store(RHS const &rhs)
        {
          // get tile size from matrix type
          index_type row_tile_size = tensor_register_type::s_dim_elem(0);
          index_type col_tile_size = tensor_register_type::s_dim_elem(1);


          // tile over full rows and columns
          tile_type tile{{0,0},{row_tile_size, col_tile_size}};
          for(tile.m_begin[0] = 0;tile.m_begin[0] < m_ref.m_tile.m_size[0]; tile.m_begin[0] += row_tile_size){
            for(tile.m_begin[1] = 0; tile.m_begin[1] < m_ref.m_tile.m_size[1];tile.m_begin[1] += col_tile_size){
              // Call rhs to evaluate this tile
              result_type x = rhs.eval_full(tile);

              // Store tile result
              auto ptr = m_ref.m_pointer +
                         tile.m_begin[0]*m_ref.m_stride[0] +
                         tile.m_begin[1]*m_ref.m_stride[1];
              x.store_strided(ptr,
                              m_ref.m_stride[0],
                              m_ref.m_stride[1]);
            }
          }
        }



      private:
        ref_type m_ref;
    };



    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorMultiply : public TensorExpressionBase<TensorMultiply<LHS_TYPE, RHS_TYPE>> {
      public:
        using self_type = TensorMultiply<LHS_TYPE, RHS_TYPE>;
        using lhs_type = LHS_TYPE;
        using rhs_type = RHS_TYPE;
        using element_type = typename LHS_TYPE::element_type;
        using index_type = typename LHS_TYPE::index_type;
        using tile_type = typename LHS_TYPE::tile_type;
        using result_type = typename LHS_TYPE::result_type;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorMultiply(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs(lhs), m_rhs(rhs)
        {}


        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval_full(tile_type const &tile) const {

//          printf("MMMult: "); tile.print();

          // get tile size from matrix type
          index_type tile_size = result_type::s_dim_elem(0);
          index_type k_size = m_lhs.getDimSize(0);
          // TODO: check that lhs and rhs are compatible
          // m_lhs.getDimSize(0) == m_rhs.getDimSize(1)
          // how do we provide checking for this kind of error?

          // tile over row of lhs and column of rhs
          tile_type lhs_tile = tile;
          tile_type rhs_tile = tile;

          result_type x(element_type(0));

          for(index_type k = 0;k < k_size; k+= tile_size){

            // evaluate both sides of operator
            lhs_tile.m_begin[1] = k;
            result_type lhs = m_lhs.eval_full(lhs_tile);

            rhs_tile.m_begin[0] = k;
            result_type rhs = m_rhs.eval_full(rhs_tile);

            // compute product into x
            x = lhs.multiply_accumulate(rhs, x);
          }

          return x;
        }



      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };


    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorAdd :  public TensorExpressionBase<TensorAdd<LHS_TYPE, RHS_TYPE>> {
      public:
        using self_type = TensorAdd<LHS_TYPE, RHS_TYPE>;
        using lhs_type = LHS_TYPE;
        using rhs_type = RHS_TYPE;
        using element_type = typename LHS_TYPE::element_type;
        using index_type = typename LHS_TYPE::index_type;
        using tile_type = typename LHS_TYPE::tile_type;
        using result_type = typename LHS_TYPE::result_type;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorAdd(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs(lhs), m_rhs(rhs)
        {}


        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval_full(tile_type const &tile) const {

          result_type x = m_lhs.eval_full(tile);
          result_type y = m_rhs.eval_full(tile);

          return x.add(y);
        }



      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorSubtract :  public TensorExpressionBase<TensorAdd<LHS_TYPE, RHS_TYPE>> {
      public:
        using self_type = TensorSubtract<LHS_TYPE, RHS_TYPE>;
        using lhs_type = LHS_TYPE;
        using rhs_type = RHS_TYPE;
        using element_type = typename LHS_TYPE::element_type;
        using index_type = typename LHS_TYPE::index_type;
        using tile_type = typename LHS_TYPE::tile_type;
        using result_type = typename LHS_TYPE::result_type;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorSubtract(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs(lhs), m_rhs(rhs)
        {}


        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval_full(tile_type const &tile) const {

          result_type x = m_lhs.eval_full(tile);
          result_type y = m_rhs.eval_full(tile);

          return x.subtract(y);
        }

      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };


  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
