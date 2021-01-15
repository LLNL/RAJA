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

#ifndef RAJA_pattern_tensor_ET_TensorLoadStore_HPP
#define RAJA_pattern_tensor_ET_TensorLoadStore_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "RAJA/pattern/tensor/internal/ET/ExpressionTemplateBase.hpp"
#include "RAJA/pattern/tensor/internal/TensorTileExec.hpp"
#include "RAJA/util/TypedViewBase.hpp"


namespace RAJA
{




  namespace internal
  {

  namespace ET
  {



    template<typename STORAGE, typename RHS_TYPE, typename REF_TYPE>
    struct TensorStoreFunctor
    {
        RHS_TYPE const &rhs;
        REF_TYPE const &ref;

        template<typename TILE_TYPE>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        void operator()(TILE_TYPE const &tile) const{

          // Create top-level storage
          STORAGE storage;

          // Call rhs to evaluate this tile
          rhs.eval(storage, tile);

          // Store result
          storage.store_ref(merge_ref_tile(ref, tile));
        }
    };

    template<typename STORAGE, typename RHS_TYPE, typename REF_TYPE>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    auto makeTensorStoreFunctor(RHS_TYPE const &rhs, REF_TYPE const &ref) ->
    TensorStoreFunctor<STORAGE, RHS_TYPE, REF_TYPE>
    {
      return TensorStoreFunctor<STORAGE, RHS_TYPE, REF_TYPE>{rhs, ref};
    }


    template<typename TENSOR_TYPE, typename REF_TYPE>
    class TensorLoadStore : public TensorExpressionBase<TensorLoadStore<TENSOR_TYPE, REF_TYPE>> {
      public:
        using self_type = TensorLoadStore<TENSOR_TYPE, REF_TYPE>;
        using tensor_type = TENSOR_TYPE;
        using element_type = typename TENSOR_TYPE::element_type;
        using index_type = typename REF_TYPE::index_type;
        using ref_type = REF_TYPE;
        using tile_type = typename REF_TYPE::tile_type;
        using result_type = TENSOR_TYPE;

        static constexpr camp::idx_t s_num_dims = result_type::s_num_dims;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        explicit
        TensorLoadStore(ref_type const &ref) : m_ref{ref}
        {
        }

        TensorLoadStore(self_type const &rhs) = default;


        RAJA_INLINE
        RAJA_HOST_DEVICE
        void print() const {
          printf("TensorLoadStore: ");
          m_ref.print();
        }

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

          store(normalizeOperand(rhs));

          return *this;
        }


        template<typename RHS>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &operator+=(RHS const &rhs)
        {
          store(TensorAdd<self_type, RHS>(*this, normalizeOperand(rhs)) );
          return *this;
        }

        template<typename RHS>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &operator-=(RHS const &rhs)
        {
          store(TensorSubtract<self_type, RHS>(*this, normalizeOperand(rhs)) );
          return *this;
        }

        template<typename RHS>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type operator*=(RHS const &rhs)
        {
          store(TensorMultiply<self_type, RHS>(*this, normalizeOperand(rhs)) );
          return *this;
        }

        template<typename RHS>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type operator/=(RHS const &rhs)
        {
          store(TensorDivide<self_type, RHS>(*this, normalizeOperand(rhs)) );
          return *this;
        }

        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        void eval(STORAGE &storage, TILE_TYPE const &tile) const {
          storage.load_ref(merge_ref_tile(m_ref, tile));
        }


        RAJA_INLINE
        RAJA_HOST_DEVICE
        constexpr
        index_type getDimSize(index_type dim) const {
          return m_ref.m_tile.m_size[dim];
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        void print_ast() const {
          printf("Load()");
        }

      private:

        RAJA_INLINE
        RAJA_HOST_DEVICE
        tile_type const &getTile() const {
          return m_ref.m_tile;
        }


        template<typename RHS>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        void store(RHS const &rhs)
        {
#ifdef RAJA_DEBUG_PRINT_ET_AST
          printf("Store(");
          rhs.print_ast();
          printf(")\n");
#endif
          tensorTileExec<tensor_type>(m_ref.m_tile,
              makeTensorStoreFunctor<tensor_type>(rhs, m_ref));
        }





      private:
        ref_type m_ref;
    };


  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
