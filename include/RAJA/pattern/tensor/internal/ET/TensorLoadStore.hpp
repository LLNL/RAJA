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
#include "RAJA/util/TypedViewBase.hpp"

namespace RAJA
{




  namespace internal
  {

  namespace ET
  {


    template<typename STORAGE, typename DIM_SEQ>
    struct TensorStoreExpanded;

    /**
     * Implement a dimension tiling loop
     */
    template<typename STORAGE, camp::idx_t DIM0, camp::idx_t ... DIM_REST>
    struct TensorStoreExpanded<STORAGE, camp::idx_seq<DIM0, DIM_REST...>>{

      using inner_t = TensorStoreExpanded<STORAGE, camp::idx_seq<DIM_REST...>>;

      template<typename REF, typename RHS, typename TTYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      void store(REF &ref, RHS const &rhs, TTYPE &tile){
        //          printf("store_tile_loop<DIM%d> %d to %d\n",
        //              (int)DIM0, (int)m_ref.m_tile.m_begin[DIM0],
        //              (int)(m_ref.m_tile.m_begin[DIM0] + m_ref.m_tile.m_size[DIM0]));

        //auto const &store_tile = getTile();
        auto orig_begin = ref.m_tile.m_begin[DIM0];
        auto orig_size = ref.m_tile.m_size[DIM0];

        // Do the full tile sizes
        for(tile.m_begin[DIM0] = orig_begin;

            tile.m_begin[DIM0] +  STORAGE::s_dim_elem(DIM0) <=
                orig_begin+orig_size;

            tile.m_begin[DIM0] += STORAGE::s_dim_elem(DIM0)){

          // Do the next inner tiling loop
          inner_t::store(ref, rhs, tile);
        }

        // Postamble if needed
        if(tile.m_begin[DIM0] <
            orig_begin + orig_size)
        {

          // convert tile to a partial tile
          auto &part_tile = make_tensor_tile_partial(tile);

          // set tile size to the remainder
          part_tile.m_size[DIM0] =
              orig_begin +
              orig_size -
              tile.m_begin[DIM0];

//            printf("store_tile_loop<DIM%d>  postamble %d to %d\n",
//                (int)DIM0, (int)part_tile.m_begin[DIM0],
//                (int)(part_tile.m_size[DIM0] + part_tile.m_size[DIM0]));


          // Do the next inner tiling loop
          inner_t::store(ref, rhs, part_tile);
        }

        // reset tile dimension
        tile.m_begin[DIM0] = orig_begin;
        tile.m_size[DIM0] = orig_size;
      }

    };

    /**
     * Termination of nested loop:  execute evaluation of ET
     */
    template<typename STORAGE>
    struct TensorStoreExpanded<STORAGE, camp::idx_seq<>>{

      template<typename REF, typename RHS, typename TTYPE>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      void store(REF &ref, RHS const &rhs, TTYPE const &tile){

        // Create top-level storage
        STORAGE storage;

        // Call rhs to evaluate this tile
        rhs.eval(storage, tile);

        // Store result
        storage.store_ref(merge_ref_tile(ref, tile));
      }

    };



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

          store_expanded(rhs, camp::make_idx_seq_t<tensor_type::s_num_dims>{});
        }

        template<typename RHS, camp::idx_t ... DIM_SEQ>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        void store_expanded(RHS const &rhs, camp::idx_seq<DIM_SEQ...> const &)
        {
          // tile over full rows and columns
          //tile_type tile{{0,0},{row_tile_size, col_tile_size}};
          tile_type tile {
            {getTile().m_begin[DIM_SEQ]...},
            {tensor_type::s_dim_elem(DIM_SEQ)...},
          };


          // Promote the tile type to a "full-tile" so that the full-element
          // register operations are used.
          // Any of the tiling loops can demote this to a partial-tile when
          // they do postamble execution
          auto &full_tile = make_tensor_tile_full(tile);

          // Do all of the tiling loops
          //store_tile_loop(rhs, full_tile, camp::idx_seq<DIM_SEQ...>{});

          using tensor_store_t =
              TensorStoreExpanded<tensor_type, camp::idx_seq<DIM_SEQ...>>;

          tensor_store_t::store(m_ref, rhs, full_tile);

        }




      private:
        ref_type m_ref;
    };


  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
