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

    class TensorRegisterConcreteBase;

  namespace ET
  {
    class TensorExpressionConcreteBase{};

    template<typename RHS, typename enable = void>
    struct NormalizeOperandHelper;


    /*
     * For TensorExpression nodes, we just return them as-is.
     */
    template<typename RHS>
    struct NormalizeOperandHelper<RHS,
    typename std::enable_if<std::is_base_of<TensorExpressionConcreteBase, RHS>::value>::type>
    {
        using return_type = RHS;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        return_type normalize(RHS const &rhs){
          return rhs;
        }
    };





    template<typename RHS>
    RAJA_INLINE
    auto normalizeOperand(RHS const &rhs) ->
    typename NormalizeOperandHelper<RHS>::return_type
    {
      return NormalizeOperandHelper<RHS>::normalize(rhs);
    }


    template<typename TENSOR_REGISTER_TYPE, typename REF_TYPE>
    class TensorLoadStore;

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorAdd;

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorSubtract;

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorMultiply;

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorDivide;


    template<typename DERIVED_TYPE>
    class TensorExpressionBase :public TensorExpressionConcreteBase {
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
        TensorAdd<self_type, RHS> operator+(RHS const &rhs) const {
          return TensorAdd<self_type, RHS>(*getThis(), rhs);
        }

        template<typename RHS>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorSubtract<self_type, RHS> operator-(RHS const &rhs) const {
          return TensorSubtract<self_type, RHS>(*getThis(), rhs);
        }

        template<typename RHS>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorMultiply<self_type, RHS> operator*(RHS const &rhs) const {
          return TensorMultiply<self_type, RHS>(*getThis(), rhs);
        }

        template<typename RHS>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorMultiply<self_type, RHS> operator/(RHS const &rhs) const {
          return TensorDivide<self_type, RHS>(*getThis(), rhs);
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
        explicit
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

          store(normalizeOperand(rhs));

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

        template<typename RHS>
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type operator/=(RHS const &rhs)
        {
          store(TensorDivide<self_type, RHS>(*this, rhs) );
          return *this;
        }

        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval(TILE_TYPE const &tile) const {

          result_type x;

          x.load_ref(merge_ref_tile(m_ref, tile));

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
//          m_ref.print();
          store_expanded(rhs, camp::make_idx_seq_t<tensor_register_type::s_num_dims>{});
        }

        template<typename RHS, camp::idx_t ... DIM_SEQ>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        void store_expanded(RHS const &rhs, camp::idx_seq<DIM_SEQ...> const &)
        {
          // tile over full rows and columns
          //tile_type tile{{0,0},{row_tile_size, col_tile_size}};
          tile_type tile {
            {m_ref.m_tile.m_begin[DIM_SEQ]...},
            {tensor_register_type::s_dim_elem(DIM_SEQ)...},
          };


          // Do all of the tiling loops
          auto &full_tile = make_tensor_tile_full(tile);
          store_tile_loop(rhs, full_tile, camp::idx_seq<DIM_SEQ...>{});

        }

        template<typename RHS, typename TTYPE, camp::idx_t DIM0, camp::idx_t ... DIM_REST>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        void store_tile_loop(RHS const &rhs, TTYPE &tile, camp::idx_seq<DIM0, DIM_REST...> const &)
        {
//          printf("store_tile_loop<DIM%d> %d to %d\n",
//              (int)DIM0, (int)m_ref.m_tile.m_begin[DIM0],
//              (int)(m_ref.m_tile.m_begin[DIM0] + m_ref.m_tile.m_size[DIM0]));

          // Do the full tile sizes
          for(tile.m_begin[DIM0] = m_ref.m_tile.m_begin[DIM0];
              tile.m_begin[DIM0]+tensor_register_type::s_dim_elem(DIM0) <= m_ref.m_tile.m_begin[DIM0]+m_ref.m_tile.m_size[DIM0];
              tile.m_begin[DIM0] += tensor_register_type::s_dim_elem(DIM0)){

            // Do the next inner tiling loop
            store_tile_loop(rhs, tile, camp::idx_seq<DIM_REST...>{});
          }

          // Postamble if needed
          if(tile.m_begin[DIM0] < m_ref.m_tile.m_begin[DIM0]+m_ref.m_tile.m_size[DIM0]){

            // convert tile to a partial tile
            auto &part_tile = make_tensor_tile_partial(tile);

            // set tile size to the remainder
            part_tile.m_size[DIM0] = m_ref.m_tile.m_begin[DIM0]+m_ref.m_tile.m_size[DIM0] - tile.m_begin[DIM0];

//            printf("store_tile_loop<DIM%d>  postamble %d to %d\n",
//                (int)DIM0, (int)part_tile.m_begin[DIM0],
//                (int)(part_tile.m_size[DIM0] + part_tile.m_size[DIM0]));


            // call next inner tiling loop
            store_tile_loop(rhs, part_tile, camp::idx_seq<DIM_REST...>{});

            // reset size
            part_tile.m_size[DIM0] = m_ref.m_tile.m_size[DIM0];
          }

        }

        template<typename RHS, typename TTYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        void store_tile_loop(RHS const &rhs, TTYPE &tile, camp::idx_seq<> const &)
        {
//          printf("store_tile_loop inner: tile="); tile.print();

          // Call rhs to evaluate this tile
          result_type x = rhs.eval(tile);

          // Store result
          x.store_ref(merge_ref_tile(m_ref, tile));

//          printf("\n");
        }


        RAJA_INLINE
        RAJA_HOST_DEVICE
        void print() const {
          printf("TensorLoadStore: ");
          m_ref.print();
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


        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval(TILE_TYPE const &tile) const {

//          printf("MMMult: "); tile.print();
//          printf("  LHS:"); m_lhs.print();
//          printf("  RHS:"); m_rhs.print();

          // get tile size from matrix type
          index_type tile_size = result_type::s_dim_elem(0);
          index_type k_size = m_lhs.getDimSize(1);
          // TODO: check that lhs and rhs are compatible
          // m_lhs.getDimSize(1) == m_rhs.getDimSize(0)
          // how do we provide checking for this kind of error?

          // tile over row of lhs and column of rhs
          TILE_TYPE lhs_tile = tile;
          lhs_tile.m_size[1] = tile_size;

          TILE_TYPE rhs_tile = tile;
          rhs_tile.m_size[0] = tile_size;

          result_type x(element_type(0));

//          printf("tile_size=%d, k_size=%d, rhs_begin=%d, lhs_begin=%d\n", (int)tile_size, (int)k_size, (int)tile.m_begin[0], (int)tile.m_begin[1]);
//          printf("tile_size=%d\n", (int)tile_size);

          // Do full tiles in k
          index_type k = 0;
          for(;k+tile_size <= k_size; k+= tile_size){
//            printf("k=%d, full tile\n", (int)k);

            // evaluate both sides of operator
            lhs_tile.m_begin[1] = k;
//            printf("  lhs_tile="); lhs_tile.print();
            result_type lhs = m_lhs.eval(lhs_tile);
//            printf("%s\n", lhs.toString().c_str());


            rhs_tile.m_begin[0] = k;
//            printf("  rhs_tile="); rhs_tile.print();
            result_type rhs = m_rhs.eval(rhs_tile);
//            printf("%s\n", rhs.toString().c_str());


            // compute product into x
            x = lhs.multiply_accumulate(rhs, x);
          }
          // remainder tile in k
          if(k < k_size){
//            printf("k=%d, partial tile\n", (int)k);
            auto &lhs_part_tile = make_tensor_tile_partial(lhs_tile);
            lhs_part_tile.m_begin[1] = k;
            lhs_part_tile.m_size[1] = k_size-k;
            result_type lhs = m_lhs.eval(lhs_part_tile);
//            printf("  lhs_tile="); lhs_part_tile.print();
//            printf("%s\n", lhs.toString().c_str());



            auto &rhs_part_tile = make_tensor_tile_partial(rhs_tile);
            rhs_part_tile.m_begin[0] = k;
            rhs_part_tile.m_size[0] = k_size-k;
            result_type rhs = m_rhs.eval(rhs_part_tile);
//            printf("  rhs_tile="); rhs_part_tile.print();
//            printf("%s\n", rhs.toString().c_str());

            // compute product into x of partial tile
            x = lhs.multiply_accumulate(rhs, x);
          }

          return x;
        }



      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };


    template<typename TENSOR_TYPE>
    class TensorConstant :  public TensorExpressionBase<TensorConstant<TENSOR_TYPE>> {
      public:
        using self_type = TensorConstant<TENSOR_TYPE>;
        using tensor_type = TENSOR_TYPE;
        using element_type = typename TENSOR_TYPE::element_type;
        using result_type = tensor_type;
        using index_type = RAJA::Index_type;


        RAJA_INLINE
        RAJA_HOST_DEVICE
        explicit
        TensorConstant(tensor_type const &value) :
        m_value(value)
        {}


        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval(TILE_TYPE const &) const {
          return m_value;
        }

      private:
        tensor_type m_value;
    };


    template<typename T>
    class TensorScalarLiteral :  public TensorExpressionBase<TensorScalarLiteral<ScalarRegister<T>>> {
      public:
        using self_type = TensorScalarLiteral<T>;
        using tensor_type = ScalarRegister<T>;
        using element_type = T;
        using result_type = T;
        using index_type = RAJA::Index_type;


        RAJA_INLINE
        RAJA_HOST_DEVICE
        explicit
        TensorScalarLiteral(element_type const &value) :
        m_value(value)
        {}


        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval(TILE_TYPE const &) const {
          return m_value;
        }

      private:
        element_type m_value;
    };



    /*
     * For TensorRegister nodes, we need to wrap this in a constant value ET node
     */
    template<typename RHS>
    struct NormalizeOperandHelper<RHS,
    typename std::enable_if<std::is_base_of<RAJA::internal::TensorRegisterConcreteBase, RHS>::value>::type>
    {
        using return_type = TensorConstant<RHS>;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        return_type normalize(RHS const &rhs){
          return return_type(rhs);
        }
    };

    /*
     * For arithmetic values, we need to wrap in a constant value ET node
     */
    template<typename RHS>
    struct NormalizeOperandHelper<RHS,
    typename std::enable_if<std::is_arithmetic<RHS>::value>::type>
    {
        using return_type = TensorScalarLiteral<RHS>;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        return_type normalize(RHS const &rhs){
          return return_type(rhs);
        }
    };


    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorAdd :  public TensorExpressionBase<TensorAdd<LHS_TYPE, RHS_TYPE>> {
      public:
        using self_type = TensorAdd<LHS_TYPE, RHS_TYPE>;
        using lhs_type = LHS_TYPE;
        using rhs_type = RHS_TYPE;
        using element_type = typename LHS_TYPE::element_type;
        using index_type = typename LHS_TYPE::index_type;
        using result_type = typename LHS_TYPE::result_type;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorAdd(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs(lhs), m_rhs(rhs)
        {}


        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval(TILE_TYPE const &tile) const {

          result_type x = m_lhs.eval(tile);
          result_type y = m_rhs.eval(tile);

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
        using result_type = typename LHS_TYPE::result_type;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorSubtract(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs(lhs), m_rhs(rhs)
        {}


        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval(TILE_TYPE const &tile) const {

          result_type x = m_lhs.eval(tile);
          result_type y = m_rhs.eval(tile);

          return x.subtract(y);
        }

      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };


    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorDivide :  public TensorExpressionBase<TensorDivide<LHS_TYPE, RHS_TYPE>> {
      public:
        using self_type = TensorDivide<LHS_TYPE, RHS_TYPE>;
        using lhs_type = LHS_TYPE;
        using rhs_type = RHS_TYPE;
        using element_type = typename LHS_TYPE::element_type;
        using index_type = typename LHS_TYPE::index_type;
        using result_type = typename LHS_TYPE::result_type;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorDivide(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs(lhs), m_rhs(rhs)
        {}


        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        result_type eval(TILE_TYPE const &tile) const {

          result_type x = m_lhs.eval(tile);
          result_type y = m_rhs.eval(tile);

          return x.divide(y);
        }

      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };


    /*
     * Overload for:    arithmetic / tensorexpression

     */
    template<typename LHS, typename RHS,
      typename std::enable_if<std::is_arithmetic<LHS>::value, bool>::type = true,
      typename std::enable_if<std::is_base_of<TensorExpressionConcreteBase, RHS>::value, bool>::type = true>
    auto operator+(LHS const &lhs, RHS const &rhs) ->
    TensorAdd<typename NormalizeOperandHelper<LHS>::return_type, RHS>
    {
      return TensorAdd<typename NormalizeOperandHelper<LHS>::return_type, RHS>(NormalizeOperandHelper<LHS>::normalize(lhs), rhs);
    }

    /*
     * Overload for:    arithmetic / tensorexpression

     */
    template<typename LHS, typename RHS,
      typename std::enable_if<std::is_arithmetic<LHS>::value, bool>::type = true,
      typename std::enable_if<std::is_base_of<TensorExpressionConcreteBase, RHS>::value, bool>::type = true>
    auto operator/(LHS const &lhs, RHS const &rhs) ->
    TensorDivide<typename NormalizeOperandHelper<LHS>::return_type, RHS>
    {
      return TensorDivide<typename NormalizeOperandHelper<LHS>::return_type, RHS>(NormalizeOperandHelper<LHS>::normalize(lhs), rhs);
    }

  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
