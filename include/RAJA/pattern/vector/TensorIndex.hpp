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

#ifndef RAJA_pattern_vector_tensorindex_HPP
#define RAJA_pattern_vector_tensorindex_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"


namespace RAJA
{

  namespace internal {
    template<typename ARG>
    struct TensorIndexTraits;
  }

  template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM>
  class TensorIndex {
    public:
      using self_type = TensorIndex<IDX, TENSOR_TYPE, DIM>;
      using value_type = strip_index_type_t<IDX>;
      using index_type = IDX;
      using tensor_type = TENSOR_TYPE;
      using tensor_traits = internal::TensorIndexTraits<self_type>;

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      constexpr
      self_type all(){
        return self_type(-1, -1);
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      constexpr
      value_type num_elem(){
        return tensor_traits::num_elem();
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorIndex() : m_index(index_type(0)), m_length(num_elem()) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      explicit TensorIndex(index_type value) : m_index(value), m_length(num_elem()) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorIndex(index_type value, value_type length) : m_index(value), m_length(length) {}

      template<typename T, camp::idx_t D>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      TensorIndex(TensorIndex<IDX, T, D> const &c) : m_index(*c), m_length(c.size()) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      index_type const &operator*() const {
        return m_index;
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      value_type size() const {
        return m_length;
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      value_type dim() const {
        return DIM;
      }

    private:
      index_type m_index;
      value_type m_length;
  };


  /*!
   * Index that specifies the starting element index of a Vector
   */
  template<typename IDX, typename VECTOR_TYPE>
  using VectorIndex =  TensorIndex<IDX, VECTOR_TYPE, 0>;

  /*!
   * Index that specifies the starting Row index of a matrix
   */
  template<typename IDX, typename MATRIX_TYPE>
  using RowIndex =  TensorIndex<IDX, MATRIX_TYPE, 0>;

  /*!
   * Index that specifies the starting Column index of a matrix
   */
  template<typename IDX, typename MATRIX_TYPE>
  using ColIndex =  TensorIndex<IDX, MATRIX_TYPE, 1>;








  namespace internal{


    /* Partial specialization for the strip_index_type_t helper in
       IndexValue.hpp
    */
    template<typename IDX, typename VECTOR_TYPE, camp::idx_t DIM>
    struct StripIndexTypeT<TensorIndex<IDX, VECTOR_TYPE, DIM>>
    {
        using type = typename TensorIndex<IDX, VECTOR_TYPE, DIM>::value_type;
    };


    // Helper that strips the Vector type from an argument
    template<typename ARG>
    struct TensorIndexTraits {
        using arg_type = ARG;
        using value_type = strip_index_type_t<ARG>;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        bool isTensorIndex(){
          return false;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        arg_type const &strip(arg_type const &arg){
          return arg;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        value_type size(arg_type const &){
          return 1;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        value_type dim(){
          return 0;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        value_type num_elem(){
          return 1;
        }
    };

    template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM>
    struct TensorIndexTraits<TensorIndex<IDX, TENSOR_TYPE, DIM>> {
        using index_type = TensorIndex<IDX, TENSOR_TYPE, DIM>;
        using arg_type = IDX;
        using value_type = strip_index_type_t<IDX>;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        bool isTensorIndex(){
          return true;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        arg_type const &strip(index_type const &arg){
          return *arg;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        value_type size(index_type const &arg){
          return arg.size();
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        value_type dim(){
          return DIM;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        value_type num_elem(){
          return TENSOR_TYPE::s_dim_elem(DIM);
        }
    };

    /*
     * Returns vector size of argument.
     *
     * For scalars, always returns 1.
     *
     * For VectorIndex types, returns the number of vector lanes.
     */
    template<typename ARG>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    bool isTensorIndex()
    {
      return TensorIndexTraits<ARG>::isTensorIndex();
    }

    template<typename ARG>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    auto stripTensorIndex(ARG const &arg) ->
    typename TensorIndexTraits<ARG>::arg_type const &
    {
      return TensorIndexTraits<ARG>::strip(arg);
    }

    /*
     * Returns vector size of argument.
     *
     * For scalars, always returns 1.
     *
     * For VectorIndex types, returns the number of vector lanes.
     */
    template<typename ARG, typename IDX>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    IDX getTensorSize(ARG const &arg, IDX dim_size)
    {
      return TensorIndexTraits<ARG>::size(arg) >= 0 ?
          IDX(TensorIndexTraits<ARG>::size(arg)) :
          dim_size;
    }

    /*
     * Returns vector dim of argument.
     *
     * For scalars, always returns 0.
     *
     * For VectorIndex types, returns the DIM argument.
     * For vector_exec, this is always 0
     *
     * For matrices, DIM means:
     *   0 : Row
     *   1 : Column
     */
    template<typename ARG>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    auto getTensorDim() ->
      decltype(TensorIndexTraits<ARG>::dim())
    {
      return TensorIndexTraits<ARG>::dim();
    }

    /*
     * Lambda<N, Seg<X>>  overload that matches VectorIndex types, and properly
     * includes the vector length with them
     */
    template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM, camp::idx_t id>
    struct LambdaSegExtractor<TensorIndex<IDX, TENSOR_TYPE, DIM>, id>
    {

      template<typename Data>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      static TensorIndex<IDX, TENSOR_TYPE, DIM> extract(Data &&data)
      {
        return TensorIndex<IDX, TENSOR_TYPE, DIM>(
            camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)],
            camp::get<id>(data.vector_sizes));
      }

    };

    /*
     * Lambda<N, Seg<X>>  overload that matches VectorIndex types, and properly
     * includes the vector length with them
     */
    template<typename IDX, typename TENSOR_TYPE, camp::idx_t DIM, camp::idx_t id>
    struct LambdaOffsetExtractor<TensorIndex<IDX, TENSOR_TYPE, DIM>, id>
    {

      template<typename Data>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      static TensorIndex<IDX, TENSOR_TYPE, DIM> extract(Data &&data)
      {
        return TensorIndex<IDX, TENSOR_TYPE, DIM>(
            IDX(camp::get<id>(data.offset_tuple)), // convert offset type to IDX
            camp::get<id>(data.vector_sizes));
      }

    };

  } // namespace internal



  template<typename IDX, typename MATRIX_TYPE, camp::idx_t DIM>
  struct TensorDim {
  };

  template<typename IDX, typename MATRIX_TYPE>
  using RowDim = TensorDim<IDX, MATRIX_TYPE, 0>;

  template<typename IDX, typename MATRIX_TYPE>
  using ColDim = TensorDim<IDX, MATRIX_TYPE, 1>;


  namespace internal
  {

  namespace ET
  {




    template<typename INDEX_TYPE, camp::idx_t NUM_DIMS>
    struct TensorTile
    {
        using index_type = INDEX_TYPE;
        index_type m_begin[NUM_DIMS];
        index_type m_size[NUM_DIMS];

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



    template<typename REG_MATRIX_TYPE, typename REF_TYPE>
    class TensorLoadStore;


    template<typename TENSOR_REG_TYPE, typename POINTER_TYPE, typename INDEX_TYPE, camp::idx_t NUM_DIMS, camp::idx_t STRIDE_ONE_DIM = -1>
    struct TensorRef;






    template<typename TENSOR_REG_TYPE, typename POINTER_TYPE, typename INDEX_TYPE, camp::idx_t NUM_DIMS, camp::idx_t STRIDE_ONE_DIM>
    struct TensorRef
    {
        using self_type = TensorRef<TENSOR_REG_TYPE, POINTER_TYPE, INDEX_TYPE, NUM_DIMS, STRIDE_ONE_DIM>;
        using tile_type = TensorTile<INDEX_TYPE, NUM_DIMS>;

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


    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorMultiply;

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorAdd;

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorSubtract;

    template<typename DERIVED_TYPE>
    class ETNode  {
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




    template<typename REG_MATRIX_TYPE, typename REF_TYPE>
    class TensorLoadStore : public ETNode<TensorLoadStore<REG_MATRIX_TYPE, REF_TYPE>> {
      public:
        using self_type = TensorLoadStore<REG_MATRIX_TYPE, REF_TYPE>;
        using tensor_type = REG_MATRIX_TYPE;
        using element_type = typename REG_MATRIX_TYPE::element_type;
        using index_type = camp::idx_t;
        using ref_type = REF_TYPE;
        using tile_type = TensorTile<index_type, 2>;
        using result_type = REG_MATRIX_TYPE;

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
        void eval_full(tile_type const &tile,
                              result_type &x) const {
          auto ptr = m_ref.m_pointer +
                     tile.m_begin[0]*m_ref.m_stride[0] +
                     tile.m_begin[1]*m_ref.m_stride[1];
          x.load_strided(ptr,
                         m_ref.m_stride[0],
                         m_ref.m_stride[1]);
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        void eval_partial(tile_type const &tile,
                          result_type &x){
          auto ptr = m_ref.m_pointer +
                     tile.m_begin[0]*m_ref.m_stride[0] +
                     tile.m_begin[1]*m_ref.m_stride[1];

          x.load_strided_nm(ptr,
                            tile.m_size[0],
                            tile.m_size[0],
                            m_ref.m_stride[0],
                            m_ref.m_stride[1]);
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
          index_type row_tile_size = tensor_type::s_dim_elem(0);
          index_type col_tile_size = tensor_type::s_dim_elem(1);


          // tile over full rows and columns
          tile_type tile{{0,0},{row_tile_size, col_tile_size}};
          for(tile.m_begin[0] = 0;tile.m_begin[0] < m_ref.m_tile.m_size[0]; tile.m_begin[0] += row_tile_size){
            for(tile.m_begin[1] = 0; tile.m_begin[1] < m_ref.m_tile.m_size[1];tile.m_begin[1] += col_tile_size){
              // Call rhs to evaluate this tile
              result_type x;
              rhs.eval_full(tile, x);

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
    class TensorMultiply : public ETNode<TensorMultiply<LHS_TYPE, RHS_TYPE>> {
      public:
        using self_type = TensorMultiply<LHS_TYPE, RHS_TYPE>;
        using lhs_type = LHS_TYPE;
        using rhs_type = RHS_TYPE;
        using element_type = typename LHS_TYPE::element_type;
        using index_type = typename LHS_TYPE::index_type;
        using tile_type = TensorTile<index_type, 2>;
        using result_type = typename LHS_TYPE::result_type;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorMultiply(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs(lhs), m_rhs(rhs)
        {}


        RAJA_INLINE
        RAJA_HOST_DEVICE
        void eval_full(tile_type const &tile,
                              result_type &x) const {

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

          for(index_type k = 0;k < k_size; k+= tile_size){

            // evaluate both sides of operator
            result_type lhs;
            lhs_tile.m_begin[1] = k;

            m_lhs.eval_full(lhs_tile, lhs);

            result_type rhs;
            rhs_tile.m_begin[0] = k;
            m_rhs.eval_full(rhs_tile, rhs);

            // compute product into x
            x = lhs.multiply_accumulate(rhs, x);
          }
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        void eval_partial(tile_type const &tile,
                          result_type &x){
          eval_full(tile, x);
        }

      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };


    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorAdd :  public ETNode<TensorAdd<LHS_TYPE, RHS_TYPE>> {
      public:
        using self_type = TensorAdd<LHS_TYPE, RHS_TYPE>;
        using lhs_type = LHS_TYPE;
        using rhs_type = RHS_TYPE;
        using element_type = typename LHS_TYPE::element_type;
        using index_type = typename LHS_TYPE::index_type;
        using tile_type = TensorTile<index_type, 2>;
        using result_type = typename LHS_TYPE::result_type;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorAdd(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs(lhs), m_rhs(rhs)
        {}


        RAJA_INLINE
        RAJA_HOST_DEVICE
        void eval_full(tile_type const &tile,
                              result_type &x) const {

          m_lhs.eval_full(tile, x);

          result_type y;
          m_rhs.eval_full(tile, y);

          x = x.add(y);
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        void eval_partial(tile_type const &tile,
                          result_type &x){
          eval_full(tile, x);
        }

      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };

    template<typename LHS_TYPE, typename RHS_TYPE>
    class TensorSubtract :  public ETNode<TensorAdd<LHS_TYPE, RHS_TYPE>> {
      public:
        using self_type = TensorSubtract<LHS_TYPE, RHS_TYPE>;
        using lhs_type = LHS_TYPE;
        using rhs_type = RHS_TYPE;
        using element_type = typename LHS_TYPE::element_type;
        using index_type = typename LHS_TYPE::index_type;
        using tile_type = TensorTile<index_type, 2>;
        using result_type = typename LHS_TYPE::result_type;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        TensorSubtract(lhs_type const &lhs, rhs_type const &rhs) :
        m_lhs(lhs), m_rhs(rhs)
        {}


        RAJA_INLINE
        RAJA_HOST_DEVICE
        void eval_full(tile_type const &tile,
                              result_type &x) const {

          m_lhs.eval_full(tile, x);

          result_type y;
          m_rhs.eval_full(tile, y);

          x = x.subtract(y);
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        void eval_partial(tile_type const &tile,
                          result_type &x){
          eval_full(tile, x);
        }

      private:
        lhs_type m_lhs;
        rhs_type m_rhs;
    };


  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
