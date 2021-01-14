/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header defining expression template behavior for operator*
 *
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_pattern_tensor_ET_MultiplyOperator_HPP
#define RAJA_pattern_tensor_ET_MultiplyOperator_HPP



namespace RAJA
{

  namespace internal
  {

  namespace ET
  {


    /*!
     * Provides default multiply, multiply add, and multiply subtract
     * operations.
     *
     * If the operands are both matrices, we perform a matrix-matrix multiply.
     * Otherwise, we perform element-wise operations.
     */
    template<typename LHS_TYPE, typename RHS_TYPE, class ENABLE = void>
    struct MultiplyOperator {

        using result_type = typename LHS_TYPE::result_type;
        using tile_type = typename LHS_TYPE::tile_type;
        static constexpr camp::idx_t s_num_dims = LHS_TYPE::s_num_dims;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void print_ast() {
          printf("Elemental");
        }


        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        int getDimSize(int dim, LHS_TYPE const &lhs, RHS_TYPE const &rhs) {
          return dim == 0 ? lhs.getDimSize(0) : rhs.getDimSize(1);
        }

        /*!
         * Evaluate operands and perform element-wise multiply
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void multiply(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){

          // evaluate LHS in-place
          lhs.eval(result, tile);

          // evaluate RHS in temporary
          STORAGE rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // compute result
          result.inplace_multiply(rhs_tmp);
        }


        /*!
         * Evaluate operands and perform element-wise multiply add
         */
        template<typename STORAGE, typename TILE_TYPE, typename ADD_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void multiply_add(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, ADD_TYPE const &add){

          // evaluate LHS in-place
          lhs.eval(result, tile);

          // evaluate RHS in temporary
          STORAGE rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // evaluate ADD in temporary
          STORAGE add_tmp;
          add.eval(add_tmp, tile);

          // compute result
          result.inplace_multiply_add(rhs_tmp, add_tmp);
        }


        /*!
         * Evaluate operands and perform element-wise multiply subtract
         */
        template<typename STORAGE, typename TILE_TYPE, typename SUB_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void multiply_subtract(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, SUB_TYPE const &sub){
          // evaluate LHS in-place
          lhs.eval(result, tile);

          // evaluate RHS in temporary
          STORAGE rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // evaluate ADD in temporary
          STORAGE sub_tmp;
          sub.eval(sub_tmp, tile);

          // compute result
          result.inplace_multiply_subtract(rhs_tmp, sub_tmp);
        }


    };


    /*!
     * Specialization that provides multiplying a scalar * tensor
     */
    template<typename LHS_TYPE, typename RHS_TYPE>
    struct MultiplyOperator<LHS_TYPE, RHS_TYPE,
    typename std::enable_if<LHS_TYPE::s_num_dims == 0>::type>
    {

        using result_type = typename RHS_TYPE::result_type;
        using tile_type = typename RHS_TYPE::tile_type;
        static constexpr camp::idx_t s_num_dims = RHS_TYPE::s_num_dims;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void print_ast() {
          printf("Scale");
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        int getDimSize(int dim, LHS_TYPE const &, RHS_TYPE const &rhs) {
          return rhs.getDimSize(dim);
        }

        /*!
         * Evaluate operands and perform element-wise multiply
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void multiply(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
          // evaluate RHS in-place
          rhs.eval(result, tile);

          // evaluate scalar
          typename LHS_TYPE::result_type lhs_tmp;
          lhs.eval(lhs_tmp, tile);

          // scale it by the LHS
          result.inplace_scale(lhs_tmp);
        }




        /*!
         * Evaluate operands and perform element-wise multiply add
         */
        template<typename STORAGE, typename TILE_TYPE, typename ADD_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void multiply_add(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, ADD_TYPE const &add){
          // evaluate RHS in-place
          rhs.eval(result, tile);

          // evaluate scalar
          typename LHS_TYPE::result_type lhs_tmp;
          lhs.eval(lhs_tmp, tile);

          // scale it by the LHS
          result.inplace_scale(lhs_tmp);

          // evaluate ADD in temporary
          STORAGE add_tmp;
          add.eval(add_tmp, tile);

          // compute result
          result.inplace_add(add_tmp);
        }


        /*!
         * Evaluate operands and perform element-wise multiply subtract
         */
        template<typename STORAGE, typename TILE_TYPE, typename SUB_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        result_type multiply_subtract(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, SUB_TYPE const &sub){
          // evaluate RHS in-place
          rhs.eval(result, tile);

          // evaluate scalar
          typename LHS_TYPE::result_type lhs_tmp;
          lhs.eval(lhs_tmp, tile);

          // scale it by the LHS
          result.inplace_scale(lhs_tmp);

          // evaluate ADD in temporary
          STORAGE sub_tmp;
          sub.eval(sub_tmp, tile);

          // compute result
          result.inplace_add(sub_tmp);
        }
    };


    /*!
     * Specialization that provides multiplying a tensor*scalar
     */
    template<typename LHS_TYPE, typename RHS_TYPE>
    struct MultiplyOperator<LHS_TYPE, RHS_TYPE,
    typename std::enable_if<RHS_TYPE::s_num_dims == 0>::type>
    {

        using result_type = typename LHS_TYPE::result_type;
        using tile_type = typename LHS_TYPE::tile_type;
        static constexpr camp::idx_t s_num_dims = LHS_TYPE::s_num_dims;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void print_ast() {
          printf("Scale");
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        int getDimSize(int dim, LHS_TYPE const &lhs, RHS_TYPE const &) {
          return lhs.getDimSize(dim);
        }

        /*!
         * Evaluate operands and perform element-wise multiply
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void multiply(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
          // evaluate RHS in-place
          lhs.eval(result, tile);

          // evaluate scalar
          typename RHS_TYPE::result_type rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // scale it by the LHS
          result.inplace_scale(rhs_tmp);
        }




        /*!
         * Evaluate operands and perform element-wise multiply add
         */
        template<typename STORAGE, typename TILE_TYPE, typename ADD_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void multiply_add(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, ADD_TYPE const &add){
          // evaluate RHS in-place
          lhs.eval(result, tile);

          // evaluate scalar
          typename RHS_TYPE::result_type rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // scale it by the LHS
          result.inplace_scale(rhs_tmp);

          // evaluate ADD in temporary
          STORAGE add_tmp;
          add.eval(add_tmp, tile);

          // compute result
          result.inplace_add(add_tmp);
        }


        /*!
         * Evaluate operands and perform element-wise multiply subtract
         */
        template<typename STORAGE, typename TILE_TYPE, typename SUB_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        result_type multiply_subtract(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, SUB_TYPE const &sub){
          // evaluate RHS in-place
          lhs.eval(result, tile);

          // evaluate scalar
          typename RHS_TYPE::result_type rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // scale it by the RHS
          result.inplace_scale(rhs_tmp);

          // evaluate ADD in temporary
          STORAGE sub_tmp;
          sub.eval(sub_tmp, tile);

          // compute result
          result.inplace_add(sub_tmp);
        }
    };


    /*!
     * Specialization for matrix-vector right multiplication.
     *
     * By default the A*x operator for two matrices produces a matrix-vector
     * multiplication.
     *
     * The right hand side vector is always treated as a column vector.
     *
     * The resulting vector type is inherited from the RHS
     *
     *
     */
    template<typename LHS_TYPE, typename RHS_TYPE>
    struct MultiplyOperator<LHS_TYPE, RHS_TYPE,
    typename std::enable_if<LHS_TYPE::s_num_dims == 2 && RHS_TYPE::s_num_dims==1>::type>
    {

      using lhs_type = LHS_TYPE;
      using rhs_type = RHS_TYPE;
      using element_type = typename LHS_TYPE::element_type;
      using index_type = typename LHS_TYPE::index_type;
      using tile_type = typename LHS_TYPE::tile_type;
      using result_type = typename RHS_TYPE::result_type;
      static constexpr camp::idx_t s_num_dims = 1;

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void print_ast() {
        printf("Matrx*Vector");
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      int getDimSize(int dim, LHS_TYPE const &, RHS_TYPE const &rhs) {
        return dim == 0 ? rhs.getDimSize(0) : 0;
      }

      /*!
       * Evaluate operands and perform element-wise multiply
       */
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
        // clear result
        result.broadcast(element_type(0));

        // multiply lhs and rhs into result
        multiply_into_result(result, tile, lhs, rhs);
      }

      template<typename STORAGE, typename TILE_TYPE, typename ADD_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_add(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, ADD_TYPE const &add){
        // evaluate add into result
        add.eval(result, tile);

        // multiply lhs and rhs into result
        multiply_into_result(result, tile, lhs, rhs);
      }

    private:
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_into_result(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &et_lhs, RHS_TYPE const &et_rhs)
      {
        using LHS_STORAGE = typename LHS_TYPE::result_type;

        // get tile size from matrix type
        index_type tile_size = lhs_type::result_type::s_dim_elem(1);
        index_type k_size = et_lhs.getDimSize(1);
        // TODO: check that lhs and rhs are compatible
        // m_lhs.getDimSize(1) == m_rhs.getDimSize(0)
        // how do we provide checking for this kind of error?

        // tile over row of lhs and column of rhs
        auto lhs_tile = LHS_TYPE::result_type::s_get_default_tile();
        lhs_tile.m_begin[0] = tile.m_begin[0];
        lhs_tile.m_size[0] = tile.m_size[0];
        lhs_tile.m_size[1] = tile_size;

        TILE_TYPE rhs_tile = tile;
        rhs_tile.m_size[0] = tile_size;

        // Do full tiles in k
        index_type k = 0;
        for(;k+tile_size <= k_size; k+= tile_size){

          // evaluate both sides of operator
          lhs_tile.m_begin[1] = k;
          LHS_STORAGE lhs;
          et_lhs.eval(lhs, lhs_tile);

          rhs_tile.m_begin[0] = k;
          STORAGE rhs;
          et_rhs.eval(rhs, rhs_tile);

          // accumulate product
          lhs.right_multiply_vector_accumulate(result, rhs);
        }
        // remainder tile in k
        if(k < k_size){
          auto &lhs_part_tile = make_tensor_tile_partial(lhs_tile);
          lhs_part_tile.m_begin[1] = k;
          lhs_part_tile.m_size[1] = k_size-k;
          LHS_STORAGE lhs;
          et_lhs.eval(lhs, lhs_part_tile);

          auto &rhs_part_tile = make_tensor_tile_partial(rhs_tile);
          rhs_part_tile.m_begin[0] = k;
          rhs_part_tile.m_size[0] = k_size-k;
          STORAGE rhs;
          et_rhs.eval(rhs, rhs_part_tile);

          // accumulate product of partial tile
          lhs.right_multiply_vector_accumulate(result, rhs);
        }

      }

    };



    /*!
     * Specialization for vector*matrix left multiplication.
     *
     * By default the x'*A operator for two matrices produces a vector-matrix
     * multiplication.
     *
     * The left hand side vector is always treated as a row vector.
     *
     * The resulting vector type is inherited from the LHS
     *
     *
     */
    template<typename LHS_TYPE, typename RHS_TYPE>
    struct MultiplyOperator<LHS_TYPE, RHS_TYPE,
    typename std::enable_if<LHS_TYPE::s_num_dims == 1 && RHS_TYPE::s_num_dims==2>::type>
    {

      using lhs_type = LHS_TYPE;
      using rhs_type = RHS_TYPE;
      using element_type = typename LHS_TYPE::element_type;
      using index_type = typename LHS_TYPE::index_type;
      using tile_type = typename RHS_TYPE::tile_type;
      using result_type = typename LHS_TYPE::result_type;
      static constexpr camp::idx_t s_num_dims = 1;

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void print_ast() {
        printf("Vector*Matrix");
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      int getDimSize(int dim, LHS_TYPE const &lhs, RHS_TYPE const &) {
        return dim == 0 ? lhs.getDimSize(0) : 0;
      }

      /*!
       * Evaluate operands and perform element-wise multiply
       */
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
        // clear result
        result.broadcast(element_type(0));

        // multiply lhs and rhs into result
        multiply_into_result(result, tile, lhs, rhs);
      }

      template<typename STORAGE, typename TILE_TYPE, typename ADD_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_add(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, ADD_TYPE const &add){
        // evaluate add into result
        add.eval(result, tile);

        // multiply lhs and rhs into result
        multiply_into_result(result, tile, lhs, rhs);
      }

    private:
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_into_result(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &et_lhs, RHS_TYPE const &et_rhs)
      {
        using RHS_STORAGE = typename RHS_TYPE::result_type;

        // get tile size from matrix type
        index_type tile_size = rhs_type::result_type::s_dim_elem(1);
        index_type k_size = et_rhs.getDimSize(1);
        // TODO: check that lhs and rhs are compatible
        // m_lhs.getDimSize(1) == m_rhs.getDimSize(0)
        // how do we provide checking for this kind of error?

        // tile over row of lhs and column of rhs
        auto rhs_tile = RHS_TYPE::result_type::s_get_default_tile();
        rhs_tile.m_begin[1] = tile.m_begin[0];
        rhs_tile.m_size[1] = tile.m_size[0];
        rhs_tile.m_size[0] = tile_size;

        TILE_TYPE lhs_tile = tile;
        lhs_tile.m_size[0] = tile_size;

        // Do full tiles in k
        index_type k = 0;
        for(;k+tile_size <= k_size; k+= tile_size){

          // evaluate both sides of operator
          rhs_tile.m_begin[0] = k;
          RHS_STORAGE rhs;
          et_rhs.eval(rhs, rhs_tile);

          lhs_tile.m_begin[0] = k;
          STORAGE lhs;
          et_lhs.eval(lhs, lhs_tile);

          // accumulate product
          rhs.left_multiply_vector_accumulate(result, lhs);

        }
        // remainder tile in k
        if(k < k_size){
          auto &rhs_part_tile = make_tensor_tile_partial(rhs_tile);
          rhs_part_tile.m_begin[0] = k;
          rhs_part_tile.m_size[0] = k_size-k;
          RHS_STORAGE rhs;
          et_rhs.eval(rhs, rhs_tile);

          auto &lhs_part_tile = make_tensor_tile_partial(lhs_tile);
          lhs_part_tile.m_begin[0] = k;
          lhs_part_tile.m_size[0] = k_size-k;
          STORAGE lhs;
          et_lhs.eval(lhs, lhs_tile);

          // compute product into x of partial tile
          rhs.left_multiply_vector_accumulate(result, lhs);
        }

      }

    };



    /*!
     * Specialization for matrix-matrix multiplication.
     *
     * By default the A*B operator for two matrices produces a matrix-matrix
     * multiplication.
     *
     */
    template<typename LHS_TYPE, typename RHS_TYPE>
    struct MultiplyOperator<LHS_TYPE, RHS_TYPE,
    typename std::enable_if<LHS_TYPE::s_num_dims == 2 && RHS_TYPE::s_num_dims==2>::type>
    {

      using lhs_type = LHS_TYPE;
      using rhs_type = RHS_TYPE;
      using element_type = typename LHS_TYPE::element_type;
      using index_type = typename LHS_TYPE::index_type;
      using tile_type = typename LHS_TYPE::tile_type;
      using result_type = typename RHS_TYPE::result_type;
      static constexpr camp::idx_t s_num_dims = 2;

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void print_ast() {
        printf("Matrx*Matrix");
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      int getDimSize(int dim, LHS_TYPE const &lhs, RHS_TYPE const &rhs) {
        return dim == 0 ? lhs.getDimSize(0) : rhs.getDimSize(1);
      }

      /*!
       * Evaluate operands and perform element-wise multiply
       */
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
        // clear result
        result.broadcast(element_type(0));

        // multiply lhs and rhs into result
        multiply_into_result(result, tile, lhs, rhs);
      }

      template<typename STORAGE, typename TILE_TYPE, typename ADD_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_add(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs, ADD_TYPE const &add){
        // evaluate add into result
        add.eval(result, tile);

        // multiply lhs and rhs into result
        multiply_into_result(result, tile, lhs, rhs);
      }

    private:
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_into_result(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &et_lhs, RHS_TYPE const &et_rhs)
      {

        // get tile size from matrix type
        index_type tile_size = result_type::s_dim_elem(1);
        index_type k_size = et_lhs.getDimSize(1);
        // TODO: check that lhs and rhs are compatible
        // m_lhs.getDimSize(1) == m_rhs.getDimSize(0)
        // how do we provide checking for this kind of error?

        // tile over row of lhs and column of rhs
        TILE_TYPE lhs_tile = tile;
        lhs_tile.m_size[1] = tile_size;

        TILE_TYPE rhs_tile = tile;
        rhs_tile.m_size[0] = tile_size;



        // Do full tiles in k
        index_type k = 0;
        for(;k+tile_size <= k_size; k+= tile_size){

          // evaluate both sides of operator
          lhs_tile.m_begin[1] = k;
          STORAGE lhs;
          et_lhs.eval(lhs, lhs_tile);

          rhs_tile.m_begin[0] = k;
          STORAGE rhs;
          et_rhs.eval(rhs, rhs_tile);

          // accumulate product
          lhs.matrix_multiply_accumulate(result, rhs);
        }
        // remainder tile in k
        if(k < k_size){
          auto &lhs_part_tile = make_tensor_tile_partial(lhs_tile);
          lhs_part_tile.m_begin[1] = k;
          lhs_part_tile.m_size[1] = k_size-k;
          STORAGE lhs;
          et_lhs.eval(lhs, lhs_part_tile);

          auto &rhs_part_tile = make_tensor_tile_partial(rhs_tile);
          rhs_part_tile.m_begin[0] = k;
          rhs_part_tile.m_size[0] = k_size-k;
          STORAGE rhs;
          et_rhs.eval(rhs, rhs_part_tile);

          // accumulate product
          lhs.matrix_multiply_accumulate(result, rhs);
        }
      }

    };




  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
