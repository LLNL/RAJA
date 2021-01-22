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
    template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE, class ENABLE = void>
    struct MultiplyOperator {

        using result_type = typename LEFT_OPERAND_TYPE::result_type;
        using tile_type = typename LEFT_OPERAND_TYPE::tile_type;
        static constexpr camp::idx_t s_num_dims = LEFT_OPERAND_TYPE::s_num_dims;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void print_ast() {
          printf("Elemental");
        }


        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        int getDimSize(int dim, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right) {
          return dim == 0 ? left.getDimSize(0) : right.getDimSize(1);
        }

        /*!
         * Evaluate operands and perform element-wise multiply
         */
        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        auto multiply(TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right) ->
          decltype(left.eval(tile) * right.eval(tile))
        {
          return left.eval(tile) * right.eval(tile);
        }


        /*!
         * Evaluate operands and perform element-wise multiply add
         */
        template<typename STORAGE, typename TILE_TYPE, typename ADD_OPERAND_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        auto multiply_add(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right, ADD_OPERAND_TYPE const &add) ->
          decltype(left.eval(tile).multiply_add(right.eval(tile), add.eval(tile)))
        {
          return left.eval(tile).multiply_add(right.eval(tile), add.eval(tile));
        }


        /*!
         * Evaluate operands and perform element-wise multiply subtract
         */
        template<typename STORAGE, typename TILE_TYPE, typename SUBTRACT_OPERAND_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        auto multiply_subtract(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right, SUBTRACT_OPERAND_TYPE const &subtract) ->
          decltype(left.eval(tile).multiply_subtract(right.eval(tile), subtract.eval(tile)))
        {
          return left.eval(tile).multiply_subtract(right.eval(tile), subtract.eval(tile));
        }


    };

#if 1
    /*!
     * Specialization that provides multiplying a scalar * tensor
     */
    template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
    struct MultiplyOperator<LEFT_OPERAND_TYPE, RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 0>::type>
    {

        using result_type = typename RIGHT_OPERAND_TYPE::result_type;
        using tile_type = typename RIGHT_OPERAND_TYPE::tile_type;
        static constexpr camp::idx_t s_num_dims = RIGHT_OPERAND_TYPE::s_num_dims;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void print_ast() {
          printf("Scale");
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        int getDimSize(int dim, LEFT_OPERAND_TYPE const &, RIGHT_OPERAND_TYPE const &right) {
          return right.getDimSize(dim);
        }

        /*!
         * Evaluate operands and perform scaling operation
         */
        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        auto multiply(TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right) ->
          decltype(right.eval(tile) * left.eval(tile))
        {
          return right.eval(tile) * left.eval(tile);
        }



        /*!
         * Evaluate operands and perform element-wise multiply add
         */
        template<typename STORAGE, typename TILE_TYPE, typename ADD_OPERAND_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        auto multiply_add(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right, ADD_OPERAND_TYPE const &add) ->
          decltype(right.eval(tile).scale(left.eval(tile)) + add.eval(tile))
        {
          return right.eval(tile).scale(left.eval(tile)) + add.eval(tile);
        }


        /*!
         * Evaluate operands and perform element-wise multiply subtract
         */
        template<typename STORAGE, typename TILE_TYPE, typename SUBTRACT_OPERAND_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        auto multiply_subtract(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right, SUBTRACT_OPERAND_TYPE const &subtract) ->
          decltype(right.eval(tile).scale(left.eval(tile)) - subtract.eval(tile))
        {
          return right.eval(tile).scale(left.eval(tile)) - subtract.eval(tile);
        }
    };


    /*!
     * Specialization that provides multiplying a tensor*scalar
     */
    template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
    struct MultiplyOperator<LEFT_OPERAND_TYPE, RIGHT_OPERAND_TYPE,
    typename std::enable_if<RIGHT_OPERAND_TYPE::s_num_dims == 0>::type>
    {

        using result_type = typename LEFT_OPERAND_TYPE::result_type;
        using tile_type = typename LEFT_OPERAND_TYPE::tile_type;
        static constexpr camp::idx_t s_num_dims = LEFT_OPERAND_TYPE::s_num_dims;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void print_ast() {
          printf("Scale");
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        int getDimSize(int dim, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &) {
          return left.getDimSize(dim);
        }

        /*!
         * Evaluate operands and perform scaling operation
         */
        template<typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        auto multiply(TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right) ->
          decltype(left.eval(tile).scale(right.eval(tile)))
        {
          return left.eval(tile).scale(right.eval(tile));
        }



        /*!
         * Evaluate operands and perform element-wise multiply add
         */
        template<typename STORAGE, typename TILE_TYPE, typename ADD_OPERAND_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        auto multiply_add(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right, ADD_OPERAND_TYPE const &add) ->
          decltype(left.eval(tile).scale(right.eval(tile)) + add.eval(tile))
        {
          return left.eval(tile).scale(right.eval(tile)) + add.eval(tile);
        }


        /*!
         * Evaluate operands and perform element-wise multiply subtract
         */
        template<typename STORAGE, typename TILE_TYPE, typename SUBTRACT_OPERAND_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        auto multiply_subtract(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right, SUBTRACT_OPERAND_TYPE const &subtract) ->
          decltype(left.eval(tile).scale(right.eval(tile)) - subtract.eval(tile))
        {
          return left.eval(tile).scale(right.eval(tile)) - subtract.eval(tile);
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
    template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
    struct MultiplyOperator<LEFT_OPERAND_TYPE, RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 2 && RIGHT_OPERAND_TYPE::s_num_dims==1>::type>
    {

      using left_type = LEFT_OPERAND_TYPE;
      using right_type = RIGHT_OPERAND_TYPE;
      using element_type = typename LEFT_OPERAND_TYPE::element_type;
      using index_type = typename LEFT_OPERAND_TYPE::index_type;
      using tile_type = typename LEFT_OPERAND_TYPE::tile_type;
      using result_type = typename RIGHT_OPERAND_TYPE::result_type;
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
      int getDimSize(int dim, LEFT_OPERAND_TYPE const &, RIGHT_OPERAND_TYPE const &right) {
        return dim == 0 ? right.getDimSize(0) : 0;
      }

      /*!
       * Evaluate operands and perform element-wise multiply
       */
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right){
        // clear result
        result.broadcast(element_type(0));

        // multiply left and right into result
        multiply_into_result(result, tile, left, right);
      }

      template<typename STORAGE, typename TILE_TYPE, typename ADD_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_add(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right, ADD_TYPE const &add){
        // evaluate add into result
        add.eval(result, tile);

        // multiply left and right into result
        multiply_into_result(result, tile, left, right);
      }

    private:
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_into_result(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &et_left, RIGHT_OPERAND_TYPE const &et_right)
      {
        using LHS_STORAGE = typename LEFT_OPERAND_TYPE::result_type;

        // get tile size from matrix type
        index_type tile_size = left_type::result_type::s_dim_elem(1);
        index_type k_size = et_left.getDimSize(1);
        // TODO: check that left and right are compatible
        // m_left.getDimSize(1) == m_right.getDimSize(0)
        // how do we provide checking for this kind of error?

        // tile over row of left and column of right
        auto left_tile = LEFT_OPERAND_TYPE::result_type::s_get_default_tile();
        left_tile.m_begin[0] = tile.m_begin[0];
        left_tile.m_size[0] = tile.m_size[0];
        left_tile.m_size[1] = tile_size;

        TILE_TYPE right_tile = tile;
        right_tile.m_size[0] = tile_size;

        // Do full tiles in k
        index_type k = 0;
        for(;k+tile_size <= k_size; k+= tile_size){

          // evaluate both sides of operator
          left_tile.m_begin[1] = k;
          LHS_STORAGE left;
          et_left.eval(left, left_tile);

          right_tile.m_begin[0] = k;
          STORAGE right;
          et_right.eval(right, right_tile);

          // accumulate product
          left.right_multiply_vector_accumulate(result, right);
        }
        // remainder tile in k
        if(k < k_size){
          auto &left_part_tile = make_tensor_tile_partial(left_tile);
          left_part_tile.m_begin[1] = k;
          left_part_tile.m_size[1] = k_size-k;
          LHS_STORAGE left;
          et_left.eval(left, left_part_tile);

          auto &right_part_tile = make_tensor_tile_partial(right_tile);
          right_part_tile.m_begin[0] = k;
          right_part_tile.m_size[0] = k_size-k;
          STORAGE right;
          et_right.eval(right, right_part_tile);

          // accumulate product of partial tile
          left.right_multiply_vector_accumulate(result, right);
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
    template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
    struct MultiplyOperator<LEFT_OPERAND_TYPE, RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 1 && RIGHT_OPERAND_TYPE::s_num_dims==2>::type>
    {

      using left_type = LEFT_OPERAND_TYPE;
      using right_type = RIGHT_OPERAND_TYPE;
      using element_type = typename LEFT_OPERAND_TYPE::element_type;
      using index_type = typename LEFT_OPERAND_TYPE::index_type;
      using tile_type = typename RIGHT_OPERAND_TYPE::tile_type;
      using result_type = typename LEFT_OPERAND_TYPE::result_type;
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
      int getDimSize(int dim, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &) {
        return dim == 0 ? left.getDimSize(0) : 0;
      }

      /*!
       * Evaluate operands and perform element-wise multiply
       */
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right){
        // clear result
        result.broadcast(element_type(0));

        // multiply left and right into result
        multiply_into_result(result, tile, left, right);
      }

      template<typename STORAGE, typename TILE_TYPE, typename ADD_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_add(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right, ADD_TYPE const &add){
        // evaluate add into result
        add.eval(result, tile);

        // multiply left and right into result
        multiply_into_result(result, tile, left, right);
      }

    private:
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_into_result(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &et_left, RIGHT_OPERAND_TYPE const &et_right)
      {
        using RHS_STORAGE = typename RIGHT_OPERAND_TYPE::result_type;

        // get tile size from matrix type
        index_type tile_size = right_type::result_type::s_dim_elem(1);
        index_type k_size = et_right.getDimSize(1);
        // TODO: check that left and right are compatible
        // m_left.getDimSize(1) == m_right.getDimSize(0)
        // how do we provide checking for this kind of error?

        // tile over row of left and column of right
        auto right_tile = RIGHT_OPERAND_TYPE::result_type::s_get_default_tile();
        right_tile.m_begin[1] = tile.m_begin[0];
        right_tile.m_size[1] = tile.m_size[0];
        right_tile.m_size[0] = tile_size;

        TILE_TYPE left_tile = tile;
        left_tile.m_size[0] = tile_size;

        // Do full tiles in k
        index_type k = 0;
        for(;k+tile_size <= k_size; k+= tile_size){

          // evaluate both sides of operator
          right_tile.m_begin[0] = k;
          RHS_STORAGE right;
          et_right.eval(right, right_tile);

          left_tile.m_begin[0] = k;
          STORAGE left;
          et_left.eval(left, left_tile);

          // accumulate product
          right.left_multiply_vector_accumulate(result, left);

        }
        // remainder tile in k
        if(k < k_size){
          auto &right_part_tile = make_tensor_tile_partial(right_tile);
          right_part_tile.m_begin[0] = k;
          right_part_tile.m_size[0] = k_size-k;
          RHS_STORAGE right;
          et_right.eval(right, right_tile);

          auto &left_part_tile = make_tensor_tile_partial(left_tile);
          left_part_tile.m_begin[0] = k;
          left_part_tile.m_size[0] = k_size-k;
          STORAGE left;
          et_left.eval(left, left_tile);

          // compute product into x of partial tile
          right.left_multiply_vector_accumulate(result, left);
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
    template<typename LEFT_OPERAND_TYPE, typename RIGHT_OPERAND_TYPE>
    struct MultiplyOperator<LEFT_OPERAND_TYPE, RIGHT_OPERAND_TYPE,
    typename std::enable_if<LEFT_OPERAND_TYPE::s_num_dims == 2 && RIGHT_OPERAND_TYPE::s_num_dims==2>::type>
    {

      using left_type = LEFT_OPERAND_TYPE;
      using right_type = RIGHT_OPERAND_TYPE;
      using element_type = typename LEFT_OPERAND_TYPE::element_type;
//      using index_type = typename LEFT_OPERAND_TYPE::index_type;
      using tile_type = typename LEFT_OPERAND_TYPE::tile_type;
      using result_type = typename RIGHT_OPERAND_TYPE::result_type;
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
      int getDimSize(int dim, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right) {
        return dim == 0 ? left.getDimSize(0) : right.getDimSize(1);
      }

      /*!
       * Evaluate operands and perform element-wise multiply
       */
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right){
        // clear result
        result.broadcast(element_type(0));

        // multiply left and right into result
        multiply_into_result(result, tile, left, right);
      }

      template<typename STORAGE, typename TILE_TYPE, typename ADD_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_add(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &left, RIGHT_OPERAND_TYPE const &right, ADD_TYPE const &add){
        // evaluate add into result
        add.eval(result, tile);

        // multiply left and right into result
        multiply_into_result(result, tile, left, right);
      }

    private:
      template<typename STORAGE, typename TILE_TYPE>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      static
      void multiply_into_result(STORAGE &result, TILE_TYPE const &tile, LEFT_OPERAND_TYPE const &et_left, RIGHT_OPERAND_TYPE const &et_right)
      {

        // get tile size from matrix type
        auto tile_size = result_type::s_dim_elem(1);
        auto k_size = et_left.getDimSize(1);

        // TODO: check that left and right are compatible
        // m_left.getDimSize(1) == m_right.getDimSize(0)
        // how do we provide checking for this kind of error?

        // tile over row of left and column of right
        TILE_TYPE left_tile = tile;
        left_tile.m_size[1] = tile_size;

        TILE_TYPE right_tile = tile;
        right_tile.m_size[0] = tile_size;



        // Do full tiles in k
        decltype(k_size) k = 0;
        for(;k+tile_size <= k_size; k+= tile_size){

          // evaluate both sides of operator
          left_tile.m_begin[1] = k;
          STORAGE left;
          et_left.eval(left, left_tile);

          right_tile.m_begin[0] = k;
          STORAGE right;
          et_right.eval(right, right_tile);

          // accumulate product
          left.matrix_multiply_accumulate(result, right);
        }
        // remainder tile in k
        if(k < k_size){
          auto &left_part_tile = make_tensor_tile_partial(left_tile);
          left_part_tile.m_begin[1] = k;
          left_part_tile.m_size[1] = k_size-k;
          STORAGE left;
          et_left.eval(left, left_part_tile);

          auto &right_part_tile = make_tensor_tile_partial(right_tile);
          right_part_tile.m_begin[0] = k;
          right_part_tile.m_size[0] = k_size-k;
          STORAGE right;
          et_right.eval(right, right_part_tile);

          // accumulate product
          left.matrix_multiply_accumulate(result, right);
        }
      }

    };

#endif


  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
