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

#ifndef RAJA_pattern_tensor_ET_Operator_HPP
#define RAJA_pattern_tensor_ET_Operator_HPP

namespace RAJA
{

  namespace internal
  {

  namespace ET
  {


    /*!
     * Provides default operations for add, subtract and divide
     *
     * For the most part, this is just element wise operations between
     * compatible tensors.
     *
     * There are specializations that handle when one operand is a scalar
     */
    template<typename LHS_TYPE, typename RHS_TYPE, class ENABLE = void>
    struct Operator {

        using result_type = typename LHS_TYPE::result_type;
        using tile_type = typename LHS_TYPE::tile_type;
        static constexpr camp::idx_t s_num_dims = LHS_TYPE::s_num_dims;

//        static_assert(LHS_TYPE::s_num_dims == RHS_TYPE::s_num_dims,
//            "The operands are incompatible");

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
         * Evaluate operands and perform element-wise addition
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void add(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){

          // evaluate LHS in-place
          lhs.eval(result, tile);

          // evaluate RHS in temporary
          STORAGE rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // compute result
          result.inplace_add(rhs_tmp);
        }

        /*!
         * Evaluate operands and perform element-wise subtraction
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void subtract(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
          // evaluate LHS in-place
          lhs.eval(result, tile);

          // evaluate RHS in temporary
          STORAGE rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // compute result
          result.inplace_subtract(rhs_tmp);
        }

        /*!
         * Evaluate operands and perform element-wise division
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void divide(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
          // evaluate LHS in-place
          lhs.eval(result, tile);

          // evaluate RHS in temporary
          STORAGE rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // compute result
          result.inplace_divide(rhs_tmp);
        }

    };

    /*!
     * Specialization when the left operand is a scalar
     */
    template<typename LHS_TYPE, typename RHS_TYPE>
    struct Operator<LHS_TYPE, RHS_TYPE,
    typename std::enable_if<LHS_TYPE::s_num_dims == 0>::type>
    {

        using result_type = typename RHS_TYPE::result_type;
        using tile_type = typename RHS_TYPE::tile_type;
        static constexpr camp::idx_t s_num_dims = RHS_TYPE::s_num_dims;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void print_ast() {
          printf("Scalar");
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        int getDimSize(int dim, LHS_TYPE const &, RHS_TYPE const &rhs) {
          return rhs.getDimSize(dim);
        }


        /*!
         * Evaluate operands and perform element-wise addition
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void add(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){

          // evaluate LHS
          typename LHS_TYPE::result_type scalar;
          lhs.eval(scalar, tile);
          result = STORAGE(scalar);

          // evaluate RHS
          STORAGE rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // compute result
          result.inplace_add(rhs_tmp);
        }

        /*!
         * Evaluate operands and perform element-wise subtraction
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void subtract(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
          // evaluate LHS
          typename LHS_TYPE::result_type scalar;
          lhs.eval(scalar, tile);
          result = STORAGE(scalar);

          // evaluate RHS
          STORAGE rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // compute result
          result.inplace_subtract(rhs_tmp);
        }

        /*!
         * Evaluate operands and perform element-wise division
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void divide(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
          // evaluate LHS
          typename LHS_TYPE::result_type scalar;
          lhs.eval(scalar, tile);
          result = STORAGE(scalar);

          // evaluate RHS
          STORAGE rhs_tmp;
          rhs.eval(rhs_tmp, tile);

          // compute result
          result.inplace_divide(rhs_tmp);
        }
    };

    /*!
     * Specialization when the right operand is a scalar
     */
    template<typename LHS_TYPE, typename RHS_TYPE>
    struct Operator<LHS_TYPE, RHS_TYPE,
    typename std::enable_if<RHS_TYPE::s_num_dims == 0>::type>
    {

        using result_type = typename LHS_TYPE::result_type;
        using tile_type = typename LHS_TYPE::tile_type;
        static constexpr camp::idx_t s_num_dims = LHS_TYPE::s_num_dims;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void print_ast() {
          printf("Scalar");
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        int getDimSize(int dim, LHS_TYPE const &lhs, RHS_TYPE const &) {
          return lhs.getDimSize(dim);
        }


        /*!
         * Evaluate operands and perform element-wise addition
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void add(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){

          // evaluate LHS
          lhs.eval(result, tile);

          // evaluate RHS
          typename RHS_TYPE::result_type scalar;
          rhs.eval(scalar, tile);
          STORAGE rhs_tmp(scalar);

          // compute result
          result.inplace_add(rhs_tmp);
        }

        /*!
         * Evaluate operands and perform element-wise subtraction
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void subtract(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
          // evaluate LHS
          lhs.eval(result, tile);

          // evaluate RHS
          typename RHS_TYPE::result_type scalar;
          rhs.eval(scalar, tile);
          STORAGE rhs_tmp(scalar);

          // compute result
          result.inplace_subtract(rhs_tmp);

        }

        /*!
         * Evaluate operands and perform element-wise division
         */
        template<typename STORAGE, typename TILE_TYPE>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        void divide(STORAGE &result, TILE_TYPE const &tile, LHS_TYPE const &lhs, RHS_TYPE const &rhs){
          // evaluate LHS
          lhs.eval(result, tile);

          // evaluate RHS
          typename RHS_TYPE::result_type scalar;
          rhs.eval(scalar, tile);
          STORAGE rhs_tmp(scalar);

          // compute result
          result.inplace_divide(rhs_tmp);

        }

    };


  } // namespace ET

  } // namespace internal

}  // namespace RAJA


#endif
