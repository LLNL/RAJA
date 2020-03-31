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

#ifndef RAJA_pattern_vector_matrix_HPP
#define RAJA_pattern_vector_matrix_HPP

#include "RAJA/config.hpp"

#include "RAJA/pattern/vector/Vector.hpp"
#include "camp/camp.hpp"
namespace RAJA
{

  enum MatrixLayout {
    MATRIX_ROW_MAJOR,
    MATRIX_COL_MAJOR
  };


namespace internal {


  template<typename VECTOR_TYPE, MatrixLayout LAYOUT, typename IDX_REG, typename IDX_ROW, typename IDX_COL>
  class MatrixImpl;

}




namespace internal {

  template<typename MATA, typename MATB>
  struct MatrixMatrixProductHelper;



  template<typename A_VECTOR_TYPE, camp::idx_t ... A_IDX_REG, camp::idx_t ... A_IDX_ROW, camp::idx_t ... A_IDX_COL,
           typename B_VECTOR_TYPE, camp::idx_t ... B_IDX_REG, camp::idx_t ... B_IDX_ROW, camp::idx_t ... B_IDX_COL>
  struct MatrixMatrixProductHelper<
    MatrixImpl<A_VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<A_IDX_REG...>, camp::idx_seq<A_IDX_ROW...>, camp::idx_seq<A_IDX_COL...> >,
    MatrixImpl<B_VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<B_IDX_REG...>, camp::idx_seq<B_IDX_ROW...>, camp::idx_seq<B_IDX_COL...> >>
  {
      using A_type = MatrixImpl<A_VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<A_IDX_REG...>, camp::idx_seq<A_IDX_ROW...>, camp::idx_seq<A_IDX_COL...> >;
      using B_type = MatrixImpl<B_VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<B_IDX_REG...>, camp::idx_seq<B_IDX_ROW...>, camp::idx_seq<B_IDX_COL...> >;

      using result_type = MatrixImpl<B_VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<B_IDX_REG...>, camp::idx_seq<A_IDX_ROW...>, camp::idx_seq<B_IDX_COL...> >;

      static_assert(sizeof...(A_IDX_COL) == sizeof...(B_IDX_ROW),
          "Matrices are incompatible for multiplication");


      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      B_VECTOR_TYPE calc_row_product(B_VECTOR_TYPE sum, A_VECTOR_TYPE const &a_row, B_type const &B){

        // Note: A_IDX_COL == B_IDX_ROW

        // Keep one partial sum for each FMA unit so we can get better ILP
        // TODO Generalize to N number of FMA units
        B_VECTOR_TYPE psum[2] = {sum, B_VECTOR_TYPE()};

        camp::sink(
                (psum[B_IDX_ROW%2] =
                    B.m_rows[B_IDX_ROW].fused_multiply_add(
                        B_VECTOR_TYPE(a_row[A_IDX_COL]),
                        psum[B_IDX_ROW%2]))...
                );

        // Final sum of partials
        return RAJA::foldl_sum<B_VECTOR_TYPE>(psum[0], psum[1]);
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_type multiply(A_type const &A, B_type const &B){
        return result_type(calc_row_product(B_VECTOR_TYPE(), A.m_rows[A_IDX_ROW], B)...);
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_type multiply_accumulate(A_type const &A, B_type const &B, result_type const &C){
        return result_type(calc_row_product(C.m_rows[A_IDX_ROW], A.m_rows[A_IDX_ROW], B)...);
      }

  };



  /*
   * Row-Major implementation of MatrixImpl
   */

  template<typename VECTOR_TYPE, camp::idx_t ... IDX_REG, camp::idx_t ... IDX_ROW, camp::idx_t ... IDX_COL>
  class MatrixImpl<VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> >{
    public:
      using self_type = MatrixImpl<VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> >;

      using vector_type = VECTOR_TYPE;
      using row_vector_type = VECTOR_TYPE;
      using element_type = typename VECTOR_TYPE::element_type;

      static constexpr camp::idx_t s_num_rows = sizeof...(IDX_ROW);
      static constexpr camp::idx_t s_num_cols = sizeof...(IDX_COL);

    private:
      template<typename A, typename B>
      friend struct MatrixMatrixProductHelper;

      vector_type m_rows[sizeof...(IDX_ROW)];

    public:

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(){}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(element_type c) :
        m_rows{(IDX_ROW >= 0) ? vector_type(c) : vector_type(c)...}
      {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(self_type const &c) :
        m_rows{c.m_rows[IDX_ROW]...}
      {}

      template<typename ... ROWS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(ROWS const &... rows) :
        m_rows{rows...}
      {
        static_assert(sizeof...(ROWS) == s_num_rows,
            "Incompatible number of row vectors");
      }

      /*!
       * Assignment operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(self_type const &v){
        return copy(v);
      }


      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &v){
        camp::sink((m_rows[IDX_ROW] = v.m_rows[IDX_ROW])...);
        return *this;
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type operator*(vector_type v){
        //return vector_type(v.dot(m_rows[IDX_ROW])...);
        vector_type result;
        camp::sink(
            result.set(IDX_ROW, v.dot(m_rows[IDX_ROW]))...
            );
        return result;
      }


      /*!
       * Matrix-Matrix product
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, RMAT>::result_type
      operator*(RMAT const &mat){
        return MatrixMatrixProductHelper<self_type,RMAT>::multiply(*this, mat);
      }

      /*!
       * Matrix-Matrix multiply accumulate
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, RMAT>::result_type
      multiply_accumulate(RMAT const &B, typename MatrixMatrixProductHelper<self_type, RMAT>::result_type const &C){
        return MatrixMatrixProductHelper<self_type,RMAT>::multiply_accumulate(*this, B, C);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator+(self_type mat){
        return self_type(
            (m_rows[IDX_ROW])+(mat.m_rows[IDX_ROW]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator-(self_type mat){
        return self_type(
            (m_rows[IDX_ROW])-(mat.m_rows[IDX_ROW]) ...
        );
      }

      template<typename IDX_I, typename IDX_J>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(IDX_I row, IDX_J col, element_type val){
        m_rows[row].set(col, val);
        return *this;
      }

      template<typename IDX_I, typename IDX_J>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(IDX_I row, IDX_J col){
        return m_rows[row].get(col);
      }

      template<typename IDX_I>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type &row(IDX_I row){
        return m_rows[row];
      }

      template<typename IDX_I>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type const &row(IDX_I row) const{
        return m_rows[row];
      }

      template<typename IDX_I, typename IDX_J>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type operator()(IDX_I row, IDX_J col){
        return get(row, col);
      }

  };




}  // namespace internal

//    StreamVectorExt<
//    RAJA::Register<REGISTER, T, RAJA::RegisterTraits<REGISTER, T>::s_num_elem>,
//    UNROLL>;



}  // namespace RAJA




#endif
