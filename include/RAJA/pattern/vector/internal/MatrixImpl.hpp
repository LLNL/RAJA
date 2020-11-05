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

#ifndef RAJA_pattern_vector_matriximpl_HPP
#define RAJA_pattern_vector_matriximpl_HPP

#include "RAJA/config.hpp"

#include "RAJA/pattern/vector/Vector.hpp"

#include "camp/camp.hpp"
namespace RAJA
{




namespace internal {


  template<typename MATRIX_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, MatrixLayout LAYOUT, typename IDX_ROW, typename IDX_COL, MatrixSizeType SIZE_TYPE>
  class MatrixImpl;



  /*
   * Matrix Shape transformation helper
   *
   * Given a starting matrix, this provides 4 aliases for compatible matrices
   *
   *    Row Major
   *    Column Major
   *    Row Major transposed
   *    Column Major transposed
   */

  template<typename MATRIX>
  struct MatrixReshapeHelper;

  template<typename T, camp::idx_t ROWS, camp::idx_t COLS, typename REGISTER_POLICY, MatrixSizeType SIZE_TYPE>
  struct MatrixReshapeHelper<
    Matrix<T, ROWS, COLS, MATRIX_ROW_MAJOR, REGISTER_POLICY, SIZE_TYPE> >
  {

    using row_major_t = Matrix<T, ROWS, COLS, MATRIX_ROW_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

    using col_major_t = Matrix<T, ROWS, COLS, MATRIX_COL_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

    using row_major_transpose_t = Matrix<T, COLS, ROWS, MATRIX_ROW_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

    using col_major_transpose_t = Matrix<T, COLS, ROWS, MATRIX_COL_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

    // Original matrix
    using orig_matrix_t = row_major_t;

    // Original matrix with row/col major flipped
    using flip_matrix_t = col_major_t;

    // transpose of original matrix keeping original row/col major
    using similar_transpose_t = row_major_transpose_t;

    // transpose of original matrix flipping original row/col major
    using flip_transpose_t = col_major_transpose_t;
  };


  template<typename T, camp::idx_t ROWS, camp::idx_t COLS, typename REGISTER_POLICY, MatrixSizeType SIZE_TYPE>
  struct MatrixReshapeHelper<
    Matrix<T, ROWS, COLS, MATRIX_COL_MAJOR, REGISTER_POLICY, SIZE_TYPE> >
  {

      using row_major_t = Matrix<T, ROWS, COLS, MATRIX_ROW_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

      using col_major_t = Matrix<T, ROWS, COLS, MATRIX_COL_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

      using row_major_transpose_t = Matrix<T, COLS, ROWS, MATRIX_ROW_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

      using col_major_transpose_t = Matrix<T, COLS, ROWS, MATRIX_COL_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

      // Original matrix
      using orig_matrix_t = col_major_t;

      // Original matrix with row/col major flipped
      using flip_matrix_t = row_major_t;

      // transpose of original matrix keeping original row/col major
      using similar_transpose_t = col_major_transpose_t;

      // transpose of original matrix flipping original row/col major
      using flip_transpose_t = row_major_transpose_t;
  };



  template<typename MATA, typename MATB, typename A_IDX_ROW, typename A_IDX_COL, typename B_IDX_ROW, typename B_IDX_COL>
  struct MatrixMatrixProductHelperExpanded;



  template<typename ELEMENT_TYPE, camp::idx_t A_ROWS, camp::idx_t A_COLS, typename REGISTER_POLICY, MatrixSizeType SIZE_TYPE,
  camp::idx_t B_ROWS, camp::idx_t B_COLS,
  camp::idx_t ... A_IDX_ROW, camp::idx_t ... A_IDX_COL, camp::idx_t ... B_IDX_ROW, camp::idx_t ... B_IDX_COL>
  struct MatrixMatrixProductHelperExpanded<
    Matrix<ELEMENT_TYPE, A_ROWS, A_COLS, MATRIX_ROW_MAJOR, REGISTER_POLICY, SIZE_TYPE>,
    Matrix<ELEMENT_TYPE, B_ROWS, B_COLS, MATRIX_ROW_MAJOR, REGISTER_POLICY, SIZE_TYPE>,
    camp::idx_seq<A_IDX_ROW...>, camp::idx_seq<A_IDX_COL...>, camp::idx_seq<B_IDX_ROW...>, camp::idx_seq<B_IDX_COL...>>
  {
      using A_type = Matrix<ELEMENT_TYPE, A_ROWS, A_COLS, MATRIX_ROW_MAJOR, REGISTER_POLICY, SIZE_TYPE>;
      using B_type = Matrix<ELEMENT_TYPE, B_ROWS, B_COLS, MATRIX_ROW_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

      using A_VECTOR_TYPE = typename A_type::vector_type;
      using B_VECTOR_TYPE = typename B_type::vector_type;

      using result_type = Matrix<ELEMENT_TYPE, A_ROWS, B_COLS, MATRIX_ROW_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

      static constexpr bool s_is_fixed = SIZE_TYPE == MATRIX_FIXED;

      static_assert(A_COLS == B_ROWS, "Matrices are incompatible for multiplication");

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      B_VECTOR_TYPE calc_row_product(B_VECTOR_TYPE sum, A_VECTOR_TYPE const &a_row, B_type const &B){

        // Note: A_IDX_COL == B_IDX_ROW

        camp::sink(
                (sum =
                    B.row(B_IDX_ROW).fused_multiply_add(
                        B_VECTOR_TYPE(a_row[A_IDX_COL]),
                        sum))...
                );

        return sum;

      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_type multiply(A_type const &A, B_type const &B){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_matrix_mm_mult_row_row ++;
#endif
        result_type res(calc_row_product(B_VECTOR_TYPE(0), A.row(A_IDX_ROW), B)...);
        res.resize(A.dim_elem(0), B.dim_elem(1));
        return res;
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_type multiply_accumulate(A_type const &A, B_type const &B, result_type const &C){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_matrix_mm_multacc_row_row ++;
#endif
        result_type res(calc_row_product(C.row(A_IDX_ROW), A.row(A_IDX_ROW), B)...);
        res.resize(A.dim_elem(0), B.dim_elem(1));
        return res;
      }

  };



  template<typename ELEMENT_TYPE, camp::idx_t A_ROWS, camp::idx_t A_COLS, typename REGISTER_POLICY, MatrixSizeType SIZE_TYPE,
  camp::idx_t B_ROWS, camp::idx_t B_COLS,
  camp::idx_t ... A_IDX_ROW, camp::idx_t ... A_IDX_COL, camp::idx_t ... B_IDX_ROW, camp::idx_t ... B_IDX_COL>
  struct MatrixMatrixProductHelperExpanded<
    Matrix<ELEMENT_TYPE, A_ROWS, A_COLS, MATRIX_COL_MAJOR, REGISTER_POLICY, SIZE_TYPE>,
    Matrix<ELEMENT_TYPE, B_ROWS, B_COLS, MATRIX_COL_MAJOR, REGISTER_POLICY, SIZE_TYPE>,
    camp::idx_seq<A_IDX_ROW...>, camp::idx_seq<A_IDX_COL...>, camp::idx_seq<B_IDX_ROW...>, camp::idx_seq<B_IDX_COL...>>
  {
      using A_type = Matrix<ELEMENT_TYPE, A_ROWS, A_COLS, MATRIX_COL_MAJOR, REGISTER_POLICY, SIZE_TYPE>;
      using B_type = Matrix<ELEMENT_TYPE, B_ROWS, B_COLS, MATRIX_COL_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

      using A_VECTOR_TYPE = typename A_type::vector_type;
      using B_VECTOR_TYPE = typename B_type::vector_type;

      static constexpr bool s_is_fixed = SIZE_TYPE == MATRIX_FIXED;

      using result_vector = changeVectorLength<B_VECTOR_TYPE, sizeof...(A_IDX_ROW)>;

      using result_type = Matrix<ELEMENT_TYPE, A_ROWS, B_COLS, MATRIX_COL_MAJOR, REGISTER_POLICY, SIZE_TYPE>;

      static_assert(A_COLS == B_ROWS, "Matrices are incompatible for multiplication");

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_vector calc_row_product(result_vector sum, B_VECTOR_TYPE const &b_col, A_type const &A){

        // Note: A_IDX_COL == B_IDX_ROW
        camp::sink(
                (sum =
                    A.col(A_IDX_COL).fused_multiply_add(
                        A_VECTOR_TYPE(b_col[B_IDX_ROW]),
                        sum))...
                );

        return sum;

      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_type multiply(A_type const &A, B_type const &B){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_matrix_mm_mult_col_col ++;
#endif
        result_type res(calc_row_product(result_vector(), B.col(B_IDX_COL), A)...);
        res.resize(A.dim_elem(0), B.dim_elem(1));
        return res;
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_type multiply_accumulate(A_type const &A, B_type const &B, result_type const &C){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_matrix_mm_multacc_col_col ++;
#endif
        result_type res(calc_row_product(C.col(B_IDX_COL), B.col(B_IDX_COL), A)...);
        res.resize(A.dim_elem(0), B.dim_elem(1));
        return res;
      }

  };




  template<typename MATA, typename MATB>
  struct MatrixMatrixProductHelper;

  template<typename ELEMENT_TYPE, camp::idx_t A_ROWS, camp::idx_t A_COLS, MatrixLayout LAYOUT, typename REGISTER_POLICY, MatrixSizeType SIZE_TYPE,
    camp::idx_t B_ROWS, camp::idx_t B_COLS>
  struct MatrixMatrixProductHelper<
    Matrix<ELEMENT_TYPE, A_ROWS, A_COLS, LAYOUT, REGISTER_POLICY, SIZE_TYPE>,
    Matrix<ELEMENT_TYPE, B_ROWS, B_COLS, LAYOUT, REGISTER_POLICY, SIZE_TYPE>> :
  public
      MatrixMatrixProductHelperExpanded<Matrix<ELEMENT_TYPE, A_ROWS, A_COLS, LAYOUT, REGISTER_POLICY, SIZE_TYPE>,
                                        Matrix<ELEMENT_TYPE, B_ROWS, B_COLS, LAYOUT, REGISTER_POLICY, SIZE_TYPE>,
                                        camp::make_idx_seq_t<A_ROWS>,
                                        camp::make_idx_seq_t<A_COLS>,
                                        camp::make_idx_seq_t<B_ROWS>,
                                        camp::make_idx_seq_t<B_COLS>>
    {};



  /**
   * Combines two matrix types into a single type for View access.
   *
   * In this case you have a RowIndex and a ColIndex, both with matrix types.
   * This combines them in a "good" way to make a single matrix type.
   *
   * It ueses the row size from the row index, and column size form the column
   * index.  All other values are taken from the row type.
   */
  template<typename ROW_MATRIX, typename COL_MATRIX>
  using MatrixViewCombiner =
      Matrix<typename ROW_MATRIX::element_type,
             ROW_MATRIX::s_num_rows,
             COL_MATRIX::s_num_cols,
             ROW_MATRIX::s_layout,
             typename ROW_MATRIX::register_policy,
             ROW_MATRIX::s_size_type>;

} // namespace internal
} // namespace RAJA


#include "RAJA/pattern/vector/internal/MatrixBase.hpp"
#include "RAJA/pattern/vector/internal/MatrixProductRef.hpp"
#include "RAJA/pattern/vector/internal/MatrixRef.hpp"

namespace RAJA
{
namespace internal {







  /*
   * Row-Major implementation of MatrixImpl
   */
  template<typename MATRIX_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t ... IDX_ROW, camp::idx_t ... IDX_COL, MatrixSizeType SIZE_TYPE>
  class MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...>, SIZE_TYPE > :
   public MatrixBase<MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...>, SIZE_TYPE >>
  {
    public:
      using self_type = MATRIX_TYPE;
      using base_type = MatrixBase<MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...>, SIZE_TYPE >>;

      using row_vector_type = typename base_type::row_vector_type;
      using col_vector_type = typename base_type::col_vector_type;
      using vector_type = row_vector_type;

      using register_type = typename vector_type::register_type;

      using element_type = ELEMENT_TYPE;

      static constexpr bool s_is_fixed = SIZE_TYPE == MATRIX_FIXED;


    private:

      vector_type m_rows[sizeof...(IDX_ROW)];
      camp::idx_t m_num_rows;

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

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl():
      m_num_rows(sizeof...(IDX_ROW))
      {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(element_type c) :
        m_rows{(IDX_ROW >= 0) ? vector_type(c) : vector_type(c)...},
        m_num_rows(sizeof...(IDX_ROW))
      {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(self_type const &c) :
        m_rows{c.m_rows[IDX_ROW]...},
        m_num_rows(c.m_num_rows)
      {}

      template<typename ... ROWS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(ROWS const &... rows) :
        m_rows{rows...},
        m_num_rows(sizeof...(ROWS))
      {
        static_assert(sizeof...(ROWS) == base_type::s_num_rows,
            "Incompatible number of row vectors");
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl &operator=(self_type const &c){
        return copy(c);
      }


      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &v){
        camp::sink((m_rows[IDX_ROW] = v.m_rows[IDX_ROW])...);
        m_num_rows = v.m_num_rows;
        return *getThis();
      }


      /*!
       * Resizes matrix to specified size, and sets all elements to zero
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &resize(camp::idx_t num_rows, camp::idx_t num_cols){
        camp::sink(
            m_rows[IDX_ROW].resize(num_cols)...
        );

        m_num_rows = num_rows;


        return *getThis();
      }

      /*!
       * Resizes matrix to specified size, and sets all elements to zero
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &clear(){
        auto num_cols = m_rows[0].size();
        camp::sink(
            m_rows[IDX_ROW].broadcast(0)...
        );
        camp::sink(
            m_rows[IDX_ROW].resize(num_cols)...
        );

        return *getThis();
      }



      /*!
       * Gets size of matrix along specified dimension
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr camp::idx_t dim_elem(camp::idx_t dim) const{
        return (dim==0) ? m_num_rows : m_rows[0].size();
      }

      /*!
       * Loads a matrix from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load(element_type const *ptr,
                      camp::idx_t row_stride = sizeof...(IDX_COL),
                      camp::idx_t col_stride = 1,
                      camp::idx_t num_rows = sizeof...(IDX_ROW),
                      camp::idx_t num_cols = sizeof...(IDX_COL))
      {

        m_num_rows = num_rows;


        if(SIZE_TYPE == MATRIX_FIXED){ // constexpr
          if(col_stride == 1){
            // load all rows as packed data
            camp::sink(
                m_rows[IDX_ROW].load_packed(ptr+IDX_ROW*row_stride)...
            );
          }
          else{
            // load all rows width a stride
            camp::sink(
                m_rows[IDX_ROW].load_strided(ptr+IDX_ROW*row_stride, col_stride)...
            );
          }

        }
        // SIZE_TYPE == MATRIX_STREAM
        else {
          if(col_stride == 1){
            // only load num_rows rows with packed data
            camp::sink(
                (IDX_ROW < m_num_rows
                ?  m_rows[IDX_ROW].load_packed_n(ptr+IDX_ROW*row_stride, num_cols)
                :  m_rows[IDX_ROW].broadcast_n(0, num_cols))... // clear to len N
            );
          }
          else{
            // only load num_rows rows with strided data
            camp::sink(
                (IDX_ROW < m_num_rows
                ?  m_rows[IDX_ROW].load_strided_n(ptr+IDX_ROW*row_stride, col_stride, num_cols)
                :  m_rows[IDX_ROW].broadcast_n(0, num_cols))... // clear to len N
            );
          }



        }


        return *getThis();
      }


      /*!
       * Stores a matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store(element_type *ptr,
                      camp::idx_t row_stride = sizeof...(IDX_COL),
                      camp::idx_t col_stride = 1) const
      {


        if(SIZE_TYPE == MATRIX_FIXED){ // constexpr
          if(col_stride == 1){
            // store all rows as packed data
            camp::sink(
                m_rows[IDX_ROW].store_packed(ptr+IDX_ROW*row_stride)...
            );
          }
          else{
            // load all rows width a stride
            camp::sink(
                m_rows[IDX_ROW].store_strided(ptr+IDX_ROW*row_stride, col_stride)...
            );
          }

        }
        // SIZE_TYPE == MATRIX_STREAM
        else {
          auto num_cols = m_rows[0].size();
          if(col_stride == 1){
            // only store num_rows rows with packed data
            camp::sink(
                (IDX_ROW < m_num_rows
                ?  m_rows[IDX_ROW].store_packed_n(ptr+IDX_ROW*row_stride, num_cols)
                :  m_rows[IDX_ROW])... // NOP, but has same as above type
            );
          }
          else{
            // only store num_rows rows with strided data
            camp::sink(
                (IDX_ROW < m_num_rows
                ?  m_rows[IDX_ROW].store_strided_n(ptr+IDX_ROW*row_stride, col_stride, num_cols)
                :  m_rows[IDX_ROW])... // NOP, but has same as above type
            );
          }
        }

        return *getThis();
      }

      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type v){
        camp::sink((m_rows[IDX_ROW].broadcast(v))...);
        return *getThis();
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      col_vector_type right_multiply_vector(row_vector_type v) const {
        //return vector_type(v.dot(m_rows[IDX_ROW])...);
        col_vector_type result;
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
      multiply(RMAT const &mat) const {
        return MatrixMatrixProductHelper<self_type,RMAT>::multiply(*getThis(), mat);
      }

      /*!
       * Matrix-Matrix multiply accumulate
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, RMAT>::result_type
      multiply_accumulate(RMAT const &B, typename MatrixMatrixProductHelper<self_type, RMAT>::result_type const &C) const {
        return MatrixMatrixProductHelper<self_type,RMAT>::multiply_accumulate(*getThis(), B, C);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(self_type mat) const {
        return self_type(
            (m_rows[IDX_ROW])+(mat.m_rows[IDX_ROW]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type mat) const {
        return self_type(
            (m_rows[IDX_ROW])-(mat.m_rows[IDX_ROW]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(camp::idx_t row, camp::idx_t col, element_type val){
        m_rows[row].set(col, val);
        return *getThis();
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(camp::idx_t row, camp::idx_t col) const {
        return m_rows[row].get(col);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type &row(camp::idx_t row){
        return m_rows[row];
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type const &row(camp::idx_t row) const{
        return m_rows[row];
      }

  }; // MatrixImpl - ROW MAJOR




  /*
   * Column-Major implementation of MatrixImpl
   */
  template<typename MATRIX_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t ... IDX_ROW, camp::idx_t ... IDX_COL, MatrixSizeType SIZE_TYPE>
  class MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...>, SIZE_TYPE > :
   public MatrixBase<MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...>, SIZE_TYPE >>
  {
    public:
      using self_type = MATRIX_TYPE;
      using base_type = MatrixBase<MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...>, SIZE_TYPE >>;

      using row_vector_type = typename base_type::row_vector_type;
      using col_vector_type = typename base_type::col_vector_type;
      using vector_type = col_vector_type;
      using register_type = typename vector_type::register_type;

      using element_type = ELEMENT_TYPE;

      static constexpr bool s_is_fixed = SIZE_TYPE == MATRIX_FIXED;


    private:

      vector_type m_cols[sizeof...(IDX_COL)];
      camp::idx_t m_num_cols;

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

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl() :
      m_num_cols(sizeof...(IDX_COL))
      {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(element_type c) :
        m_cols{(IDX_COL >= 0) ? vector_type(c) : vector_type(c)...},
        m_num_cols(sizeof...(IDX_COL))
      {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(self_type const &c) :
        m_cols{c.m_cols[IDX_COL]...},
        m_num_cols(c.m_num_cols)
      {}

      template<typename ... COLS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(COLS const &... cols) :
      m_cols{cols...},
      m_num_cols(sizeof...(COLS))
      {
        static_assert(sizeof...(COLS) == base_type::s_num_cols,
            "Incompatible number of column vectors");
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl &operator=(self_type const &c){
        return copy(c);
      }

      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &v){
        camp::sink((m_cols[IDX_COL] = v.m_cols[IDX_COL])...);
        m_num_cols = v.m_num_cols;
        return *getThis();
      }

      /*!
       * Resizes matrix to specified size, and sets all elements to zero
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &resize(camp::idx_t num_rows, camp::idx_t num_cols){
        camp::sink(
            m_cols[IDX_COL].resize(num_rows)...
        );

        m_num_cols = num_cols;


        return *getThis();
      }

      /*!
       * Resizes matrix to specified size, and sets all elements to zero
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &clear(){
        auto num_rows = m_cols[0].size();
        camp::sink(
            m_cols[IDX_COL].broadcast(0)...
        );
        camp::sink(
            m_cols[IDX_COL].resize(num_rows)...
        );

        return *getThis();
      }

      /*!
       * Gets size of matrix along specified dimension
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr camp::idx_t dim_elem(camp::idx_t dim) const{
        return (dim==0) ? m_cols[0].size() : m_num_cols;
      }

      /*!
       * Loads a matrix from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load(element_type const *ptr,
                      camp::idx_t row_stride = 1,
                      camp::idx_t col_stride = sizeof...(IDX_ROW),
                      camp::idx_t num_rows = sizeof...(IDX_ROW),
                      camp::idx_t num_cols = sizeof...(IDX_COL))
      {
        m_num_cols = num_cols;

        if(SIZE_TYPE == MATRIX_FIXED){ // constexpr
          if(row_stride == 1){
            // load all rows as packed data
            camp::sink(
                m_cols[IDX_COL].load_packed(ptr+IDX_COL*col_stride)...
            );
          }
          else{
            // load all rows width a stride
            camp::sink(
                m_cols[IDX_COL].load_strided(ptr+IDX_COL*col_stride, row_stride)...
            );
          }

        }
        // SIZE_TYPE == MATRIX_STREAM
        else {
          if(row_stride == 1){
            // only load num_rows rows with packed data
            camp::sink(
                (IDX_COL < m_num_cols
                ?  m_cols[IDX_COL].load_packed_n(ptr+IDX_COL*col_stride, num_rows)
                :  m_cols[IDX_COL].broadcast_n(0, num_rows))... // NOP, but has same as above type
            );
          }
          else{
            // only load num_rows rows with strided data
            camp::sink(
                (IDX_COL < m_num_cols
                ?   m_cols[IDX_COL].load_strided_n(ptr+IDX_COL*col_stride, row_stride, num_rows)
                :  m_cols[IDX_COL].broadcast_n(0, num_rows))... // NOP, but has same as above type
            );
          }
        }

        return *getThis();
      }


      /*!
       * Stores a matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store(element_type *ptr,
                      camp::idx_t row_stride = 1,
                      camp::idx_t col_stride = sizeof...(IDX_ROW)) const
      {


        if(SIZE_TYPE == MATRIX_FIXED){ // constexpr
          if(row_stride == 1){
            // store all rows as packed data
            camp::sink(
                m_cols[IDX_COL].store_packed(ptr+IDX_COL*col_stride)...
            );
          }
          else{
            // store all rows width a stride
            camp::sink(
                // only store rows that are active
                (IDX_COL < m_num_cols
                ?  m_cols[IDX_COL].store_strided(ptr+IDX_COL*col_stride, row_stride)
                :  m_cols[IDX_COL])... // NOP, but has same as above type
            );
          }

        }
        // SIZE_TYPE == MATRIX_STREAM
        else {
          auto num_rows = m_cols[0].size();
          if(row_stride == 1){
            // only store num_rows rows with packed data
            camp::sink(
                (IDX_COL < m_num_cols
                ?  m_cols[IDX_COL].store_packed_n(ptr+IDX_COL*col_stride, num_rows)
                :  m_cols[IDX_COL])... // NOP, but has same as above type
            );
          }
          else{
            // only store num_rows rows with strided data
            camp::sink(
                (IDX_COL < m_num_cols
                ?   m_cols[IDX_COL].store_strided_n(ptr+IDX_COL*col_stride, row_stride, num_rows)
                :  m_cols[IDX_COL])... // NOP, but has same as above type
            );
          }
        }


        return *getThis();
      }

      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type v){
        camp::sink((m_cols[IDX_COL].broadcast(v))...);
        return *getThis();
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      col_vector_type right_multiply_vector(row_vector_type v) const {
        return
        RAJA::sum<col_vector_type>(( m_cols[IDX_COL] * v.get(IDX_COL))...);
      }


      /*!
       * Matrix-Matrix product
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, RMAT>::result_type
      multiply(RMAT const &mat) const {
        return MatrixMatrixProductHelper<self_type,RMAT>::multiply(*getThis(), mat);
      }

      /*!
       * Matrix-Matrix multiply accumulate
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, RMAT>::result_type
      multiply_accumulate(RMAT const &B, typename MatrixMatrixProductHelper<self_type, RMAT>::result_type const &C) const {
        return MatrixMatrixProductHelper<self_type,RMAT>::multiply_accumulate(*getThis(), B, C);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(self_type mat) const {
        return self_type(
            (m_cols[IDX_COL])+(mat.m_cols[IDX_COL]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type mat) const {
        return self_type(
            (m_cols[IDX_COL])-(mat.m_cols[IDX_COL]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(camp::idx_t row, camp::idx_t col, element_type val){
        m_cols[col].set(row, val);
        return *getThis();
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(camp::idx_t row, camp::idx_t col) const {
        return m_cols[col].get(row);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type &col(camp::idx_t c){
        return m_cols[c];
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type const &col(camp::idx_t c) const{
        return m_cols[c];
      }

  }; // MatrixImpl - COLUMN MAJOR



}  // namespace internal



}  // namespace RAJA




#endif
