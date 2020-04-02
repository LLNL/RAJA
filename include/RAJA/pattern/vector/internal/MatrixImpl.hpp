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
#include "RAJA/pattern/vector/internal/MatrixRef.hpp"
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

  template<typename VECTOR_TYPE, camp::idx_t ... IDX_REG, camp::idx_t ... IDX_ROW, camp::idx_t ... IDX_COL>
  struct MatrixReshapeHelper<
    MatrixImpl<VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> > >
  {

    using orig_vector_t = VECTOR_TYPE;
    using flip_vector_t = changeVectorLength<VECTOR_TYPE, sizeof...(IDX_ROW)>;

    using row_major_t = MatrixImpl<orig_vector_t,
                                  MATRIX_ROW_MAJOR,
                                  camp::idx_seq<IDX_REG...>,
                                  camp::idx_seq<IDX_ROW...>,
                                  camp::idx_seq<IDX_COL...> >;

    using col_major_t = MatrixImpl<flip_vector_t,
                                   MATRIX_COL_MAJOR,
                                   camp::make_idx_seq_t<flip_vector_t::num_registers()>,
                                   camp::idx_seq<IDX_ROW...>,
                                   camp::idx_seq<IDX_COL...> >;

    using row_major_transpose_t = MatrixImpl<flip_vector_t,
                                             MATRIX_ROW_MAJOR,
                                             camp::make_idx_seq_t<flip_vector_t::num_registers()>,
                                             camp::idx_seq<IDX_COL...>,
                                             camp::idx_seq<IDX_ROW...> >;

    using col_major_transpose_t = MatrixImpl<orig_vector_t,
                                             MATRIX_COL_MAJOR,
                                             camp::idx_seq<IDX_REG...>,
                                             camp::idx_seq<IDX_COL...>,
                                             camp::idx_seq<IDX_ROW...> >;

    // Original matrix
    using orig_matrix_t = row_major_t;

    // Original matrix with row/col major flipped
    using flip_matrix_t = col_major_t;

    // transpose of original matrix keeping original row/col major
    using similar_transpose_t = row_major_transpose_t;

    // transpose of original matrix flipping original row/col major
    using flip_transpose_t = col_major_transpose_t;
  };


  template<typename VECTOR_TYPE, camp::idx_t ... IDX_REG, camp::idx_t ... IDX_ROW, camp::idx_t ... IDX_COL>
  struct MatrixReshapeHelper<
    MatrixImpl<VECTOR_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> > >
  {
      using orig_vector_t = VECTOR_TYPE;
      using flip_vector_t = changeVectorLength<VECTOR_TYPE, sizeof...(IDX_COL)>;


      using row_major_t = MatrixImpl<flip_vector_t,
                                    MATRIX_ROW_MAJOR,
                                    camp::make_idx_seq_t<flip_vector_t::num_registers()>,
                                    camp::idx_seq<IDX_ROW...>,
                                    camp::idx_seq<IDX_COL...> >;

      using col_major_t = MatrixImpl<orig_vector_t,
                                     MATRIX_COL_MAJOR,
                                     camp::idx_seq<IDX_REG...>,
                                     camp::idx_seq<IDX_ROW...>,
                                     camp::idx_seq<IDX_COL...> >;

      using row_major_transpose_t = MatrixImpl<orig_vector_t,
                                               MATRIX_ROW_MAJOR,
                                               camp::idx_seq<IDX_REG...>,
                                               camp::idx_seq<IDX_COL...>,
                                               camp::idx_seq<IDX_ROW...> >;

      using col_major_transpose_t = MatrixImpl<flip_vector_t,
                                               MATRIX_COL_MAJOR,
                                               camp::make_idx_seq_t<flip_vector_t::num_registers()>,
                                               camp::idx_seq<IDX_COL...>,
                                               camp::idx_seq<IDX_ROW...> >;

      // Original matrix
      using orig_matrix_t = col_major_t;

      // Original matrix with row/col major flipped
      using flip_matrix_t = row_major_t;

      // transpose of original matrix keeping original row/col major
      using similar_transpose_t = col_major_transpose_t;

      // transpose of original matrix flipping original row/col major
      using flip_transpose_t = row_major_transpose_t;
  };



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



  template<typename A_VECTOR_TYPE, camp::idx_t ... A_IDX_REG, camp::idx_t ... A_IDX_ROW, camp::idx_t ... A_IDX_COL,
           typename B_VECTOR_TYPE, camp::idx_t ... B_IDX_REG, camp::idx_t ... B_IDX_ROW, camp::idx_t ... B_IDX_COL>
  struct MatrixMatrixProductHelper<
    MatrixImpl<A_VECTOR_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<A_IDX_REG...>, camp::idx_seq<A_IDX_ROW...>, camp::idx_seq<A_IDX_COL...> >,
    MatrixImpl<B_VECTOR_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<B_IDX_REG...>, camp::idx_seq<B_IDX_ROW...>, camp::idx_seq<B_IDX_COL...> >>
  {
      using A_type = MatrixImpl<A_VECTOR_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<A_IDX_REG...>, camp::idx_seq<A_IDX_ROW...>, camp::idx_seq<A_IDX_COL...> >;
      using B_type = MatrixImpl<B_VECTOR_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<B_IDX_REG...>, camp::idx_seq<B_IDX_ROW...>, camp::idx_seq<B_IDX_COL...> >;

      using result_vector = changeVectorLength<B_VECTOR_TYPE, sizeof...(A_IDX_ROW)>;
      using result_register_seq = camp::make_idx_seq_t<result_vector::num_registers()>;

      using result_type = MatrixImpl<result_vector, MATRIX_COL_MAJOR, result_register_seq, camp::idx_seq<A_IDX_ROW...>, camp::idx_seq<B_IDX_COL...> >;

      static_assert(sizeof...(A_IDX_COL) == sizeof...(B_IDX_ROW),
          "Matrices are incompatible for multiplication");

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_vector calc_row_product(result_vector sum, B_VECTOR_TYPE const &b_col, A_type const &A){

        // Note: A_IDX_COL == B_IDX_ROW

        // Keep one partial sum for each FMA unit so we can get better ILP
        // TODO Generalize to N number of FMA units
        result_vector psum[2] = {sum, result_vector()};

        camp::sink(
                (psum[A_IDX_COL%2] =
                    A.m_cols[A_IDX_COL].fused_multiply_add(
                        A_VECTOR_TYPE(b_col[B_IDX_ROW]),
                        psum[A_IDX_COL%2]))...
                );

        // Final sum of partials
        return RAJA::foldl_sum<result_vector>(psum[0], psum[1]);
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_type multiply(A_type const &A, B_type const &B){
        return result_type(calc_row_product(result_vector(), B.m_cols[B_IDX_COL], A)...);
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_type multiply_accumulate(A_type const &A, B_type const &B, result_type const &C){
        return result_type(calc_row_product(C.m_cols[B_IDX_COL], B.m_cols[B_IDX_COL], A)...);
      }

  };



} // namespace internal
} // namespace RAJA


#include "RAJA/pattern/vector/internal/MatrixBase.hpp"

namespace RAJA
{
namespace internal {

  template<camp::idx_t DEFAULT, bool IS_STATIC>
  class SemiStaticValue;

  template<camp::idx_t DEFAULT>
  class SemiStaticValue<DEFAULT, false>
  {
    private:
      camp::idx_t m_value;

    public:
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      SemiStaticValue() : m_value(DEFAULT) {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr camp::idx_t get() const{
        return m_value;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      void set(camp::idx_t new_value){
        m_value = new_value;
      }

      static constexpr bool s_is_fixed = false;
  };

  template<camp::idx_t DEFAULT>
  class SemiStaticValue<DEFAULT, true>
  {
    public:
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr camp::idx_t get() const{
        return DEFAULT;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      void set(camp::idx_t ){
        // NOP
      }

      static constexpr bool s_is_fixed = true;
  };





  /*
   * Row-Major implementation of MatrixImpl
   */
  template<typename VECTOR_TYPE, camp::idx_t ... IDX_REG, camp::idx_t ... IDX_ROW, camp::idx_t ... IDX_COL>
  class MatrixImpl<VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> > :
   public MatrixBase<MatrixImpl<VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> >>
  {
    public:
      using self_type = MatrixImpl<VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> >;
      using base_type = MatrixBase<MatrixImpl<VECTOR_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> >>;

      using vector_type = VECTOR_TYPE;
      using row_vector_type = changeVectorLength<VECTOR_TYPE, sizeof...(IDX_COL)>;
      using col_vector_type = changeVectorLength<VECTOR_TYPE, sizeof...(IDX_ROW)>;
      using element_type = typename VECTOR_TYPE::element_type;

      static_assert(vector_type::num_elem() == row_vector_type::num_elem(),
          "Internal mismatch in vector types");



    private:
      template<typename A, typename B>
      friend struct MatrixMatrixProductHelper;

      vector_type m_rows[sizeof...(IDX_ROW)];
      SemiStaticValue<sizeof...(IDX_ROW), VECTOR_TYPE::s_is_fixed> m_num_rows;

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
        m_rows{c.m_rows[IDX_ROW]...},
        m_num_rows{c.m_num_rows}
      {}

      template<typename ... ROWS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(ROWS const &... rows) :
        m_rows{rows...}
      {
        static_assert(sizeof...(ROWS) == base_type::s_num_rows,
            "Incompatible number of row vectors");
      }


      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &v){
        camp::sink((m_rows[IDX_ROW] = v.m_rows[IDX_ROW])...);
        m_num_rows.set(v.m_num_rows.get());
        return *this;
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
        camp::sink(
            // only load num_rows rows
            (IDX_ROW < num_rows
            ?  m_rows[IDX_ROW].load(ptr+IDX_ROW*row_stride, col_stride, num_cols) // LOAD
            :  m_rows[IDX_ROW])... // NOP, but has same as above type
        );
        m_num_rows.set(num_rows);

        return *this;
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
        camp::sink(
            // only store rows that are active
            (IDX_ROW < m_num_rows.get()
            ?  m_rows[IDX_ROW].store(ptr+IDX_ROW*row_stride, col_stride) // Store
            :  m_rows[IDX_ROW])... // NOP, but has same as above type
        );

        return *this;
      }

      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type v){
        camp::sink((m_rows[IDX_ROW].broadcast(v))...);
        return *this;
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
        return MatrixMatrixProductHelper<self_type,RMAT>::multiply(*this, mat);
      }

      /*!
       * Matrix-Matrix multiply accumulate
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, RMAT>::result_type
      multiply_accumulate(RMAT const &B, typename MatrixMatrixProductHelper<self_type, RMAT>::result_type const &C) const {
        return MatrixMatrixProductHelper<self_type,RMAT>::multiply_accumulate(*this, B, C);
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

  }; // MatrixImpl - ROW MAJOR




  /*
   * Column-Major implementation of MatrixImpl
   */
  template<typename VECTOR_TYPE, camp::idx_t ... IDX_REG, camp::idx_t ... IDX_ROW, camp::idx_t ... IDX_COL>
  class MatrixImpl<VECTOR_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> > :
   public MatrixBase<MatrixImpl<VECTOR_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> >>
  {
    public:
      using self_type = MatrixImpl<VECTOR_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> >;
      using base_type = MatrixBase<MatrixImpl<VECTOR_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<IDX_REG...>, camp::idx_seq<IDX_ROW...>, camp::idx_seq<IDX_COL...> >>;
      using vector_type = VECTOR_TYPE;

      using row_vector_type = changeVectorLength<VECTOR_TYPE, sizeof...(IDX_COL)>;
      using col_vector_type = changeVectorLength<VECTOR_TYPE, sizeof...(IDX_ROW)>;

      static_assert(vector_type::num_elem() == col_vector_type::num_elem(),
          "Internal mismatch in vector types");

      using element_type = typename VECTOR_TYPE::element_type;


    private:
      template<typename A, typename B>
      friend struct MatrixMatrixProductHelper;

      vector_type m_cols[sizeof...(IDX_COL)];
      SemiStaticValue<sizeof...(IDX_COL), VECTOR_TYPE::s_is_fixed> m_num_cols;

    public:

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(){}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(element_type c) :
        m_cols{(IDX_COL >= 0) ? vector_type(c) : vector_type(c)...}
      {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(self_type const &c) :
        m_cols{c.m_cols[IDX_COL]...},
        m_num_cols{c.m_num_cols}
      {}

      template<typename ... COLS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(COLS const &... cols) :
      m_cols{cols...}
      {
        static_assert(sizeof...(COLS) == base_type::s_num_cols,
            "Incompatible number of column vectors");
      }


      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &v){
        camp::sink((m_cols[IDX_COL] = v.m_cols[IDX_COL])...);
        m_num_cols.set(v.m_num_cols.get());
        return *this;
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
        camp::sink(
            // only load num_rows rows
            (IDX_COL < num_cols
            ?  m_cols[IDX_COL].load(ptr+IDX_COL*col_stride, row_stride, num_rows) // LOAD
            :  m_cols[IDX_COL])... // NOP, but has same as above type
        );
        m_num_cols.set(num_cols);

        return *this;
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
        camp::sink(
            // only store rows that are active
            (IDX_COL < m_num_cols.get()
            ?  m_cols[IDX_COL].store(ptr+IDX_COL*col_stride, row_stride) // Store
            :  m_cols[IDX_COL])... // NOP, but has same as above type
        );

        return *this;
      }

      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type v){
        camp::sink((m_cols[IDX_COL].broadcast(v))...);
        return *this;
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      col_vector_type right_multiply_vector(row_vector_type v) const {
        return
        foldl_sum<col_vector_type>(( m_cols[IDX_COL] * v.get(IDX_COL))...);
      }


      /*!
       * Matrix-Matrix product
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, RMAT>::result_type
      multiply(RMAT const &mat) const {
        return MatrixMatrixProductHelper<self_type,RMAT>::multiply(*this, mat);
      }

      /*!
       * Matrix-Matrix multiply accumulate
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, RMAT>::result_type
      multiply_accumulate(RMAT const &B, typename MatrixMatrixProductHelper<self_type, RMAT>::result_type const &C) const {
        return MatrixMatrixProductHelper<self_type,RMAT>::multiply_accumulate(*this, B, C);
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

      template<typename IDX_I, typename IDX_J>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(IDX_I row, IDX_J col, element_type val){
        m_cols[col].set(row, val);
        return *this;
      }

      template<typename IDX_I, typename IDX_J>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(IDX_I row, IDX_J col){
        return m_cols[col].get(row);
      }

      template<typename IDX_I>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type &col(IDX_I c){
        return m_cols[c];
      }

      template<typename IDX_I>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type const &col(IDX_I c) const{
        return m_cols[c];
      }

  }; // MatrixImpl - COLUMN MAJOR



}  // namespace internal



}  // namespace RAJA




#endif
