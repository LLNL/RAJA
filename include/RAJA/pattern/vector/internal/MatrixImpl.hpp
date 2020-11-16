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


  template<typename MATRIX_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, MatrixLayout LAYOUT, typename IDX_SEQ>
  class MatrixImpl;



  template<typename MATA, typename MATB, typename IDX_SEQ>
  struct MatrixMatrixProductHelperExpanded;


  template<typename ELEMENT_TYPE, typename REGISTER_POLICY, camp::idx_t ... IDX>
  struct MatrixMatrixProductHelperExpanded<
    RegisterMatrix<ELEMENT_TYPE, MATRIX_ROW_MAJOR, REGISTER_POLICY>,
    RegisterMatrix<ELEMENT_TYPE, MATRIX_ROW_MAJOR, REGISTER_POLICY>,
    camp::idx_seq<IDX...>>
  {
      using A_type = RegisterMatrix<ELEMENT_TYPE, MATRIX_ROW_MAJOR, REGISTER_POLICY>;
      using B_type = RegisterMatrix<ELEMENT_TYPE, MATRIX_ROW_MAJOR, REGISTER_POLICY>;

      using vector_type = Register<REGISTER_POLICY, ELEMENT_TYPE>;

      using result_type = RegisterMatrix<ELEMENT_TYPE, MATRIX_ROW_MAJOR, REGISTER_POLICY>;

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      vector_type calc_row_product(vector_type sum, vector_type const &a_row, B_type const &B){

        // Note: A_IDX_COL == B_IDX_ROW

        camp::sink(
                (sum =
                    B.row(IDX).fused_multiply_add(
                        vector_type(a_row[IDX]),
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
        return result_type(calc_row_product(vector_type(0), A.row(IDX), B)...);
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_type multiply_accumulate(A_type const &A, B_type const &B, result_type const &C){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_matrix_mm_multacc_row_row ++;
#endif
        return result_type(calc_row_product(C.row(IDX), A.row(IDX), B)...);
      }

  };



  template<typename ELEMENT_TYPE, typename REGISTER_POLICY, camp::idx_t ... IDX>
  struct MatrixMatrixProductHelperExpanded<
    RegisterMatrix<ELEMENT_TYPE, MATRIX_COL_MAJOR, REGISTER_POLICY>,
    RegisterMatrix<ELEMENT_TYPE, MATRIX_COL_MAJOR, REGISTER_POLICY>,
    camp::idx_seq<IDX...>>
  {
      using A_type = RegisterMatrix<ELEMENT_TYPE, MATRIX_COL_MAJOR, REGISTER_POLICY>;
      using B_type = RegisterMatrix<ELEMENT_TYPE, MATRIX_COL_MAJOR, REGISTER_POLICY>;

      using vector_type = Register<REGISTER_POLICY, ELEMENT_TYPE>;

      using result_type = RegisterMatrix<ELEMENT_TYPE, MATRIX_COL_MAJOR, REGISTER_POLICY>;


      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      vector_type calc_row_product(vector_type sum, vector_type const &b_col, A_type const &A){

        // Note: A_IDX_COL == B_IDX_ROW
        camp::sink(
                (sum =
                    A.col(IDX).fused_multiply_add(
                        vector_type(b_col[IDX]),
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
        return result_type(calc_row_product(vector_type(0), B.col(IDX), A)...);
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      result_type multiply_accumulate(A_type const &A, B_type const &B, result_type const &C){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_matrix_mm_multacc_col_col ++;
#endif
        return result_type(calc_row_product(C.col(IDX), B.col(IDX), A)...);
      }

  };




  template<typename MATA, typename MATB>
  struct MatrixMatrixProductHelper;

  template<typename ELEMENT_TYPE, MatrixLayout LAYOUT, typename REGISTER_POLICY>
  struct MatrixMatrixProductHelper<
    RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>,
    RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>> :
  public
      MatrixMatrixProductHelperExpanded<RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>,
                                        RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>,
                                        camp::make_idx_seq_t<Register<REGISTER_POLICY, ELEMENT_TYPE>::s_num_elem>>
    {};




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
  template<typename MATRIX_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t ... IDX>
  class MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX...>> :
   public MatrixBase<MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX...>>>
  {
    public:
      using self_type = MATRIX_TYPE;
      using base_type = MatrixBase<MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_ROW_MAJOR, camp::idx_seq<IDX...> >>;

      using vector_type = typename base_type::vector_type;
      using register_type = typename vector_type::register_type;
      using element_type = ELEMENT_TYPE;


    private:

      vector_type m_rows[sizeof...(IDX)];

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
      MatrixImpl(): base_type()
      {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(element_type c) :
        base_type(),
        m_rows{(IDX >= 0) ? vector_type(c) : vector_type(c)...}
      {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(self_type const &c) :
        base_type(c),
        m_rows{c.m_rows[IDX]...}
      {}

      template<typename ... ROWS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(ROWS const &... rows) :
        base_type(),
        m_rows{rows...}
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
        camp::sink((m_rows[IDX] = v.m_rows[IDX])...);
        base_type::copy(v);
        return *getThis();
      }




      /*!
       * Resizes matrix to specified size, and sets all elements to zero
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &clear(){
        camp::sink(
            m_rows[IDX].broadcast(0)...
        );

        return *getThis();
      }

      /*!
       * Loads a dense full matrix from memory.
       *
       * Column entries must be stride-1, rows may be any striding
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_packed(element_type const *ptr,
          int row_stride, int)
      {
        // load all rows as packed data
        camp::sink(
            m_rows[IDX].load_packed(ptr+IDX*row_stride)...
        );

        return *getThis();
      }

      /*!
       * Loads a strided full matrix from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_strided(element_type const *ptr,
          int row_stride, int col_stride)
      {
        // load all rows width a stride
        camp::sink(
            m_rows[IDX].load_strided(ptr+IDX*row_stride, col_stride)...
        );

        return *getThis();
      }

      /*!
       * Loads a dense partial matrix from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_packed_nm(element_type const *ptr,
          int row_stride, int,
          int num_rows, int num_cols)
      {
        // only load num_rows rows with packed data
        camp::sink(
            (IDX < num_rows
            ?  m_rows[IDX].load_packed_n(ptr+IDX*row_stride, num_cols)
            :  m_rows[IDX].broadcast(0))... // clear to len N
        );

        return *getThis();
      }

      /*!
       * Loads a strided partial matrix from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_strided_nm(element_type const *ptr,
          int row_stride, int col_stride,
          int num_rows, int num_cols)
      {
        // only load num_rows rows with strided data
        camp::sink(
            (IDX < num_rows
            ?  m_rows[IDX].load_strided_n(ptr+IDX*row_stride, col_stride, num_cols)
            :  m_rows[IDX].broadcast(0))... // clear to len N
        );

        return *getThis();
      }



      /*!
       * Store a dense full matrix to memory.
       *
       * Column entries must be stride-1, rows may be any striding
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &store_packed(element_type *ptr,
          int row_stride, int)
      {
        // store all rows as packed data
        camp::sink(
            m_rows[IDX].store_packed(ptr+IDX*row_stride)...
        );

        return *getThis();
      }

      /*!
       * Store a strided full matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &store_strided(element_type *ptr,
          int row_stride, int col_stride)
      {
        // store all rows width a column stride
        camp::sink(
            m_rows[IDX].store_strided(ptr+IDX*row_stride, col_stride)...
        );

        return *getThis();
      }

      /*!
       * Store a dense partial matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &store_packed_nm(element_type *ptr,
          int row_stride, int ,
          int num_rows, int num_cols)
      {
        // only store num_rows rows with packed data
        camp::sink(
            (IDX < num_rows
            ?  m_rows[IDX].store_packed_n(ptr+IDX*row_stride, num_cols)
            :  m_rows[IDX])... // NOP, but has same as above type
        );

        return *getThis();
      }

      /*!
       * Store a strided partial matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &store_strided_nm(element_type *ptr,
          int row_stride, int col_stride,
          int num_rows, int num_cols)
      {
        // only store num_rows rows with strided data
        camp::sink(
            (IDX < num_rows
            ?  m_rows[IDX].store_strided_n(ptr+IDX*row_stride, col_stride, num_cols)
            :  m_rows[IDX])... // NOP, but has same as above type
        );

        return *getThis();
      }




      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type v){
        camp::sink((m_rows[IDX].broadcast(v))...);
        return *getThis();
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type right_multiply_vector(vector_type v) const {
        vector_type result;
        camp::sink(
            result.set(IDX, v.dot(m_rows[IDX]))...
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
            (m_rows[IDX])+(mat.m_rows[IDX]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type mat) const {
        return self_type(
            (m_rows[IDX])-(mat.m_rows[IDX]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(int row, int col, element_type val){
        m_rows[row].set(col, val);
        return *getThis();
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(int row, int col) const {
        return m_rows[row].get(col);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type &row(int row){
        return m_rows[row];
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type const &row(int row) const{
        return m_rows[row];
      }

  }; // MatrixImpl - ROW MAJOR




  /*
   * Column-Major implementation of MatrixImpl
   */
  template<typename MATRIX_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t ... IDX>
  class MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<IDX...>> :
   public MatrixBase<MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<IDX...> >>
  {
    public:
      using self_type = MATRIX_TYPE;
      using base_type = MatrixBase<MatrixImpl<MATRIX_TYPE, REGISTER_POLICY, ELEMENT_TYPE, MATRIX_COL_MAJOR, camp::idx_seq<IDX...> >>;

      using vector_type = typename base_type::vector_type;
      using register_type = typename vector_type::register_type;
      using element_type = ELEMENT_TYPE;


    private:

      vector_type m_cols[sizeof...(IDX)];

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
        base_type()
      {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(element_type c) :
        base_type(),
        m_cols{(IDX >= 0) ? vector_type(c) : vector_type(c)...}
      {}

      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(self_type const &c) :
        base_type(c),
        m_cols{c.m_cols[IDX]...}
      {}

      template<typename ... COLS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      MatrixImpl(COLS const &... cols) :
      base_type(),
      m_cols{cols...}
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
        camp::sink((m_cols[IDX] = v.m_cols[IDX])...);
        base_type::copy(v);
        return *getThis();
      }


      /*!
       * Resizes matrix to specified size, and sets all elements to zero
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &clear(){

        camp::sink(
            m_cols[IDX].broadcast(0)...
        );

        return *getThis();
      }




      /*!
       * Loads a dense full matrix from memory.
       *
       * Column entries must be stride-1, rows may be any striding
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_packed(element_type const *ptr,
          int, int col_stride)
      {
        // load all rows as packed data
        camp::sink(
            m_cols[IDX].load_packed(ptr+IDX*col_stride)...
        );

        return *getThis();
      }

      /*!
       * Loads a strided full matrix from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_strided(element_type const *ptr,
          int row_stride, int col_stride)
      {
        // load all rows width a stride
        camp::sink(
            m_cols[IDX].load_strided(ptr+IDX*col_stride, row_stride)...
        );

        return *getThis();
      }

      /*!
       * Loads a dense partial matrix from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_packed_nm(element_type const *ptr,
          int, int col_stride,
          int num_rows, int num_cols)
      {
        // only load num_rows rows with packed data
        camp::sink(
            (IDX < num_cols
            ?  m_cols[IDX].load_packed_n(ptr+IDX*col_stride, num_rows)
            :  m_cols[IDX].broadcast(0))... // NOP, but has same as above type
        );

        return *getThis();
      }

      /*!
       * Loads a strided partial matrix from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_strided_nm(element_type const *ptr,
          int row_stride, int col_stride,
          int num_rows, int num_cols)
      {
        // only load num_rows rows with strided data
        camp::sink(
            (IDX < num_cols
            ?   m_cols[IDX].load_strided_n(ptr+IDX*col_stride, row_stride, num_rows)
            :  m_cols[IDX].broadcast(0))... // NOP, but has same as above type
        );

        return *getThis();
      }



      /*!
       * Store a dense full matrix to memory.
       *
       * Column entries must be stride-1, rows may be any striding
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &store_packed(element_type *ptr,
          int, int col_stride)
      {
        // store all rows as packed data
        camp::sink(
            m_cols[IDX].store_packed(ptr+IDX*col_stride)...
        );

        return *getThis();
      }

      /*!
       * Store a strided full matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &store_strided(element_type *ptr,
          int row_stride, int col_stride)
      {
        // store all rows width a column stride
        camp::sink(
            m_cols[IDX].store_strided(ptr+IDX*col_stride, row_stride)...
        );

        return *getThis();
      }

      /*!
       * Store a dense partial matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &store_packed_nm(element_type *ptr,
          int, int col_stride,
          int num_rows, int num_cols)
      {
        // only store num_rows rows with packed data
        camp::sink(
            (IDX < num_cols
            ?  m_cols[IDX].store_packed_n(ptr+IDX*col_stride, num_rows)
            :  m_cols[IDX])... // NOP, but has same as above type
        );

        return *getThis();
      }

      /*!
       * Store a strided partial matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &store_strided_nm(element_type *ptr,
          int row_stride, int col_stride,
          int num_rows, int num_cols)
      {
        // only store num_rows rows with strided data
        camp::sink(
            (IDX < num_cols
            ?  m_cols[IDX].store_strided_n(ptr+IDX*col_stride, row_stride, num_rows)
            :  m_cols[IDX])... // NOP, but has same as above type
        );

        return *getThis();
      }



      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type v){
        camp::sink((m_cols[IDX].broadcast(v))...);
        return *getThis();
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type right_multiply_vector(vector_type v) const {
        return
        RAJA::sum<vector_type>(( m_cols[IDX] * v.get(IDX))...);
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
            (m_cols[IDX])+(mat.m_cols[IDX]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type mat) const {
        return self_type(
            (m_cols[IDX])-(mat.m_cols[IDX]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(int row, int col, element_type val){
        m_cols[col].set(row, val);
        return *getThis();
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(int row, int col) const {
        return m_cols[col].get(row);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type &col(int c){
        return m_cols[c];
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type const &col(int c) const{
        return m_cols[c];
      }

  }; // MatrixImpl - COLUMN MAJOR



}  // namespace internal



}  // namespace RAJA




#endif
