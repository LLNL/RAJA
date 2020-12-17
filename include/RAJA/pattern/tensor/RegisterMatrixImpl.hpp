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

#ifndef RAJA_pattern_simd_register_matriximpl_HPP
#define RAJA_pattern_simd_register_matriximpl_HPP

#include "camp/camp.hpp"
#include "RAJA/config.hpp"
#include "RAJA/policy/tensor/arch.hpp"


//#define DEBUG_MATRIX_LOAD_STORE


namespace RAJA
{

  template<camp::idx_t ROW, camp::idx_t COL>
  struct MatrixLayout : public camp::idx_seq<ROW, COL>{
    static_assert(ROW == 0 || COL == 0, "invalid template arguments");
    static_assert(ROW == 1 || COL == 1, "invalid template arguments");
    static_assert(ROW+COL == 1, "invalid template arguments");

    RAJA_INLINE
    RAJA_HOST_DEVICE
    static
    constexpr
    bool is_column_major(){
      return COL == 1;
    }

    RAJA_INLINE
    RAJA_HOST_DEVICE
    static
    constexpr
    bool is_row_major(){
      return ROW == 1;
    }
  };


  using MATRIX_ROW_MAJOR = MatrixLayout<1, 0>;
  using MATRIX_COL_MAJOR = MatrixLayout<0, 1>;

  struct VectorLayout{};


  namespace internal{
    template<typename REGISTER_POLICY, typename ELEMENT_TYPE, typename LAYOUT, typename IDX_SEQ>
    class RegisterMatrixImpl;
  }

  template<typename T, typename LAYOUT, typename REGISTER_POLICY = RAJA::default_register>
  using RegisterMatrix = internal::RegisterMatrixImpl<
      REGISTER_POLICY, T, LAYOUT,
      camp::make_idx_seq_t<RegisterTraits<REGISTER_POLICY, T>::s_num_elem> >;



namespace internal {


  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, typename LAYOUT, typename IDX_SEQ>
  class RegisterMatrixImpl;



  template<typename MATA, typename MATB, typename IDX_SEQ>
  struct MatrixMatrixProductHelperExpanded;


  template<typename ELEMENT_TYPE, typename LAYOUT, typename REGISTER_POLICY, camp::idx_t ... REG_IDX>
  struct MatrixMatrixProductHelperExpanded<
    RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>,
    RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>,
    camp::idx_seq<REG_IDX...>>
  {
      using matrix_type = RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>;
      using vector_type = VectorRegister<REGISTER_POLICY, ELEMENT_TYPE>;
      using result_type = matrix_type;

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      int calc_vec_product(vector_type &sum, vector_type const &a_vec, matrix_type const &B){

        camp::sink(
                (sum =
                    B.vec(REG_IDX).fused_multiply_add(
                        a_vec.get_and_broadcast(REG_IDX),
                        sum))...
                );

        return 0;
      }

      template<camp::idx_t J>
      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      int calc_vec_product2(matrix_type &sum, matrix_type const &A, matrix_type const &B){

        camp::sink(
                (sum.vec(REG_IDX) =
                    B.vec(J).fused_multiply_add(
                        A.vec(REG_IDX).get_and_broadcast(J),
                        sum.vec(REG_IDX)))...
                );

        return 0;
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      matrix_type multiply(matrix_type const &A, matrix_type const &B){
        matrix_type sum(0);

        if(LAYOUT::is_row_major()){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_matrix_mm_mult_row_row ++;
#endif
//          camp::sink(calc_vec_product(sum.vec(REG_IDX), A.vec(REG_IDX), B)...);
          camp::sink(calc_vec_product2<REG_IDX>(sum, A, B)...);
        }
        else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_matrix_mm_mult_col_col ++;
#endif
//          camp::sink(calc_vec_product(sum.vec(REG_IDX), B.vec(REG_IDX), A)...);
          camp::sink(calc_vec_product2<REG_IDX>(sum, B, A)...);
        }
        return sum;
      }

      RAJA_HOST_DEVICE
      static
      RAJA_INLINE
      matrix_type multiply_accumulate(matrix_type const &A, matrix_type const &B, matrix_type C){
        if(LAYOUT::is_row_major()){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_matrix_mm_multacc_row_row ++;
#endif
          camp::sink(calc_vec_product(C.vec(REG_IDX), A.vec(REG_IDX), B)...);
        }
        else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_matrix_mm_multacc_col_col ++;
#endif
          camp::sink(calc_vec_product(C.vec(REG_IDX), B.vec(REG_IDX), A)...);
        }
        return C;
      }

  };






  template<typename MATA, typename MATB>
  struct MatrixMatrixProductHelper;

  template<typename ELEMENT_TYPE, typename LAYOUT, typename REGISTER_POLICY>
  struct MatrixMatrixProductHelper<
    RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>,
    RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>> :
  public
      MatrixMatrixProductHelperExpanded<RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>,
                                        RegisterMatrix<ELEMENT_TYPE, LAYOUT, REGISTER_POLICY>,
                                        camp::make_idx_seq_t<VectorRegister<REGISTER_POLICY, ELEMENT_TYPE>::s_num_elem>>
    {};




} // namespace internal
} // namespace RAJA



namespace RAJA
{
namespace internal {







  /*
   * Row-Major implementation of MatrixImpl
   */
  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, typename LAYOUT, camp::idx_t ... REG_IDX>
  class RegisterMatrixImpl<REGISTER_POLICY, ELEMENT_TYPE, LAYOUT, camp::idx_seq<REG_IDX...>>
  {
    public:
      using self_type = RegisterMatrixImpl<REGISTER_POLICY, ELEMENT_TYPE, LAYOUT, camp::idx_seq<REG_IDX...>>;

      //using vector_type = TensorRegister<REGISTER_POLICY, ELEMENT_TYPE, VectorLayout, camp::idx_seq<4>, 0>;
      using vector_type = VectorRegister<REGISTER_POLICY, ELEMENT_TYPE>;
      using register_policy = REGISTER_POLICY;
      using element_type = ELEMENT_TYPE;

      using layout_type = LAYOUT;


    private:

      vector_type m_registers[sizeof...(REG_IDX)];

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

      RegisterMatrixImpl() = default;

      RAJA_HOST_DEVICE
      RAJA_INLINE
      RegisterMatrixImpl(element_type c) :
        m_registers{(REG_IDX >= 0) ? vector_type(c) : vector_type(c)...}
      {}

      RegisterMatrixImpl(self_type const &c) = default;
      RegisterMatrixImpl(self_type && c) = default;


      template<typename ... REGS>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      RegisterMatrixImpl(vector_type reg0, REGS const &... regs) :
        m_registers{reg0, regs...}
      {
        static_assert(1+sizeof...(REGS) == sizeof...(REG_IDX),
            "Incompatible number of registers");
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return vector_type::is_root();
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_column_major() {
        return LAYOUT::is_column_major();
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_row_major() {
        return LAYOUT::is_row_major();
      }

      /*!
       * Gets the maximum size of matrix along specified dimension
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr camp::idx_t s_dim_elem(camp::idx_t ){
        return vector_type::s_num_elem;
      }

      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(element_type value)
      {
        broadcast(value);
        return *this;
      }


      RegisterMatrixImpl &operator=(self_type const &c) = default;


      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &v){
        camp::sink((m_registers[REG_IDX] = v.m_registers[REG_IDX])...);
        return *getThis();
      }




      /*!
       * Resizes matrix to specified size, and sets all elements to zero
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &clear(){
        camp::sink(
            m_registers[REG_IDX].broadcast(0)...
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
          int row_stride, int col_stride)
      {
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: load_packed, stride=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, col_stride);
#endif
        if(LAYOUT::is_row_major()){
          camp::sink(
              m_registers[REG_IDX].load_packed(ptr+REG_IDX*row_stride)...
          );
        }
        else{
          camp::sink(
              m_registers[REG_IDX].load_packed(ptr+REG_IDX*col_stride)...
          );
        }

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
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: load_strided, stride=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, col_stride);
#endif
        if(LAYOUT::is_row_major()){
          camp::sink(
              m_registers[REG_IDX].load_strided(ptr+REG_IDX*row_stride, col_stride)...
          );
        }
        else{
          camp::sink(
              m_registers[REG_IDX].load_strided(ptr+REG_IDX*col_stride, row_stride)...
          );
        }

        return *getThis();
      }

      /*!
       * Loads a dense partial matrix from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_packed_nm(element_type const *ptr,
          int row_stride, int col_stride,
          int num_rows, int num_cols)
      {
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: load_packed_nm, stride=%d,%d, nm=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, 1, num_rows, num_cols);
#endif

        if(LAYOUT::is_row_major()){
          camp::sink(
              (REG_IDX < num_rows
              ?  m_registers[REG_IDX].load_packed_n(ptr+REG_IDX*row_stride, num_cols)
              :  m_registers[REG_IDX].broadcast(0))... // clear to len N
          );
        }
        else{
          camp::sink(
              (REG_IDX < num_cols
              ?  m_registers[REG_IDX].load_packed_n(ptr+REG_IDX*col_stride, num_rows)
              :  m_registers[REG_IDX].broadcast(0))... // clear to len N
          );
        }

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
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: load_strided_nm, stride=%d,%d, nm=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, col_stride, num_rows, num_cols);
#endif
        if(LAYOUT::is_row_major()){
          camp::sink(
              (REG_IDX < num_rows
              ?  m_registers[REG_IDX].load_strided_n(ptr+REG_IDX*row_stride, col_stride, num_cols)
              :  m_registers[REG_IDX].broadcast(0))... // clear to len N
          );
        }
        else{
          camp::sink(
              (REG_IDX < num_cols
              ?  m_registers[REG_IDX].load_strided_n(ptr+REG_IDX*col_stride, row_stride, num_rows)
              :  m_registers[REG_IDX].broadcast(0))... // clear to len N
          );
        }

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
          int row_stride, int col_stride)
      {
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: store_packed, stride=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, 1);
#endif
        if(LAYOUT::is_row_major()){
          camp::sink(
              m_registers[REG_IDX].store_packed(ptr+REG_IDX*row_stride)...
          );
        }
        else{
          camp::sink(
              m_registers[REG_IDX].store_packed(ptr+REG_IDX*col_stride)...
          );
        }

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
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: store_strided, stride=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, col_stride);
#endif
        if(LAYOUT::is_row_major()){
          // store all rows width a column stride
          camp::sink(
              m_registers[REG_IDX].store_strided(ptr+REG_IDX*row_stride, col_stride)...
          );
        }
        else{
          // store all rows width a column stride
          camp::sink(
              m_registers[REG_IDX].store_strided(ptr+REG_IDX*col_stride, row_stride)...
          );
        }


        return *getThis();
      }

      /*!
       * Store a dense partial matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &store_packed_nm(element_type *ptr,
          int row_stride, int col_stride,
          int num_rows, int num_cols)
      {
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: RM store_packed_nm, stride=%d,%d, nm=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, 1, num_rows, num_cols);
#endif
        if(LAYOUT::is_row_major()){
          camp::sink(
              (REG_IDX < num_rows
              ?  m_registers[REG_IDX].store_packed_n(ptr+REG_IDX*row_stride, num_cols)
              :  m_registers[REG_IDX])... // NOP, but has same as above type
          );
        }
        else {
          camp::sink(
              (REG_IDX < num_cols
              ?  m_registers[REG_IDX].store_packed_n(ptr+REG_IDX*col_stride, num_rows)
              :  m_registers[REG_IDX])... // NOP, but has same as above type
          );
        }

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
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: RM store_strided_nm, stride=%d,%d, nm=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, col_stride, num_rows, num_cols);
#endif
        if(LAYOUT::is_row_major()){
          camp::sink(
              (REG_IDX < num_rows
              ?  m_registers[REG_IDX].store_strided_n(ptr+REG_IDX*row_stride, col_stride, num_cols)
              :  m_registers[REG_IDX])... // NOP, but has same as above type
          );
        }
        else {
          camp::sink(
              (REG_IDX < num_cols
              ?  m_registers[REG_IDX].store_strided_n(ptr+REG_IDX*col_stride, row_stride, num_rows)
              :  m_registers[REG_IDX])... // NOP, but has same as above type
          );
        }

        return *getThis();
      }




      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type v){
        camp::sink((m_registers[REG_IDX].broadcast(v))...);
        return *getThis();
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type right_multiply_vector(vector_type v) const {
        if(LAYOUT::is_row_major()){
          vector_type result;
          camp::sink(
              result.set(REG_IDX, v.dot(m_registers[REG_IDX]))...
              );

          return result;
        }
        else{
          return
                RAJA::sum<vector_type>(( m_registers[REG_IDX] * v.get(REG_IDX))...);
        }
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
            (m_registers[REG_IDX])+(mat.m_registers[REG_IDX]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type mat) const {
        return self_type(
            (m_registers[REG_IDX])-(mat.m_registers[REG_IDX]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(int row, int col, element_type val){
        if(LAYOUT::is_row_major()){
          m_registers[row].set(col, val);
        }
        else{
          m_registers[col].set(row, val);
        }
        return *getThis();
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(int row, int col) const {
        return LAYOUT::is_row_major() ?
             m_registers[row].get(col) :
             m_registers[col].get(row);
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type &vec(int i){
        return m_registers[i];
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type const &vec(int i) const{
        return m_registers[i];
      }


      template<typename IDX_I, typename IDX_J>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type operator()(IDX_I row, IDX_J col){
        return getThis()->get(row, col);
      }



      /*!
       * @brief Add two vector registers
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator+(self_type const &x) const
      {
        return getThis()->add(x);
      }


      /*!
       * @brief Add a vector to this vector
       * @param x Vector to add to this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator+=(self_type const &x)
      {
        *getThis() = getThis()->add(x);
        return *getThis();
      }

      /*!
       * @brief Negate the value of this vector
       * @return Value of -(*this)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator-() const
      {
        return self_type(0).subtract(*getThis());
      }

      /*!
       * @brief Subtract two vector registers
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator-(self_type const &x) const
      {
        return getThis()->subtract(x);
      }

      /*!
       * @brief Subtract a vector from this vector
       * @param x Vector to subtract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator-=(self_type const &x)
      {
        *getThis() = getThis()->subtract(x);
        return *getThis();
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type operator*(vector_type v) const {
        return getThis()->right_multiply_vector(v);
      }

      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename MatrixMatrixProductHelper<self_type, self_type>::result_type
      operator*(self_type const &mat) const {
        return getThis()->multiply(mat);
      }


      /*!
       * @brief Multiply a vector with this vector
       * @param x Vector to multiple with this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator*=(self_type const &x)
      {
        *getThis() = getThis()->multiply(x);
        return *getThis();
      }

      /*!
       * @brief Converts to matrix to a string
       *
       *
       */
      RAJA_INLINE
      std::string toString(bool one_line=false) const {
        std::string s = "Matrix(" + std::to_string(vector_type::s_num_elem) +
            "x" + std::to_string(vector_type::s_num_elem);
        if(!one_line){
          s +=")\n";
        }


        s += "[ ";

        //
        for(camp::idx_t r = 0;r < vector_type::s_num_elem; ++ r){
          if(r > 0){
            s += ", ";
            if(!one_line){
              s+= "\n  ";
            }
          }
          s += "[";
          for(camp::idx_t c = 0;c < vector_type::s_num_elem; ++ c){
            if(c > 0){
              s += ", ";
            }
            s += std::to_string(getThis()->get(r,c));
          }
          s += "]";
        }

        s += " ]";
        if(!one_line){
          s+="\n";
        }
        return s;
      }

  }; // MatrixImpl - ROW MAJOR




}  // namespace internal



}  // namespace RAJA




#endif
