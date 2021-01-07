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

#ifndef RAJA_pattern_tensor_MatrixRegisterImpl_HPP
#define RAJA_pattern_tensor_MatrixRegisterImpl_HPP

#include "camp/camp.hpp"
#include "RAJA/config.hpp"
#include "RAJA/pattern/tensor/MatrixRegister.hpp"
#include "RAJA/pattern/tensor/internal/MatrixMatrixMultiply.hpp"

//#define DEBUG_MATRIX_LOAD_STORE


namespace RAJA
{


  /*
   * 2D (Matrix) specialization of TensorRegister
   */
  template<typename REGISTER_POLICY, typename T, camp::idx_t ROW_ORD, camp::idx_t COL_ORD, camp::idx_t ROW_SIZE, camp::idx_t COL_SIZE, camp::idx_t ... VAL_SEQ>
  class TensorRegister<REGISTER_POLICY, T, TensorLayout<ROW_ORD, COL_ORD>, camp::idx_seq<ROW_SIZE, COL_SIZE>, camp::idx_seq<VAL_SEQ... >> :
    public internal::TensorRegisterBase<TensorRegister<REGISTER_POLICY, T, TensorLayout<ROW_ORD, COL_ORD>, camp::idx_seq<ROW_SIZE, COL_SIZE>, camp::idx_seq<VAL_SEQ... >>>
  {
    public:
      using self_type = TensorRegister<REGISTER_POLICY, T, TensorLayout<ROW_ORD, COL_ORD>, camp::idx_seq<ROW_SIZE, COL_SIZE>, camp::idx_seq<VAL_SEQ... >>;
      using base_type = internal::TensorRegisterBase<TensorRegister<REGISTER_POLICY, T, TensorLayout<ROW_ORD, COL_ORD>, camp::idx_seq<ROW_SIZE, COL_SIZE>, camp::idx_seq<VAL_SEQ... >>>;
      using vector_type = VectorRegister<T, REGISTER_POLICY>;
      using register_policy = REGISTER_POLICY;
      using element_type = T;
      using layout_type = TensorLayout<ROW_ORD, COL_ORD>;

    private:

      vector_type m_values[sizeof...(VAL_SEQ)];


    public:

      TensorRegister() = default;

      RAJA_HOST_DEVICE
      RAJA_INLINE
      TensorRegister(element_type c) :
        m_values{(VAL_SEQ >= 0) ? vector_type(c) : vector_type(c)...}
      {}

      TensorRegister(self_type const &c) = default;
      TensorRegister(self_type && c) = default;


      /*
       * Overload for:    assignment of ET to a TensorRegister

       */
      template<typename RHS,
        typename std::enable_if<std::is_base_of<RAJA::internal::ET::TensorExpressionConcreteBase, RHS>::value, bool>::type = true>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      TensorRegister(RHS const &rhs)
      {
        // evaluate a single tile of the ET, storing in this TensorRegister
        copy( rhs.eval(base_type::s_get_default_tile()) );
      }


      template<typename ... REGS>
      explicit
      RAJA_HOST_DEVICE
      RAJA_INLINE
      TensorRegister(vector_type reg0, REGS const &... regs) :
        m_values{reg0, regs...}
      {
        static_assert(1+sizeof...(REGS) == sizeof...(VAL_SEQ),
            "Incompatible number of registers");
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return vector_type::is_root();
      }


      /*!
       * Returns true if the underlying data packed for a given tensor ref
       *
       * This is true if either:
       *   It's column major and the rows are stride one
       *   It's row major and the columns are stride one
       */
      template<camp::idx_t STRIDE_ONE_DIM>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_ref_packed() {
        return (STRIDE_ONE_DIM == 0 && layout_type::is_column_major()) ||
            (STRIDE_ONE_DIM == 1 && layout_type::is_row_major());
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


      TensorRegister &operator=(self_type const &c) = default;

//      /*
//       * Overload for:    assignment of ET to a TensorRegister
//
//       */
//      template<typename RHS,
//        typename std::enable_if<std::is_base_of<RAJA::internal::ET::TensorExpressionConcreteBase, RHS>::value, bool>::type = true>
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      self_type const &operator=(RHS const &rhs)
//      {
//        // evaluate a single tile of the ET, storing in this TensorRegister
//        copy( rhs.eval(base_type::s_get_default_tile()) );
//
//        return *this;
//      }


      /*!
       * Provide matrix-matrix multiply for operator* between to matrices
       */
      template<typename T2, typename L, typename RP>
      self_type
      operator*(MatrixRegister<T2, L, RP> const &y) const
      {
        return matrix_multiply(y);
      }

      /*!
       * Provide right matrix-vector multiply for operator* between this
       * matrix and a vector.
       */
      template<typename T2, typename RP>
      VectorRegister<T2, RP>
      operator*(VectorRegister<T2, RP> const &y) const
      {
        return right_multiply_vector(y);
      }


      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &v){
        camp::sink((m_values[VAL_SEQ] = v.m_values[VAL_SEQ])...);
        return *this;
      }




      /*!
       * Resizes matrix to specified size, and sets all elements to zero
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &clear(){
        camp::sink(
            m_values[VAL_SEQ].broadcast(0)...
        );

        return *this;
      }



      /*!
       * @brief Performs load specified by TensorRef object.
       */
      template<typename POINTER_TYPE, typename INDEX_TYPE, internal::ET::TensorTileSize TENSOR_SIZE, camp::idx_t STRIDE_ONE_DIM>
      RAJA_INLINE
      self_type &load_ref(internal::ET::TensorRef<self_type, POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE, 2, STRIDE_ONE_DIM> const &ref){

        auto ptr = ref.m_pointer + ref.m_tile.m_begin[0]*ref.m_stride[0] +
                                   ref.m_tile.m_begin[1]*ref.m_stride[1];

        // check for packed data
        if(is_ref_packed<STRIDE_ONE_DIM>()){
          // full vector?
          if(TENSOR_SIZE == internal::ET::TENSOR_FULL){
            load_packed(ptr, ref.m_stride[0], ref.m_stride[1]);
          }
          // partial
          else{
            load_packed_nm(ptr, ref.m_stride[0], ref.m_stride[1],
                                ref.m_tile.m_size[0], ref.m_tile.m_size[1]);
          }

        }
        // strided data
        else
        {
          // full vector?
          if(TENSOR_SIZE == internal::ET::TENSOR_FULL){
            load_strided(ptr, ref.m_stride[0], ref.m_stride[1]);
          }
          // partial
          else{
            load_strided_nm(ptr, ref.m_stride[0], ref.m_stride[1],
                                ref.m_tile.m_size[0], ref.m_tile.m_size[1]);
          }
        }
        return *this;
      }


      /*!
       * @brief Performs load specified by TensorRef object.
       */
      template<typename POINTER_TYPE, typename INDEX_TYPE, internal::ET::TensorTileSize TENSOR_SIZE, camp::idx_t STRIDE_ONE_DIM>
      RAJA_INLINE
      self_type const &store_ref(internal::ET::TensorRef<self_type, POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE,2, STRIDE_ONE_DIM> const &ref) const {

        auto ptr = ref.m_pointer + ref.m_tile.m_begin[0]*ref.m_stride[0] +
                                   ref.m_tile.m_begin[1]*ref.m_stride[1];

        // check for packed data
        if(is_ref_packed<STRIDE_ONE_DIM>())
        {
          // full vector?
          if(TENSOR_SIZE == internal::ET::TENSOR_FULL){
            store_packed(ptr, ref.m_stride[0], ref.m_stride[1]);
          }
          // partial
          else{
            store_packed_nm(ptr, ref.m_stride[0], ref.m_stride[1],
                                ref.m_tile.m_size[0], ref.m_tile.m_size[1]);
          }

        }
        // strided data
        else
        {
          // full vector?
          if(TENSOR_SIZE == internal::ET::TENSOR_FULL){
            store_strided(ptr, ref.m_stride[0], ref.m_stride[1]);
          }
          // partial
          else{
            store_strided_nm(ptr, ref.m_stride[0], ref.m_stride[1],
                                ref.m_tile.m_size[0], ref.m_tile.m_size[1]);
          }
        }
        return *this;
        return *this;
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
        if(layout_type::is_row_major()){
          camp::sink(
              m_values[VAL_SEQ].load_packed(ptr+VAL_SEQ*row_stride)...
          );
        }
        else{
          camp::sink(
              m_values[VAL_SEQ].load_packed(ptr+VAL_SEQ*col_stride)...
          );
        }

        return *this;
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
        if(layout_type::is_row_major()){
          camp::sink(
              m_values[VAL_SEQ].load_strided(ptr+VAL_SEQ*row_stride, col_stride)...
          );
        }
        else{
          camp::sink(
              m_values[VAL_SEQ].load_strided(ptr+VAL_SEQ*col_stride, row_stride)...
          );
        }

        return *this;
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

        if(layout_type::is_row_major()){
          camp::sink(
              (VAL_SEQ < num_rows
              ?  m_values[VAL_SEQ].load_packed_n(ptr+VAL_SEQ*row_stride, num_cols)
              :  m_values[VAL_SEQ].broadcast(0))... // clear to len N
          );
        }
        else{
          camp::sink(
              (VAL_SEQ < num_cols
              ?  m_values[VAL_SEQ].load_packed_n(ptr+VAL_SEQ*col_stride, num_rows)
              :  m_values[VAL_SEQ].broadcast(0))... // clear to len N
          );
        }

        return *this;
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
        if(layout_type::is_row_major()){
          camp::sink(
              (VAL_SEQ < num_rows
              ?  m_values[VAL_SEQ].load_strided_n(ptr+VAL_SEQ*row_stride, col_stride, num_cols)
              :  m_values[VAL_SEQ].broadcast(0))... // clear to len N
          );
        }
        else{
          camp::sink(
              (VAL_SEQ < num_cols
              ?  m_values[VAL_SEQ].load_strided_n(ptr+VAL_SEQ*col_stride, row_stride, num_rows)
              :  m_values[VAL_SEQ].broadcast(0))... // clear to len N
          );
        }

        return *this;
      }



      /*!
       * Store a dense full matrix to memory.
       *
       * Column entries must be stride-1, rows may be any striding
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_packed(element_type *ptr,
          int row_stride, int col_stride) const
      {
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: store_packed, stride=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, 1);
#endif
        if(layout_type::is_row_major()){
          camp::sink(
              m_values[VAL_SEQ].store_packed(ptr+VAL_SEQ*row_stride)...
          );
        }
        else{
          camp::sink(
              m_values[VAL_SEQ].store_packed(ptr+VAL_SEQ*col_stride)...
          );
        }

        return *this;
      }

      /*!
       * Store a strided full matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_strided(element_type *ptr,
          int row_stride, int col_stride) const
      {
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: store_strided, stride=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, col_stride);
#endif
        if(layout_type::is_row_major()){
          // store all rows width a column stride
          camp::sink(
              m_values[VAL_SEQ].store_strided(ptr+VAL_SEQ*row_stride, col_stride)...
          );
        }
        else{
          // store all rows width a column stride
          camp::sink(
              m_values[VAL_SEQ].store_strided(ptr+VAL_SEQ*col_stride, row_stride)...
          );
        }


        return *this;
      }

      /*!
       * Store a dense partial matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_packed_nm(element_type *ptr,
          int row_stride, int col_stride,
          int num_rows, int num_cols) const
      {
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: RM store_packed_nm, stride=%d,%d, nm=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, 1, num_rows, num_cols);
#endif
        if(layout_type::is_row_major()){
          camp::sink(
              (VAL_SEQ < num_rows
              ?  m_values[VAL_SEQ].store_packed_n(ptr+VAL_SEQ*row_stride, num_cols)
              :  m_values[VAL_SEQ])... // NOP, but has same as above type
          );
        }
        else {
          camp::sink(
              (VAL_SEQ < num_cols
              ?  m_values[VAL_SEQ].store_packed_n(ptr+VAL_SEQ*col_stride, num_rows)
              :  m_values[VAL_SEQ])... // NOP, but has same as above type
          );
        }

        return *this;
      }

      /*!
       * Store a strided partial matrix to memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_strided_nm(element_type *ptr,
          int row_stride, int col_stride,
          int num_rows, int num_cols) const
      {
#if defined(__CUDA_ARCH__) && defined(DEBUG_MATRIX_LOAD_STORE)
        printf("th%d,%d: RM store_strided_nm, stride=%d,%d, nm=%d,%d\n",
            threadIdx.x, threadIdx.y, row_stride, col_stride, num_rows, num_cols);
#endif
        if(layout_type::is_row_major()){
          camp::sink(
              (VAL_SEQ < num_rows
              ?  m_values[VAL_SEQ].store_strided_n(ptr+VAL_SEQ*row_stride, col_stride, num_cols)
              :  m_values[VAL_SEQ])... // NOP, but has same as above type
          );
        }
        else {
          camp::sink(
              (VAL_SEQ < num_cols
              ?  m_values[VAL_SEQ].store_strided_n(ptr+VAL_SEQ*col_stride, row_stride, num_rows)
              :  m_values[VAL_SEQ])... // NOP, but has same as above type
          );
        }

        return *this;
      }




      /*!
       * Copy contents of another matrix operator
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type v){
        camp::sink((m_values[VAL_SEQ].broadcast(v))...);
        return *this;
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type right_multiply_vector(vector_type v) const {
        if(layout_type::is_row_major()){
          vector_type result;
          camp::sink(
              result.set(VAL_SEQ, v.dot(m_values[VAL_SEQ]))...
              );

          return result;
        }
        else{
          return
                RAJA::sum<vector_type>(( m_values[VAL_SEQ] * v.get(VAL_SEQ))...);
        }
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type left_multiply_vector(vector_type v) const {

        if(layout_type::is_column_major()){
          vector_type result;
          camp::sink(
              result.set(VAL_SEQ, v.dot(m_values[VAL_SEQ]))...
              );

          return result;
        }
        else{
          return
                RAJA::sum<vector_type>(( m_values[VAL_SEQ] * v.get(VAL_SEQ))...);
        }
      }


      /*!
       * element-wise multiplication
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type mat) const {
        return self_type(
            (m_values[VAL_SEQ])*(mat.m_values[VAL_SEQ]) ...
        );
      }

      /*!
       * element-wise fused multiply add
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply_add(self_type mat, self_type add) const {
        return self_type(
            m_values[VAL_SEQ].multiply_add(mat.m_values[VAL_SEQ], add) ...
        );
      }


      /*!
       * Matrix-Matrix product
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename internal::MatrixMatrixMultiplyHelper<self_type, RMAT>::result_type
      matrix_multiply(RMAT const &mat) const {
        return internal::MatrixMatrixMultiplyHelper<self_type,RMAT>::multiply(*this, mat);
      }

      /*!
       * Matrix-Matrix multiply accumulate
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename internal::MatrixMatrixMultiplyHelper<self_type, RMAT>::result_type
      matrix_multiply_add(RMAT const &B, typename internal::MatrixMatrixMultiplyHelper<self_type, RMAT>::result_type const &C) const {
        return internal::MatrixMatrixMultiplyHelper<self_type,RMAT>::multiply_accumulate(*this, B, C);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(self_type mat) const {
        return self_type(
            (m_values[VAL_SEQ])+(mat.m_values[VAL_SEQ]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type mat) const {
        return self_type(
            (m_values[VAL_SEQ])-(mat.m_values[VAL_SEQ]) ...
        );
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(int row, int col, element_type val){
        if(layout_type::is_row_major()){
          m_values[row].set(col, val);
        }
        else{
          m_values[col].set(row, val);
        }
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(int row, int col) const {
        return layout_type::is_row_major() ?
             m_values[row].get(col) :
             m_values[col].get(row);
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type &vec(int i){
        return m_values[i];
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      vector_type const &vec(int i) const{
        return m_values[i];
      }


      template<typename IDX_I, typename IDX_J>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type operator()(IDX_I row, IDX_J col){
        return this->get(row, col);
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
            s += std::to_string(this->get(r,c));
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







}  // namespace RAJA




#endif
