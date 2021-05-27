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
#include "RAJA/util/BitMask.hpp"

//#define DEBUG_MATRIX_LOAD_STORE


namespace RAJA
{

  namespace internal {

    /**
     * This is a helper class that provides mapping from matrix indices to
     * elements in a register.
     */
    template<typename LAYOUT, camp::idx_t ROW_SIZE, camp::idx_t COL_SIZE, camp::idx_t REGISTER_SIZE>
    class MatrixRegisterMap {
      public:
        using layout_type = LAYOUT;

        // number of values in one register
        static constexpr camp::idx_t s_elements_per_register = REGISTER_SIZE;

        // number of registers to hold entire matrix
        static constexpr camp::idx_t s_num_registers =
            (ROW_SIZE*COL_SIZE) / s_elements_per_register;

        // We only allow matrix sizes that exactly fit in some number of registers
        static_assert((ROW_SIZE*COL_SIZE) == s_num_registers*s_elements_per_register,
            "Matrix must exactly fit into an integer number of registers");

//        // Matrix size for the within-register dimension
//        // for row-major, regdim is the columns
//        // for column-major, regdim is the rows
//        static constexpr camp::idx_t s_elements_per_regdim =
//            layout_type::is_row_major() ? COL_SIZE : ROW_SIZE;
//
//        // Number of registers per regdim, if there are 1 or more registers
//        // If a register is split over multiple regdims, this is zero
//        static constexpr camp::idx_t s_registers_per_regdim =
//            s_elements_per_regdim / s_elements_per_register;
//
//        // Number of regdim per register, if there are 1 or more regdim
//        // If multiple registers are used for a regdim, this is zero
//        static constexpr camp::idx_t s_regdim_per_register =
//            s_elements_per_register / s_elements_per_regdim;


        using log_base2_t = RAJA::LogBase2<s_elements_per_register>;

        static constexpr camp::idx_t s_shift_per_register =
            log_base2_t::value;

        static constexpr camp::idx_t s_mask_per_register =
            (1<<log_base2_t::value)-1;



      public:
        template<typename IDX>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        constexpr
        static
        auto to_register(IDX row, IDX col) -> IDX {
          return layout_type::is_row_major() ?
              (row*IDX(COL_SIZE) + col) >> IDX(s_shift_per_register) :
              (col*IDX(ROW_SIZE) + row) >> IDX(s_shift_per_register);
        }

        template<typename IDX>
        RAJA_INLINE
        RAJA_HOST_DEVICE
        constexpr
        static
        auto to_lane(IDX row, IDX col) -> IDX {
          return layout_type::is_row_major() ?
              (row*IDX(COL_SIZE) + col) & IDX(s_mask_per_register) :
              (col*IDX(ROW_SIZE) + row) & IDX(s_mask_per_register);
        }
    };
  }

  /*
   * 2D (Matrix) specialization of TensorRegister
   */
  template<typename REGISTER_POLICY, typename T, camp::idx_t ROW_ORD, camp::idx_t COL_ORD, camp::idx_t ROW_SIZE, camp::idx_t COL_SIZE>
  class TensorRegister<REGISTER_POLICY, T, TensorLayout<ROW_ORD, COL_ORD>, camp::idx_seq<ROW_SIZE, COL_SIZE>> :
    public internal::TensorRegisterBase<TensorRegister<REGISTER_POLICY, T, TensorLayout<ROW_ORD, COL_ORD>, camp::idx_seq<ROW_SIZE, COL_SIZE>>>
  {
    public:
      using self_type = TensorRegister<REGISTER_POLICY, T, TensorLayout<ROW_ORD, COL_ORD>, camp::idx_seq<ROW_SIZE, COL_SIZE>>;
      using base_type = internal::TensorRegisterBase<TensorRegister<REGISTER_POLICY, T, TensorLayout<ROW_ORD, COL_ORD>, camp::idx_seq<ROW_SIZE, COL_SIZE>>>;
      using register_type = VectorRegister<T, REGISTER_POLICY>;
      using register_policy = REGISTER_POLICY;
      using element_type = T;
      using layout_type = TensorLayout<ROW_ORD, COL_ORD>;

      using transpose_tensor_type = TensorRegister<REGISTER_POLICY, T, TensorLayout<!ROW_ORD, !COL_ORD>, camp::idx_seq<ROW_SIZE, COL_SIZE>>;

    private:

      static constexpr camp::idx_t s_register_width =
          RegisterTraits<REGISTER_POLICY,T>::s_num_elem;

      // provides a mapping of indices to registers
      using register_map = internal::MatrixRegisterMap<layout_type, ROW_SIZE, COL_SIZE, s_register_width>;

      // Number of registers that completely contain this matrix
      static constexpr camp::idx_t s_num_registers = register_map::s_num_registers;

      using base_type::m_registers;

    public:




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
      constexpr camp::idx_t s_dim_elem(camp::idx_t dim){
        return dim == 0 ? ROW_SIZE : COL_SIZE;
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

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(self_type const &c){
        return copy(c);
      }


      /*!
       * Provide matrix-matrix multiply for operator* between to matrices
       */
      template<typename T2, typename L, typename RP>
      self_type
      operator*(SquareMatrixRegister<T2, L, RP> const &y) const
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
       * @brief Performs load specified by TensorRef object.
       */
      template<typename POINTER_TYPE, typename INDEX_TYPE, internal::TensorTileSize TENSOR_SIZE, camp::idx_t STRIDE_ONE_DIM>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &load_ref(internal::TensorRef<POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE, 2, STRIDE_ONE_DIM> const &ref){

        auto ptr = ref.m_pointer + ref.m_tile.m_begin[0]*ref.m_stride[0] +
                                   ref.m_tile.m_begin[1]*ref.m_stride[1];

        // check for packed data
        if(is_ref_packed<STRIDE_ONE_DIM>()){
          // full vector?
          if(TENSOR_SIZE == internal::TENSOR_FULL){
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
          if(TENSOR_SIZE == internal::TENSOR_FULL){
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
      template<typename POINTER_TYPE, typename INDEX_TYPE, internal::TensorTileSize TENSOR_SIZE, camp::idx_t STRIDE_ONE_DIM>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type const &store_ref(internal::TensorRef<POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE,2, STRIDE_ONE_DIM> const &ref) const {

        auto ptr = ref.m_pointer + ref.m_tile.m_begin[0]*ref.m_stride[0] +
                                   ref.m_tile.m_begin[1]*ref.m_stride[1];

        // check for packed data
        if(is_ref_packed<STRIDE_ONE_DIM>())
        {
          // full vector?
          if(TENSOR_SIZE == internal::TENSOR_FULL){
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
          if(TENSOR_SIZE == internal::TENSOR_FULL){
            store_strided(ptr, ref.m_stride[0], ref.m_stride[1]);
          }
          // partial
          else{
            store_strided_nm(ptr, ref.m_stride[0], ref.m_stride[1],
                                ref.m_tile.m_size[0], ref.m_tile.m_size[1]);
          }
        }
        return *this;
      }



      /*!
       * Loads a dense full matrix from memory.
       *
       * For row-major, column entries must be stride-1
       * For column-major, row entries must be stride-1
       *
       * Non-stride-1 dimension can have any striding... so this is can
       * be a "semi-dense" matrix.
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
        // if it's dense in columns and rows, just do a dense load
        if((layout_type::is_row_major()&&(row_stride==ROW_SIZE)) ||
           (layout_type::is_column_major()&&(col_stride==COL_SIZE))){

          for(camp::idx_t reg = 0;reg < s_num_registers;++ reg){
            m_registers[reg].load_packed(ptr + reg*s_register_width);
          }

        }
        // Do semi-dense load for row-major
        else if(layout_type::is_row_major()){

          // one or more registers per column
          if(true){  //s_registers_per_dim){
            for(camp::idx_t row = 0;row < ROW_SIZE;++ row){
//              for(camp::idx_t dimreg = 0;dimreg < s_registers_per_dim;++ dimreg){

//                camp::idx_t reg = dimreg + row*s_registers_per_dim;

                camp::idx_t offset = row*row_stride;// + dimreg*s_register_width;

                m_registers[row].load_packed(ptr + offset);

//              }
            }
          }
          // more than one column per register
          else{
            // yikes!
          }
        }
        // Do semi-dense load for column-major
        else{
          // one or more registers per row
          if(true){ //s_registers_per_dim){
            for(camp::idx_t col = 0;col < COL_SIZE;++ col){
//              for(camp::idx_t dimreg = 0;dimreg < s_registers_per_dim;++ dimreg){

//                camp::idx_t reg = dimreg + col*s_registers_per_dim;

                camp::idx_t offset = col*col_stride; // + dimreg*s_register_width;

                m_registers[col].load_packed(ptr + offset);

//              }
            }
          }
          // more than one row per register
          else{
            // yikes!
          }
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
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            m_registers[i].load_strided(ptr+i*row_stride, col_stride);
          }
        }
        else{
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            m_registers[i].load_strided(ptr+i*col_stride, row_stride);
          }
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
          for(camp::idx_t i = 0;i < num_rows;++ i){
            m_registers[i].load_packed_n(ptr+i*row_stride, num_cols);
          }
          for(camp::idx_t i = num_rows;i < s_num_registers;++ i){
            m_registers[i] = register_type(0); // clear remainder
          }
        }
        else{
          for(camp::idx_t i = 0;i < num_cols;++ i){
            m_registers[i].load_packed_n(ptr+i*col_stride, num_rows);
          }
          for(camp::idx_t i = num_cols;i < s_num_registers;++ i){
            m_registers[i] = register_type(0); // clear remainder
          }
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
          for(camp::idx_t i = 0;i < num_rows;++ i){
            m_registers[i].load_strided_n(ptr+i*row_stride, col_stride, num_cols);
          }
          for(camp::idx_t i = num_rows;i < s_num_registers;++ i){
            m_registers[i] = register_type(0); // clear remainder
          }
        }
        else{
          for(camp::idx_t i = 0;i < num_cols;++ i){
            m_registers[i].load_strided_n(ptr+i*col_stride, row_stride, num_rows);
          }
          for(camp::idx_t i = num_cols;i < s_num_registers;++ i){
            m_registers[i] = register_type(0); // clear remainder
          }
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
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            m_registers[i].store_packed(ptr+i*row_stride);
          }
        }
        else{
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            m_registers[i].store_packed(ptr+i*col_stride);
          }
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
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            m_registers[i].store_strided(ptr+i*row_stride, col_stride);
          }
        }
        else{
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            m_registers[i].store_strided(ptr+i*col_stride, row_stride);
          }
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
          for(camp::idx_t i = 0;i < num_rows;++ i){
            m_registers[i].store_packed_n(ptr+i*row_stride, num_cols);
          }
        }
        else{
          for(camp::idx_t i = 0;i < num_cols;++ i){
            m_registers[i].store_packed_n(ptr+i*col_stride, num_rows);
          }
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
          for(camp::idx_t i = 0;i < num_rows;++ i){
            m_registers[i].store_strided_n(ptr+i*row_stride, col_stride, num_cols);
          }
        }
        else{
          for(camp::idx_t i = 0;i < num_cols;++ i){
            m_registers[i].store_strided_n(ptr+i*col_stride, row_stride, num_rows);
          }
        }

        return *this;
      }







      /*!
       * Matrix transpose, keeping layout
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type transpose() const {

        static constexpr camp::idx_t num_elem = register_type::s_num_elem;

        /*
         * We use Eklundh's Algorithm: Recursive block transpose because
         * it's easy to implement using SIMD register permutation primitives
         *
         * Executes in n*log(n) row operations
         *
         */
        self_type result = *this;
        for(camp::idx_t lvl = 0; (1<<lvl) < num_elem;++ lvl){
          // At this level, we do block transposes of NxN sub-matrices, where
          // N = 1<<lvl

          auto const &vals = result.m_registers;

          self_type tmp;
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            if(((i>>lvl)&0x1) == 0){
              tmp.m_registers[i] = vals[i - (i&(1<<lvl))].transpose_shuffle_left(lvl, vals[i - (i&(1<<lvl)) + (1<<lvl)]);
            }
            else{
              tmp.m_registers[i] = vals[i - (i&(1<<lvl))].transpose_shuffle_right(lvl, vals[i - (i&(1<<lvl)) + (1<<lvl)]);
            }
          }
          result = tmp;
        }

        return result;

      }


      /*!
       * Matrix transpose inplace
       *
       * Modifies contents of this matrix
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void inplace_transpose() {
        *this = transpose();
      }

      /*!
       * Transpose this matrix by swapping row/column majorness
       *
       * Row major matrix returns column major, and visa versa.
       *
       * This has zero cost.
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      transpose_tensor_type const &transpose_type() const {
        return reinterpret_cast<transpose_tensor_type const &>(*this);
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      register_type right_multiply_vector(register_type v) const {
        if(layout_type::is_row_major()){
          register_type result;
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            result.set(v.dot(m_registers[i]), i);
          }
          return result;
        }
        else{
          register_type result(0);
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            result +=  m_registers[i] * v.get(i);
          }
          return result;
        }
      }

      /*!
       * Matrix vector product
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      register_type left_multiply_vector(register_type v) const {
        if(layout_type::is_column_major()){
          register_type result;
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            result.set(v.dot(m_registers[i]), i);
          }
          return result;
        }
        else{
          register_type result(0);
          for(camp::idx_t i = 0;i < s_num_registers;++ i){
            result +=  m_registers[i] * v.get(i);
          }
          return result;
        }
      }


      /*!
       * Matrix vector product with accumulation into another vector
       *
       * acc += (this) * v
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void right_multiply_vector_accumulate(register_type &acc, register_type v) const {
        acc.inplace_add(right_multiply_vector(v));
      }

      /*!
       * Matrix vector product with accumulation into another vector
       *
       * acc += v * (this)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void left_multiply_vector_accumulate(register_type &acc, register_type v) const {
        acc.inplace_add(left_multiply_vector(v));
      }





      /*!
       * Matrix-Matrix product
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename internal::MatrixMatrixMultiplyHelper<self_type, RMAT>::result_type
      matrix_multiply(RMAT const &mat) const {
        typename internal::MatrixMatrixMultiplyHelper<self_type, RMAT>::result_type res(0);
        internal::MatrixMatrixMultiplyHelper<self_type,RMAT>::multiply(*this, mat, res);
        return res;
      }

      /*!
       * Matrix-Matrix multiply add
       */
      template<typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      typename internal::MatrixMatrixMultiplyHelper<self_type, RMAT>::result_type
      matrix_multiply_add(RMAT const &B, typename internal::MatrixMatrixMultiplyHelper<self_type, RMAT>::result_type const &C) const {
        typename internal::MatrixMatrixMultiplyHelper<self_type, RMAT>::result_type res(C);
        internal::MatrixMatrixMultiplyHelper<self_type,RMAT>::multiply_accumulate(*this, B, res);
        return res;
      }

      /*!
       * Matrix-Matrix multiply accumulate
       */
      template<typename ACCMAT, typename RMAT>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void
      matrix_multiply_accumulate(ACCMAT &acc, RMAT const &B) const {
        internal::MatrixMatrixMultiplyHelper<self_type,RMAT>::multiply_accumulate(*this, B, acc);
      }



      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(element_type val, int row, int col){
        m_registers[register_map::to_register(row, col)].set(val, register_map::to_lane(row,col));
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(int row, int col) const {
        return m_registers[register_map::to_register(row, col)].get(register_map::to_lane(row,col));
      }





      /*!
       * @brief Converts to matrix to a string
       *
       *
       */
      RAJA_INLINE
      std::string to_string(bool one_line=false) const {
        std::string s = "Matrix(" + std::to_string(register_type::s_num_elem) +
            "x" + std::to_string(register_type::s_num_elem);
        if(!one_line){
          s +=")\n";
        }


        s += "[ ";

        //
        for(camp::idx_t r = 0;r < register_type::s_num_elem; ++ r){
          if(r > 0){
            s += ", ";
            if(!one_line){
              s+= "\n  ";
            }
          }
          s += "[";
          for(camp::idx_t c = 0;c < register_type::s_num_elem; ++ c){
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

  }; // MatrixRegisterImpl






}  // namespace RAJA




#endif
