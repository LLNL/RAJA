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

#ifndef RAJA_pattern_tensor_VectorRegisterImpl_HPP
#define RAJA_pattern_tensor_VectorRegisterImpl_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

#include "camp/camp.hpp"
#include "RAJA/pattern/tensor/internal/TensorRegisterBase.hpp"
#include "RAJA/pattern/tensor/stats.hpp"
#include "RAJA/util/BitMask.hpp"


namespace RAJA
{


  /*!
   * This provides a Tensor specialization for vectors
   */
  template<typename REGISTER_POLICY, typename T, camp::idx_t SIZE>
  class TensorRegister<REGISTER_POLICY, T, VectorLayout, camp::idx_seq<SIZE>> :
    public internal::TensorRegisterBase<TensorRegister<REGISTER_POLICY, T, VectorLayout, camp::idx_seq<SIZE>>>
  {
    public:
      using self_type = TensorRegister<REGISTER_POLICY, T, VectorLayout, camp::idx_seq<SIZE>>;
      using base_type = internal::TensorRegisterBase<TensorRegister<REGISTER_POLICY, T, VectorLayout, camp::idx_seq<SIZE>>>;
      using element_type = camp::decay<T>;
      using layout_type = TensorLayout<0>;
      using register_type = Register<T, REGISTER_POLICY>;

      static constexpr camp::idx_t s_num_elem = SIZE;

      using int_element_type = typename register_type::int_vector_type::element_type;
      using int_vector_type = TensorRegister<REGISTER_POLICY, int_element_type, VectorLayout, camp::idx_seq<SIZE>>;

    private:

      static constexpr camp::idx_t s_register_num_elem = register_type::s_num_elem;

      static constexpr camp::idx_t s_num_full_registers = s_num_elem/s_register_num_elem;

      static constexpr camp::idx_t s_num_partial_lanes =  s_num_elem%s_register_num_elem;

      static constexpr camp::idx_t s_num_registers =
          (s_num_partial_lanes > 0) ?
              s_num_full_registers + 1 :
              s_num_full_registers;

      using log_base2_t = RAJA::LogBase2<s_register_num_elem>;

      static constexpr camp::idx_t s_shift_per_register =
          log_base2_t::value;

      static constexpr camp::idx_t s_mask_per_register =
          (1<<log_base2_t::value)-1;


      template<typename IDX>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      static
      auto to_register(IDX i) -> IDX {
        return i >> IDX(s_shift_per_register);
      }

      template<typename IDX>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      static
      auto to_lane(IDX i) -> IDX {
        return i & IDX(s_mask_per_register);
      }


      using base_type::m_registers;

    public:



      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      TensorRegister(){}


      RAJA_HOST_DEVICE
      RAJA_INLINE
      TensorRegister(element_type c)
      {
        this->broadcast(c);
      }


      RAJA_INLINE
      RAJA_HOST_DEVICE
      TensorRegister(self_type const &c) :
        base_type(c)
      {
      }

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
        *this = rhs.eval(base_type::s_get_default_tile());
      }


      template<typename ... REGS>
      explicit
      RAJA_HOST_DEVICE
      RAJA_INLINE
      TensorRegister(register_type reg0, REGS const &... regs) :
        base_type(reg0, regs...)
      {
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return register_type::is_root();
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
        return STRIDE_ONE_DIM == 0;
      }


      /*!
       * Gets the maximum size of matrix along specified dimension
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr camp::idx_t s_dim_elem(camp::idx_t dim){
        return dim == 0 ? s_num_elem : 0;
      }


      /*!
       * @brief Set entire vector to a single scalar value
       * @param value Value to set all vector elements to
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(element_type value)
      {
        this->broadcast(value);
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(self_type const &c){
        return this->copy(c);
      }

      /*!
       * Provide left vector-matrix multiply for operator* between
       * this vector and a matrix
       */
      template<typename T2, typename L, typename RP>
      self_type
      operator*(SquareMatrixRegister<T2, L, RP> const &y) const
      {
        return y.left_vector_multiply(*this);
      }


      /*!
       * @brief Performs load specified by TensorRef object.
       */
      template<typename POINTER_TYPE, typename INDEX_TYPE, internal::TensorTileSize TENSOR_SIZE, camp::idx_t STRIDE_ONE_DIM>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_ref(internal::TensorRef<POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE, 1, STRIDE_ONE_DIM> const &ref){

        auto ptr = ref.m_pointer + ref.m_tile.m_begin[0]*ref.m_stride[0];

        // check for packed data
        if(STRIDE_ONE_DIM == 0){
          // full vector?
          if(TENSOR_SIZE == internal::TENSOR_FULL){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_packed ++;
#endif
            load_packed(ptr);
          }
          // partial
          else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_packed_n ++;
#endif
            load_packed_n(ptr, ref.m_tile.m_size[0]);
          }

        }
        // strided data
        else
        {
          // full vector?
          if(TENSOR_SIZE == internal::TENSOR_FULL){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided ++;
#endif
            load_strided(ptr, ref.m_stride[0]);
          }
          // partial
          else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_load_strided_n ++;
#endif
            load_strided_n(ptr, ref.m_stride[0], ref.m_tile.m_size[0]);
          }
        }
        return *this;
      }


      /*!
       * @brief Performs load specified by TensorRef object.
       */
      template<typename POINTER_TYPE, typename INDEX_TYPE, internal::TensorTileSize TENSOR_SIZE, camp::idx_t STRIDE_ONE_DIM>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_ref(internal::TensorRef<POINTER_TYPE, INDEX_TYPE, TENSOR_SIZE, 1, STRIDE_ONE_DIM> const &ref) const {

        auto ptr = ref.m_pointer + ref.m_tile.m_begin[0]*ref.m_stride[0];

        // check for packed data
        if(STRIDE_ONE_DIM == 0){
          // full vector?
          if(TENSOR_SIZE == internal::TENSOR_FULL){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_packed ++;
#endif
            store_packed(ptr);
          }
          // partial
          else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_packed_n ++;
#endif
            store_packed_n(ptr, ref.m_tile.m_size[0]);
          }

        }
        // strided data
        else
        {
          // full vector?
          if(TENSOR_SIZE == internal::TENSOR_FULL){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_strided ++;
#endif
            store_strided(ptr, ref.m_stride[0]);
          }
          // partial
          else{
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::tensor_stats::num_vector_store_strided_n ++;
#endif
            store_strided_n(ptr, ref.m_stride[0], ref.m_tile.m_size[0]);
          }
        }
        return *this;
      }


      /*!
       * Loads a dense full vector from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_packed(element_type const *ptr)
      {
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          m_registers[reg].load_packed(ptr+reg*s_register_num_elem);
        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].load_packed_n(ptr+s_num_full_registers*s_register_num_elem, s_num_partial_lanes);
        }
        return *this;
      }

      /*!
       * Loads a strided full vector from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_strided(element_type const *ptr, int stride)
      {
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          m_registers[reg].load_strided(ptr+reg*s_register_num_elem*stride, stride);
        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].load_strided_n(ptr+s_num_full_registers*s_register_num_elem*stride, stride, s_num_partial_lanes);
        }
        return *this;
      }

      /*!
       * Loads a dense partial vector from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_packed_n(element_type const *ptr, int N)
      {
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          if(N >= reg*s_register_num_elem + s_register_num_elem){
            m_registers[reg].load_packed(ptr+reg*s_register_num_elem);
          }
          else{
            m_registers[reg].load_packed_n(ptr+reg*s_register_num_elem,
                                           N-reg*s_register_num_elem);

            for(camp::idx_t r = reg+1;r < s_num_full_registers;++ r){
              m_registers[r].broadcast(0);
            }
            return *this;
          }

        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].load_packed_n(
              ptr+s_num_full_registers*s_register_num_elem,
              N-s_num_full_registers*s_register_num_elem);
        }
        return *this;
      }

      /*!
       * Loads a strided partial vector from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_strided_n(element_type const *ptr,
          int stride, int N)
      {
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          if(N >= reg*s_register_num_elem + s_register_num_elem){
            m_registers[reg].load_strided(ptr+reg*s_register_num_elem*stride, stride);
          }
          else{
            m_registers[reg].load_strided_n(ptr+reg*s_register_num_elem*stride,
                                            stride,
                                            N-reg*s_register_num_elem);
            for(camp::idx_t r = reg+1;r < s_num_full_registers;++ r){
              m_registers[r].broadcast(0);
            }
            return *this;
          }

        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].load_strided_n(
              ptr+s_num_full_registers*s_register_num_elem*stride,
              stride,
              N-s_num_full_registers*s_register_num_elem);
        }
        return *this;
      }


      /*!
       * @brief Generic gather operation for full vector.
       *
       * Must provide another register containing offsets of all values
       * to be loaded relative to supplied pointer.
       *
       * Offsets are element-wise, not byte-wise.
       *
       */
      RAJA_INLINE
      self_type &gather(element_type const *ptr, int_vector_type offsets){
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          m_registers[reg].gather(ptr, offsets.vec(reg));
        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].gather_n(ptr, offsets.vec(s_num_full_registers), s_num_partial_lanes);
        }
        return *this;
      }

      /*!
       * @brief Generic gather operation for n-length subvector.
       *
       * Must provide another register containing offsets of all values
       * to be loaded relative to supplied pointer.
       *
       * Offsets are element-wise, not byte-wise.
       *
       */
      RAJA_INLINE
      self_type &gather_n(element_type const *ptr, int_vector_type offsets, camp::idx_t N){
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          if(N >= reg*s_register_num_elem + s_register_num_elem){
            m_registers[reg].gather(ptr, offsets.vec(reg));
          }
          else{
            m_registers[reg].gather_n(ptr, offsets.vec(reg), N-reg*s_register_num_elem);
            for(camp::idx_t r = reg+1;r < s_num_full_registers;++ r){
              m_registers[r].broadcast(0);
            }
            return *this;
          }

        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].gather_n(
              ptr,
              offsets.vec(s_num_full_registers),
              N-s_num_full_registers*s_register_num_elem);
        }
        return *this;
      }


      /*!
       * Loads a dense full vector from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_packed(element_type *ptr) const
      {
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          m_registers[reg].store_packed(ptr+reg*s_register_num_elem);
        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].store_packed_n(ptr+s_num_full_registers*s_register_num_elem, s_num_partial_lanes);
        }
        return *this;
      }

      /*!
       * Loads a strided full vector from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_strided(element_type *ptr, int stride) const
      {
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          m_registers[reg].store_strided(ptr+reg*s_register_num_elem*stride, stride);
        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].store_strided_n(ptr+s_num_full_registers*s_register_num_elem*stride, stride, s_num_partial_lanes);
        }
        return *this;
      }

      /*!
       * Loads a dense partial vector from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_packed_n(element_type *ptr, int N) const
      {
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          if(N >= reg*s_register_num_elem + s_register_num_elem){
            m_registers[reg].store_packed(ptr+reg*s_register_num_elem);
          }
          else{
            m_registers[reg].store_packed_n(ptr+reg*s_register_num_elem,
                                           N-reg*s_register_num_elem);
            return *this;
          }

        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].store_packed_n(
              ptr+s_num_full_registers*s_register_num_elem,
              N-s_num_full_registers*s_register_num_elem);
        }
        return *this;
      }

      /*!
       * Loads a strided partial vector from memory
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_strided_n(element_type  *ptr,
          int stride, int N) const
      {
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          if(N >= reg*s_register_num_elem + s_register_num_elem){
            m_registers[reg].store_strided(ptr+reg*s_register_num_elem*stride, stride);
          }
          else{
            m_registers[reg].store_strided_n(ptr+reg*s_register_num_elem*stride,
                                            stride,
                                            N-reg*s_register_num_elem);
            return *this;
          }

        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].store_strided_n(
              ptr+s_num_full_registers*s_register_num_elem*stride,
              stride,
              N-s_num_full_registers*s_register_num_elem);
        }
        return *this;
      }



      /*!
       * @brief Generic scatter operation for full vector.
       *
       * Must provide another register containing offsets of all values
       * to be stored relative to supplied pointer.
       *
       * Offsets are element-wise, not byte-wise.
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &scatter(element_type *ptr, int_vector_type const &offsets) const {
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          m_registers[reg].scatter(ptr, offsets.vec(reg));
        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].scatter_n(ptr, offsets.vec(s_num_full_registers), s_num_partial_lanes);
        }
        return *this;
      }

      /*!
       * @brief Generic scatter operation for n-length subvector.
       *
       * Must provide another register containing offsets of all values
       * to be stored relative to supplied pointer.
       *
       * Offsets are element-wise, not byte-wise.
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &scatter_n(element_type *ptr, int_vector_type const &offsets, camp::idx_t N) const {
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          if(N >= reg*s_register_num_elem + s_register_num_elem){
            m_registers[reg].scatter(ptr, offsets.vec(reg));
          }
          else{
            m_registers[reg].scatter_n(ptr, offsets.vec(reg), N-reg*s_register_num_elem);

            return *this;
          }

        }
        if(s_num_partial_lanes){
          m_registers[s_num_full_registers].scatter_n(
              ptr,
              offsets.vec(s_num_full_registers),
              N-s_num_full_registers*s_register_num_elem);
        }
        return *this;
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &den) const {
        self_type result;
        for(camp::idx_t reg = 0;reg < s_num_full_registers;++ reg){
          result.vec(reg) = m_registers[reg].divide(den.vec(reg));
        }
        if(s_num_partial_lanes){
          result.vec(s_num_full_registers) = m_registers[s_num_full_registers].divide_n(den.vec(s_num_full_registers), s_num_partial_lanes);
        }
        return result;
      }

      /*!
       * @brief Divide n elements of this vector by another vector
       * @param x Vector to divide by
       * @param n Number of elements to divide
       * @return Value of (*this)+x
       */
      RAJA_SUPPRESS_HD_WARN
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide_n(self_type const &b, camp::idx_t n) const {
        self_type q(*this);
        for(camp::idx_t i = 0;i < n;++i){
          q.set(this->get(i) / b.get(i), i);
        }
        return q;
      }

      /*!
       * @brief Divide n elements of this vector by a scalar
       * @param x Scalar to divide by
       * @param n Number of elements to divide
       * @return Value of (*this)+x
       */
      RAJA_SUPPRESS_HD_WARN
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide_n(element_type const &b, camp::idx_t n) const {
        self_type q(*this);
        for(camp::idx_t i = 0;i < n;++i){
          q.set(this->get(i) / b, i);
        }
        return q;
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type min() const
      {
        // special case where there's just one parital register
        if(s_num_full_registers == 0){
          return m_registers[0].min_n(s_num_partial_lanes);
        }

        element_type result = m_registers[0].min();
        for(camp::idx_t i = 1;i < s_num_full_registers;++ i){
          result = RAJA::min<element_type>(result, m_registers[i].min());
        }
        if(s_num_partial_lanes){
          result = RAJA::min<element_type>(result, m_registers[s_num_full_registers].min_n(s_num_partial_lanes));
        }
        return result;
      }

      /*!
       * @brief Returns the smallest element over the first N lanes
       */
      RAJA_INLINE
      element_type min_n(int N) const
      {
        // special case where there's just one parital register
        if(N < s_register_num_elem){
          return m_registers[0].min_n(N);
        }

        element_type result = m_registers[0].min();
        for(camp::idx_t reg = 1;reg < s_num_full_registers;++ reg){
          if(N >= reg*s_register_num_elem + s_register_num_elem){
            result = RAJA::min<element_type>(result, m_registers[reg].min());
          }
          else{
            return RAJA::min<element_type>(result, m_registers[reg].min_n(N-reg*s_register_num_elem));
          }
        }
        if(N-s_num_full_registers*s_register_num_elem > 0){
          result = RAJA::min<element_type>(result, m_registers[s_num_full_registers].min_n(N-s_num_full_registers*s_register_num_elem));
        }
        return result;
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_INLINE
      element_type max() const
      {
        // special case where there's just one parital register
        if(s_num_full_registers == 0){
          return m_registers[0].max_n(s_num_partial_lanes);
        }

        element_type result = m_registers[0].max();
        for(camp::idx_t i = 1;i < s_num_full_registers;++ i){
          result = RAJA::max<element_type>(result, m_registers[i].max());
        }
        if(s_num_partial_lanes){
          result = RAJA::max<element_type>(result, m_registers[s_num_full_registers].max_n(s_num_partial_lanes));
        }
        return result;
      }

      /*!
       * @brief Returns the largest element over the first N lanes
       */
      RAJA_INLINE
      element_type max_n(int N) const
      {
        // special case where there's just one parital register
        if(N < s_register_num_elem){
          return m_registers[0].max_n(N);
        }

        element_type result = m_registers[0].max();
        for(camp::idx_t reg = 1;reg < s_num_full_registers;++ reg){
          if(N >= reg*s_register_num_elem + s_register_num_elem){
            result = RAJA::max<element_type>(result, m_registers[reg].max());
          }
          else{
            return RAJA::max<element_type>(result, m_registers[reg].max_n(N-reg*s_register_num_elem));
          }
        }
        if(N-s_num_full_registers*s_register_num_elem > 0){
          result = RAJA::max<element_type>(result, m_registers[s_num_full_registers].max_n(N-s_num_full_registers*s_register_num_elem));
        }
        return result;
      }

      /*!
       * @brief Returns the sum of all elements
       */
      RAJA_INLINE
      element_type sum() const
      {
        // first do a vector sum of all registers
        register_type s = m_registers[0];
        for(camp::idx_t i = 1;i < s_num_registers;++ i){
          s += m_registers[i];
        }
        // then a horizontal sum of result
        return s.sum();
      }


      /*!
       * @brief The * operator of two vectors is a element-wise multiply
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator*(self_type const &x) const {
        return this->multiply(x);
      }


      /*!
       * Provides vector-level building block for matrix transpose operations.
       *
       * This is a non-optimized reference version which will be used if
       * no architecture specialized version is supplied
       *
       * This is a permute-and-shuffle left operation
       *
       *           X=   x0  x1  x2  x3  x4  x5  x6  x7...
       *           Y=   y0  y1  y2  y3  y4  y5  y6  y7...
       *
       *  lvl=0    Z=   x0  y0  x2  y2  x4  y4  x6  y6...
       *  lvl=1    Z=   x0  x1  y0  y1  x4  x5  y4  y5...
       *  lvl=2    Z=   x0  x1  x2  x3  y0  y1  y2  y3...
       */
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      self_type transpose_shuffle_left(camp::idx_t lvl, self_type const &y) const
//      {
//        auto const &x = *this;
//
//        self_type z;
//
//        for(camp::idx_t i = 0;i < s_num_elem;++ i){
//
//          // extract value x or y
//          camp::idx_t xy_select = (i >> lvl) & 0x1;
//
//
//          z.set(xy_select == 0 ? x.get(i) : y.get(i - (1<<lvl)), i);
//        }
//
//        return z;
//      }


      /*!
       * Provides vector-level building block for matrix transpose operations.
       *
       * This is a non-optimized reference version which will be used if
       * no architecture specialized version is supplied
       *
       * This is a permute-and-shuffle right operation
       *
       *           X=   x0  x1  x2  x3  x4  x5  x6  x7...
       *           Y=   y0  y1  y2  y3  y4  y5  y6  y7...
       *
       *  lvl=0    Z=   x1  y1  x3  y3  x5  y5  x7  y7...
       *  lvl=1    Z=   x2  x3  y2  y3  x6  x7  y6  y7...
       *  lvl=2    Z=   x4  x5  x6  x7  y4  y5  y6  y7...
       */
//      RAJA_INLINE
//      RAJA_HOST_DEVICE
//      self_type transpose_shuffle_right(int lvl, self_type const &y) const
//      {
//        auto const &x = *this;
//
//        self_type z;
//
//        camp::idx_t i0 = 1<<lvl;
//
//        for(camp::idx_t i = 0;i < s_num_elem;++ i){
//
//          // extract value x or y
//          camp::idx_t xy_select = (i >> lvl) & 0x1;
//
//          z.set(xy_select == 0 ? x.get(i0 + i) : y.get(i0 + i - (1<<lvl)), i);
//        }
//
//        return z;
//      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(element_type val, int idx){
        m_registers[to_register(idx)].set(val, to_lane(idx));
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(int idx) const {
        return m_registers[to_register(idx)].get(to_lane(idx));
      }



      /*!
       * @brief Converts to vector to a string
       *
       *
       */
      RAJA_INLINE
      std::string to_string() const {
        std::string s = "Vector(" + std::to_string(s_num_elem) + ")[ ";

        //
        for(camp::idx_t i = 0;i < s_num_elem; ++ i){
          s += std::to_string(this->get(i)) + " ";
        }

        camp::idx_t physical_size = s_num_registers * s_register_num_elem;
        if(s_num_elem < physical_size){
          s += "{";
          for(camp::idx_t i = s_num_elem;i < physical_size; ++ i){
            s += std::to_string(this->get(i)) + " ";
          }
          s += "}";
        }


        s += " ]\n";

        return s;
      }


  };



}  // namespace RAJA


// Bring in the register policy file so we get the default register type
// and all of the register traits setup
#include "RAJA/policy/tensor/arch.hpp"


#endif
