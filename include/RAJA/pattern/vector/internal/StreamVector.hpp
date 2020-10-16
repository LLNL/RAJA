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

#ifndef RAJA_pattern_vector_streamvector_HPP
#define RAJA_pattern_vector_streamvector_HPP

#include "RAJA/pattern/vector/internal/VectorBase.hpp"


namespace RAJA
{
  namespace internal
  {

    // Variable Length Vector Implementation
    template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t ... REG_SEQ, camp::idx_t ... PART_REG_SEQ, camp::idx_t NUM_ELEM>
    class VectorImpl<VECTOR_STREAM, REGISTER_POLICY, ELEMENT_TYPE, camp::idx_seq<REG_SEQ...>, camp::idx_seq<PART_REG_SEQ...>, NUM_ELEM> :
      public VectorBase<VectorImpl<VECTOR_STREAM, REGISTER_POLICY, ELEMENT_TYPE, camp::idx_seq<REG_SEQ...>, camp::idx_seq<PART_REG_SEQ...>, NUM_ELEM>>
    {
      public:
        using self_type = VectorImpl<VECTOR_STREAM, REGISTER_POLICY, ELEMENT_TYPE, camp::idx_seq<REG_SEQ...>, camp::idx_seq<PART_REG_SEQ...>, NUM_ELEM>;
        using base_type = VectorBase<self_type>;

        using base_type::s_num_elem;
        using base_type::s_num_reg_elem;
        using base_type::s_num_registers;
        using base_type::s_is_fixed;
        using base_type::s_num_partial_elem;


        using element_type = ELEMENT_TYPE;


      private:
        camp::idx_t m_length;

        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr
        camp::idx_t regNumElem(camp::idx_t reg) const {
          // How many elements of this register are there?
          return (1+reg)*s_num_reg_elem < m_length
            ? s_num_reg_elem                       // Full register
            : m_length-reg*s_num_reg_elem;  // Partial register
        }

      public:

        /*!
         * @brief Default constructor, zeros register contents
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorImpl() :
          base_type(),
          m_length(NUM_ELEM)
        {
        }

        /*!
         * @brief Copy constructor
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorImpl(self_type const &c) :
          base_type(c),
          m_length(c.m_length)
        {
        }

        /*!
         * @brief Copy construct given size of other vector
         *
         * For StreamVector, vector length is copied, and the registers
         * are uninitialized
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorImpl(self_type const &c, InitSizeOnlyTag const &) :
          base_type(),
          m_length(c.m_length)
        {
        }

        /*!
         * @brief Copy assignment constructor
         *
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &operator=(self_type const &c)
        {
          return copy(c);
        }

        /*!
         * @brief Scalar constructor (broadcast)
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorImpl(element_type const &c) :
          base_type(c),
          m_length(NUM_ELEM)
        {
        }


        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& load_packed(element_type const *ptr){

          m_length = NUM_ELEM;

          return base_type::load_packed(ptr);

        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& load_packed_n(element_type const *ptr, camp::idx_t N){

          m_length = N;

          return base_type::load_packed_n(ptr, N);

        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& load_strided(element_type const *ptr, camp::idx_t stride){

          m_length = NUM_ELEM;

          return base_type::load_strided(ptr, stride);

        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& load_strided_n(element_type const *ptr, camp::idx_t stride, camp::idx_t N){

          m_length = N;

          return base_type::load_strided_n(ptr, stride,N);

        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type const& store_packed(element_type *ptr) const {

          if(m_length == NUM_ELEM){
            return base_type::store_packed(ptr);
          }

          return base_type::store_packed_n(ptr, m_length);
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type const& store_strided(element_type *ptr, camp::idx_t stride) const {

          if(m_length == NUM_ELEM){
            return base_type::store_strided(ptr, stride);
          }


          return base_type::store_strided_n(ptr, stride, m_length);
        }

        /*! @brief Returns vector size, this is dynamically sized from
         *         zero up to NUM_ELEM.
         *
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr
        camp::idx_t size() const
        {
          return m_length;
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &broadcast(element_type const &value){

          m_length = NUM_ELEM;

          return base_type::broadcast(value);
        }


        /*!
         * @brief Copy values of another vector
         * @param x The other vector to copy
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type  &copy(self_type const &x){
          m_length = x.m_length;
          return base_type::copy(x);
        }

        /*!
         * @brief Returns the sum of all elements in the vector
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type sum(camp::idx_t = NUM_ELEM) const {
          return foldl_sum<element_type>(base_type::m_registers[REG_SEQ].sum(regNumElem(REG_SEQ))...);
        }

        /*!
         * @brief Returns the maximum value of all elements in the vector
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type max(camp::idx_t = NUM_ELEM) const {
          return foldl_max<element_type>(base_type::m_registers[REG_SEQ].max(regNumElem(REG_SEQ))...);
        }


        /*!
         * @brief Returns the minimum value of all elements in the vector
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type min(camp::idx_t = NUM_ELEM) const {
          return foldl_min<element_type>(base_type::m_registers[REG_SEQ].min(regNumElem(REG_SEQ))...);
        }


        /*!
         * @brief Element-wise division of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type divide(self_type const &b) const {

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_divide ++;
#endif

          self_type result(b, InitSizeOnlyTag{});

          camp::sink( (result.m_registers[REG_SEQ] = base_type::m_registers[REG_SEQ].divide(b.m_registers[REG_SEQ], regNumElem(REG_SEQ)))...);

          return result;
        }


    };


  } //namespace internal
}  // namespace RAJA


#endif
