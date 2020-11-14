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

#ifndef RAJA_pattern_vector_fixedvector_HPP
#define RAJA_pattern_vector_fixedvector_HPP

#include "RAJA/pattern/vector/internal/VectorBase.hpp"

namespace RAJA
{
  namespace internal
  {

    // Fixed Length Vector Implementation
    template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t ... REG_SEQ, camp::idx_t ... PART_REG_SEQ, int NUM_ELEM>
    class VectorImpl<VECTOR_FIXED, REGISTER_POLICY, ELEMENT_TYPE, camp::idx_seq<REG_SEQ...>, camp::idx_seq<PART_REG_SEQ...>, NUM_ELEM> :
      public VectorBase<VectorImpl<VECTOR_FIXED, REGISTER_POLICY, ELEMENT_TYPE, camp::idx_seq<REG_SEQ...>, camp::idx_seq<PART_REG_SEQ...>, NUM_ELEM>>
    {
      public:
        using self_type = VectorImpl<VECTOR_FIXED, REGISTER_POLICY, ELEMENT_TYPE, camp::idx_seq<REG_SEQ...>, camp::idx_seq<PART_REG_SEQ...>, NUM_ELEM>;
        using base_type = VectorBase<self_type>;

        using base_type::s_num_elem;
        using base_type::s_num_reg_elem;
        using base_type::s_num_registers;
        using base_type::s_is_fixed;
        using base_type::s_num_partial_elem;

        using element_type = ELEMENT_TYPE;

        using vector_type = self_type;
        using register_type = typename base_type::register_type;



      public:


        /*!
         * @brief Default constructor, zeros register contents
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorImpl() :
          base_type()
        {
        }

        /*!
         * @brief Copy constructor
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorImpl(self_type const &c) :
          base_type(c)
        {
        }

        /*!
         * @brief Copy construct given size of other vector
         *
         * For FixedVector, vector length is fixed so this is same as
         * the default constructor
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorImpl(self_type const &, InitSizeOnlyTag const &) :
          base_type()
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
          base_type(c)
        {
        }


        /*! @brief Returns vector size, this is statically defined as NUM_ELEM
         *
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr
        int size() const
        {
          return NUM_ELEM;
        }


        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &resize(int )
        {
          return *this;
        }


        /*!
         * @brief Copy values of another vector
         * @param x The other vector to copy
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &copy(self_type const &x){
          return base_type::copy(x);
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

          self_type result;

          camp::sink( (result.m_registers[REG_SEQ] = base_type::m_registers[REG_SEQ].divide(b.m_registers[REG_SEQ]))...);
          camp::sink( (result.m_registers[PART_REG_SEQ] = base_type::m_registers[PART_REG_SEQ].divide_n(b.m_registers[PART_REG_SEQ], s_num_partial_elem))...);

          return result;
        }

    };



  } //namespace internal
}  // namespace RAJA


#endif
