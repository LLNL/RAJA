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

#ifndef RAJA_pattern_vector_vector_HPP
#define RAJA_pattern_vector_vector_HPP

#include "RAJA/pattern/vector/internal/VectorImpl.hpp"

namespace RAJA
{

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
  class Vector : public internal::makeVectorBase<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>
  {
    public:
      using self_type = Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>;
      using base_type = internal::makeVectorBase<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>;

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Vector(){}

      /*!
       * @brief Copy constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Vector(self_type const &c) : base_type(c){}

      /*!
       * @brief Scalar constructor (broadcast)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Vector(ELEMENT_TYPE const &c) :base_type(c){}

  };

  //
  // Operator Overloads for scalar OP vector
  //


  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>
  operator+(typename Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>::element_type x, Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE> const &y){
    return Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>(x) + y;
  }

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>
  operator-(typename Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>::element_type x, Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE> const &y){
    return Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>(x) - y;
  }

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>
  operator*(typename Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>::element_type x, Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE> const &y){
    return Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>(x) * y;
  }

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>
  operator/(typename Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>::element_type x, Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE> const &y){
    return Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>(x) / y;
  }





  template<typename T, camp::idx_t UNROLL = 1, typename REGISTER_POLICY = policy::register_default>
  using StreamVector =
      Vector<REGISTER_POLICY,
                           T,
                           UNROLL*RAJA::Register<REGISTER_POLICY, T>::s_num_elem,
                           VECTOR_STREAM>;

  template<typename T, camp::idx_t NUM_ELEM, typename REGISTER_POLICY = policy::register_default>
  using FixedVector =
      Vector<REGISTER_POLICY,
                           T,
                           NUM_ELEM,
                           VECTOR_FIXED>;


  template<typename VECTOR_TYPE, camp::idx_t NUM_ELEM>
  using changeVectorLength = typename internal::VectorNewLengthHelper<VECTOR_TYPE, NUM_ELEM>::type;

}  // namespace RAJA


#endif
