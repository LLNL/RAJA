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

#ifndef RAJA_policy_vector_register_scalar_HPP
#define RAJA_policy_vector_register_scalar_HPP

#include "RAJA/pattern/tensor/internal/VectorRegisterBase.hpp"

namespace RAJA
{



  template<typename T>
  struct RegisterTraits<scalar_register, T>{
      using element_type = T;
      using register_policy = scalar_register;
      static constexpr camp::idx_t s_num_bits = sizeof(T)*8;
      static constexpr camp::idx_t s_num_elem = 1;
  };

  /**
   * A specialization for a single element register.
   * We will implement this as a scalar value, and let the compiler use
   * whatever registers it deems appropriate.
   */
  template<typename T>
  class TensorRegister<scalar_register, T, VectorLayout, camp::idx_seq<1>, camp::idx_seq<0>> :
      public internal::VectorRegisterBase<TensorRegister<scalar_register, T, VectorLayout, camp::idx_seq<1>, camp::idx_seq<0>>>
  {
    public:
      using register_policy = scalar_register;
      using self_type = TensorRegister<scalar_register, T, VectorLayout, camp::idx_seq<1>, camp::idx_seq<0>>;
      using element_type = T;
      using register_type = T;



    private:
      T m_value;

    public:

      static constexpr camp::idx_t s_num_elem = 1;

      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      TensorRegister() : m_value(0) {
      }

      /*!
       * @brief Copy constructor from underlying simd register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      TensorRegister(element_type const &c) : m_value(c) {}


      /*!
       * @brief Copy constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      TensorRegister(self_type const &c) : m_value(c.m_value) {}


      /*!
       * @brief Copy assignment constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(self_type const &c){
        m_value = c.m_value;
        return *this;
      }


      /*!
       * @brief Load a full register from a stride-one memory location
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_packed(element_type const *ptr){
        m_value = ptr[0];
        return *this;
      }

      /*!
       * @brief Partially load a register from a stride-one memory location given
       *        a run-time number of elements.
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_packed_n(element_type const *ptr, camp::idx_t N){
        if(N > 0){
          m_value = ptr[0];
        }
        else{
          m_value = element_type(0);
        }
        return *this;
      }

      /*!
       * @brief Gather a full register from a strided memory location
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_strided(element_type const *ptr, camp::idx_t ){
        m_value = ptr[0];
        return *this;
      }


      /*!
       * @brief Partially load a register from a stride-one memory location given
       *        a run-time number of elements.
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load_strided_n(element_type const *ptr, camp::idx_t , camp::idx_t N){
        if(N > 0){
          m_value = ptr[0];
        }
        else{
          m_value = element_type(0);
        }
        return *this;
      }


      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_packed(element_type *ptr) const{
        ptr[0] = m_value;
        return *this;
      }

      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_packed_n(element_type *ptr, camp::idx_t N) const{
        if(N > 0){
          ptr[0] = m_value;
        }
        return *this;
      }

      /*!
       * @brief Store entire register to consecutive memory locations
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_strided(element_type *ptr, camp::idx_t ) const{
        ptr[0] = m_value;
        return *this;
      }


      /*!
       * @brief Store partial register to consecutive memory locations
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store_strided_n(element_type *ptr, camp::idx_t , camp::idx_t N) const{
        if(N > 0){
          ptr[0] = m_value;
        }
        return *this;
      }



      /*!
       * @brief Get scalar value from vector register
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      constexpr
      RAJA_INLINE
      RAJA_HOST_DEVICE
      element_type get(camp::idx_t) const
      {return m_value;}


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type &set(element_type value, camp::idx_t)
      {
        m_value = value;
        return *this;
      }



      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type const &a){
        m_value = a;
        return *this;
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &copy(self_type const &src){
        m_value = src.m_value;
        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(self_type const &b) const {
        return self_type(m_value + b.m_value);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &b) const {
        return self_type(m_value - b.m_value);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &b) const {
        return self_type(m_value * b.m_value);
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &b) const {
        return self_type(m_value / b.m_value);
      }

      /*!
       * @brief Fused multiply add: fma(b, c) = (*this)*b+c
       *
       * Derived types can override this to implement intrinsic FMA's
       *
       * @param b Second product operand
       * @param c Sum operand
       * @return Value of (*this)*b+c
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type multiply_add(self_type const &b, self_type const &c) const
      {
        return m_value * b.m_value + c.m_value;
      }

      /*!
       * @brief Fused multiply subtract: fms(b, c) = (*this)*b-c
       *
       * Derived types can override this to implement intrinsic FMS's
       *
       * @param b Second product operand
       * @param c Subtraction operand
       * @return Value of (*this)*b-c
       */
      RAJA_INLINE
      RAJA_HOST_DEVICE
      self_type multiply_subtract(self_type const &b, self_type const &c) const
      {
        return m_value * b.m_value - c.m_value;
      }

      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      element_type sum(camp::idx_t = 1) const
      {
        return m_value;
      }


      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      element_type dot(self_type const &b) const
      {
        return m_value * b.m_value;
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      element_type max(camp::idx_t = 1) const
      {
        return m_value;
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type vmax(self_type a) const
      {
        return self_type(RAJA::max<element_type>(m_value, a.m_value));
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type min(camp::idx_t = 1) const
      {
        return m_value;
      }

      /*!
       * @brief Returns element-wise largest values
       * @return Vector of the element-wise max values
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type vmin(self_type a) const
      {
        return self_type(RAJA::min<element_type>(m_value, a.m_value));
      }



  };

}  // namespace RAJA


#endif
