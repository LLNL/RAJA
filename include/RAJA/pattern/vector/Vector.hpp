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

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VECTOR_LENGTH_TYPE VECTOR_TYPE>
  class Vector
  {
    private:
      using type_helper_t = internal::detail::VectorTypeHelper<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>;

    public:
      using self_type = Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>;
      using vector_type = self_type;
      using register_policy = REGISTER_POLICY;
      using element_type = ELEMENT_TYPE;


      using register_types_t = typename type_helper_t::register_list_t;
      using register_tuple_t = typename type_helper_t::register_tuple_t;

      using register0_type = camp::at_v<register_types_t, 0>;

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static constexpr camp::idx_t num_elem(camp::idx_t = 0){
        return NUM_ELEM;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      static constexpr camp::idx_t num_registers(){
        return camp::tuple_size<register_tuple_t>::value;
      }


      //static constexpr bool s_is_fixed = (VECTOR_TYPE == VECTOR_FIXED);


    private:

      using op_helper_t = internal::detail::VectorOpHelper<register_tuple_t, camp::make_idx_seq_t<camp::tuple_size<register_tuple_t>::value>, NUM_ELEM, VECTOR_TYPE>;

      register_tuple_t m_registers;

      camp::idx_t m_length;

    public:


      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Vector() : m_length(num_elem()){}

      /*!
       * @brief Copy constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Vector(self_type const &c) :
        m_registers(c.m_registers),
        m_length(c.m_length)
      {
      }

      /*!
       * @brief Scalar constructor (broadcast)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      Vector(element_type const &c) :
        m_length(num_elem())
      {
        broadcast(c);
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return register0_type::is_root();
      }



      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &load(element_type const *ptr, camp::idx_t stride = 1, camp::idx_t length = NUM_ELEM){

        m_length = length;

        op_helper_t::load(m_registers, ptr, stride, length);

        return *this;
      }


      /*!
       * @brief Strided store operation, where scalars are stored in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       *
       * Note: this could be done with "scatter" instructions if they are
       * available.
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type const &store(element_type *ptr, camp::idx_t stride = 1) const{

        op_helper_t::store(m_registers, ptr, stride, m_length);

        return *this;
      }

      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      camp::idx_t size() const
      {
        return m_length;
      }

      /*!
       * @brief Get scalar value from vector
       * This will not be the most efficient due to the offset calculation.
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type operator[](camp::idx_t i) const
      {
        return get(i);
      }

      /*!
       * @brief Get scalar value from vector
       * This will not be the most efficient due to the offset calculation.
       * @param i Offset of scalar to get
       * @return Returns scalar value at i
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type get(camp::idx_t i) const
      {
        return op_helper_t::get(m_registers, i);
      }


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &set(camp::idx_t i, element_type value)
      {
        op_helper_t::set(m_registers, i, value);
        return *this;
      }


      /*!
       * @brief assign all vector values to same scalar value
       * @param value The scalar value to use
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &broadcast(element_type const &value){

        op_helper_t::broadcast(m_registers, value);

        m_length = num_elem();

        return *this;
      }


      /*!
       * @brief Copy values of another vector
       * @param x The other vector to copy
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type  &copy(self_type const &x){
        m_registers = x.m_registers;
        m_length = x.m_length;
        return *this;
      }

      /*!
       * @brief Element-wise addition of two vectors
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(self_type const &x) const {
        self_type result;

        op_helper_t::add(m_registers, x.m_registers, result.m_registers);

        result.m_length = RAJA::min(m_length, x.m_length);

        return result;
      }

      /*!
       * @brief Element-wise subtraction of two vectors
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &x) const {
        self_type result;

        op_helper_t::subtract(m_registers, x.m_registers, result.m_registers);

        result.m_length = RAJA::min(m_length, x.m_length);

        return result;
      }

      /*!
       * @brief Element-wise multiplication of two vectors
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &x) const {
        self_type result;

        op_helper_t::multiply(m_registers, x.m_registers, result.m_registers);

        result.m_length = RAJA::min(m_length, x.m_length);

        return result;
      }

      /*!
       * @brief Element-wise division of two vectors
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &x) const {
        self_type result;

        op_helper_t::divide(m_registers, x.m_registers, result.m_registers);

        result.m_length = RAJA::min(m_length, x.m_length);

        return result;
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

      /*!
       * @brief Assign one register to antoher
       * @param x Vector to copy
       * @return Value of (*this)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator=(self_type const &x)
      {
        copy(x);

        return *this;
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
        return add(x);
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
        *this = add(x);
        return *this;
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
        return subtract(x);
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
        *this = subtract(x);
        return *this;
      }

      /*!
       * @brief Multiply two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      VectorProductRef<self_type> operator*(self_type const &x) const
      {
        return VectorProductRef<self_type>(*this, x);
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
        (*this) = multiply(x);
        return *this;
      }

      /*!
       * @brief Divide two vector registers, element wise
       * @param x Vector to subctract from this register
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type operator/(self_type const &x) const
      {
        return divide(x);
      }

      /*!
       * @brief Divide this vector by another vector
       * @param x Vector to divide by
       * @return Value of (*this)+x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type &operator/=(self_type const &x)
      {
        (*this) = divide(x);
        return *this;
      }


      /**
        * @brief Fused multiply add: fma(b, c) = (*this)*b+c
        *
        * Derived types can override this to implement intrinsic FMA's
        *
        * @param b Second product operand
        * @param c Sum operand
        * @return Value of (*this)*b+c
        */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type fused_multiply_add(self_type const &b, self_type const &c) const
      {
        self_type result;

        op_helper_t::fused_multiply_add(m_registers, b.m_registers, c.m_registers, result.m_registers);

        result.m_length = RAJA::foldl_min<camp::idx_t>(m_length, b.m_length, c.m_length);

        return result;
      }

      /**
        * @brief Fused multiply subtract: fms(b, c) = (*this)*b-c
        *
        * Derived types can override this to implement intrinsic FMA's
        *
        * @param b Second product operand
        * @param c Subtraction operand
        * @return Value of (*this)*b-c
        */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type fused_multiply_subtract(self_type const &b, self_type const &c) const
      {
        self_type result;

        op_helper_t::fused_multiply_subtract(m_registers, b.m_registers, c.m_registers, result.m_registers);

        result.m_length = RAJA::foldl_min<camp::idx_t>(m_length, b.m_length, c.m_length);

        return result;
      }


      /*!
       * @brief Sum the elements of this vector
       * @return Sum of the values of the vectors scalar elements
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type sum() const
      {
        return op_helper_t::sum(m_registers, m_length);
      }

      /*!
       * @brief Dot product of two vectors
       * @param x Other vector to dot with this vector
       * @return Value of (*this) dot x
       *
       * NOTE: we could really do something more optimized here!
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type dot(self_type const &x) const
      {
        return multiply(x).sum();
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type max() const
      {
        return op_helper_t::max(m_registers, m_length);
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type min() const
      {
        return op_helper_t::min(m_registers, m_length);
      }


      /*!
       * @brief Returns element-wise max of two vectors
       * @return The vector containing max values of this and vector x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type vmax(self_type const &x) const
      {
        self_type result;

        op_helper_t::vmax(m_registers, x.m_registers, result.m_registers);

        result.m_length = RAJA::min(m_length, x.m_length);

        return result;
      }

      /*!
       * @brief Returns element-wise minimum of two vectors
       * @return The vector containing minimum values of this and vector x
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type vmin(self_type const &x) const
      {
        self_type result;

        op_helper_t::vmin(m_registers, x.m_registers, result.m_registers);

        result.m_length = RAJA::min(m_length, x.m_length);

        return result;
      }

  };

  //
  // Operator Overloads for scalar OP vector
  //


  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VECTOR_LENGTH_TYPE VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>
  operator+(typename Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>::element_type x, Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE> const &y){
    return Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>(x) + y;
  }

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VECTOR_LENGTH_TYPE VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>
  operator-(typename Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>::element_type x, Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE> const &y){
    return Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>(x) - y;
  }

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VECTOR_LENGTH_TYPE VECTOR_TYPE>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>
  operator*(typename Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>::element_type x, Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE> const &y){
    return Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>(x) * y;
  }

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VECTOR_LENGTH_TYPE VECTOR_TYPE>
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
                           UNROLL*RAJA::RegisterTraits<REGISTER_POLICY, T>::num_elem(),
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
