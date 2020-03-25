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

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/vector/VectorProductRef.hpp"

#include <array>

namespace RAJA
{

  namespace internal
  {

  namespace detail
  {

    class VectorRegisterAccessHelper{


    };

    template<typename ELEMENT_TYPE, typename IDX>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    ELEMENT_TYPE
    VectorGetByIndex(IDX){
      return ELEMENT_TYPE(); // termination case: this is undefined behavior!
    }

    template<typename ELEMENT_TYPE, typename IDX, typename REGISTER0, typename ... REGISTER_REST>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    ELEMENT_TYPE
    VectorGetByIndex(IDX i, REGISTER0 const &r0, REGISTER_REST const &...r_rest){
      if((camp::idx_t)i < (camp::idx_t)REGISTER0::s_num_elem){
        return r0.get(i);
      }
      else{
        return VectorGetByIndex<ELEMENT_TYPE>((camp::idx_t)i-(camp::idx_t)REGISTER0::s_num_elem, r_rest...);
      }
    }




    template<typename IDX, typename ELEMENT_TYPE>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    void
    VectorSetByIndex(IDX, ELEMENT_TYPE){
      // NOP: this is undefined behavior!
    }

    template<typename IDX, typename ELEMENT_TYPE, typename REGISTER0, typename ... REGISTER_REST>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    void
    VectorSetByIndex(IDX i, ELEMENT_TYPE value, REGISTER0 &r0, REGISTER_REST &...r_rest){
      if((camp::idx_t)i < (camp::idx_t)REGISTER0::s_num_elem){
        r0.set(i, value);
      }
      else{
        VectorSetByIndex((camp::idx_t)i-(camp::idx_t)REGISTER0::s_num_elem, value, r_rest...);
      }
    }




  } // namespace detail

/*!
 * \file
 * Vector operation functions in the namespace RAJA

 *
 */

  template<typename REGISTER_TUPLE, typename IDX_SEQ, bool FIXED_LENGTH>
  class VectorImpl;

  template<typename REGISTER0_TYPE, typename ... REGISTER_TYPES, camp::idx_t ... IDX_SEQ, bool FIXED_LENGTH>
  class VectorImpl<camp::list<REGISTER0_TYPE, REGISTER_TYPES...>, camp::idx_seq<IDX_SEQ...>, FIXED_LENGTH>
  {
    public:
      using register_types_t = camp::list<REGISTER0_TYPE, REGISTER_TYPES...>;
      using register_tuple_t = camp::tuple<REGISTER0_TYPE, REGISTER_TYPES...>;

      static constexpr camp::idx_t s_num_elem =
          RAJA::foldl_sum<camp::idx_t>(REGISTER0_TYPE::s_num_elem, REGISTER_TYPES::s_num_elem...);

      using self_type = VectorImpl<camp::list<REGISTER0_TYPE, REGISTER_TYPES...>, camp::idx_seq<IDX_SEQ...>, FIXED_LENGTH>;
      using vector_type = self_type;
      using element_type = typename REGISTER0_TYPE::element_type;


      static constexpr camp::idx_t s_is_fixed = FIXED_LENGTH;


    private:

      register_tuple_t m_registers;

      camp::idx_t m_length;

    public:


      /*!
       * @brief Default constructor, zeros register contents
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      VectorImpl() : m_length(s_num_elem){}

      /*!
       * @brief Copy constructor
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      VectorImpl(self_type const &c) :
        m_registers(c.m_registers),
        m_length(c.m_length)
      {
      }

      /*!
       * @brief Scalar constructor (broadcast)
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      VectorImpl(element_type const &c) :
        m_length(s_num_elem)
      {
        broadcast(c);
      }


      RAJA_HOST_DEVICE
      RAJA_INLINE
      static
      constexpr
      bool is_root() {
        return REGISTER0_TYPE::is_root();
      }



      /*!
       * @brief Strided load constructor, when scalars are located in memory
       * locations ptr, ptr+stride, ptr+2*stride, etc.
       *
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void load(element_type const *ptr, camp::idx_t stride = 1, camp::idx_t length = s_num_elem){
        m_length = length;
        if(s_is_fixed || m_length == s_num_elem){
          camp::sink(camp::get<IDX_SEQ>(m_registers).load(
              ptr + IDX_SEQ*stride*REGISTER0_TYPE::s_num_elem,
              stride
          )...);
        }
        else{
          for(camp::idx_t i = 0;i < length;++ i){
            set(i, ptr[i*stride]);
          }
        }
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
      void store(element_type *ptr, camp::idx_t stride = 1) const{
        if(s_is_fixed || m_length == s_num_elem){
          camp::sink(camp::get<IDX_SEQ>(m_registers).store(
              ptr + IDX_SEQ*stride*REGISTER0_TYPE::s_num_elem,
              stride
          )...);
        }
        else{
          for(camp::idx_t i = 0;i < m_length;++ i){
            ptr[i*stride] = (*this)[i];
          }
        }
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
        return detail::VectorGetByIndex<element_type>(i, camp::get<IDX_SEQ>(m_registers)...);
      }


      /*!
       * @brief Set scalar value in vector register
       * @param i Offset of scalar to set
       * @param value Value of scalar to set
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void set(camp::idx_t i, element_type value)
      {
        detail::VectorSetByIndex(i, value, camp::get<IDX_SEQ>(m_registers)...);
      }


      /*!
       * @brief assign all vector values to same scalar value
       * @param value The scalar value to use
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void broadcast(element_type const &value){
        camp::sink(camp::get<IDX_SEQ>(m_registers).broadcast(value)...);
        m_length = s_num_elem;
      }


      /*!
       * @brief Copy values of another vector
       * @param x The other vector to copy
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      void copy(self_type const &x){
        m_registers = x.m_registers;
        m_length = x.m_length;
      }

      /*!
       * @brief Element-wise addition of two vectors
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type add(self_type const &x) const {
        self_type result(*this);

        camp::sink((camp::get<IDX_SEQ>(result.m_registers) = camp::get<IDX_SEQ>(m_registers) + camp::get<IDX_SEQ>(x.m_registers))...);

        result.m_length = RAJA::min(m_length, x.m_length);

        return result;
      }

      /*!
       * @brief Element-wise subtraction of two vectors
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type subtract(self_type const &x) const {
        self_type result(*this);

        camp::sink((camp::get<IDX_SEQ>(result.m_registers) = camp::get<IDX_SEQ>(m_registers) - camp::get<IDX_SEQ>(x.m_registers))...);

        result.m_length = RAJA::min(m_length, x.m_length);

        return result;
      }

      /*!
       * @brief Element-wise multiplication of two vectors
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type multiply(self_type const &x) const {
        self_type result(*this);

        camp::sink((camp::get<IDX_SEQ>(result.m_registers) = camp::get<IDX_SEQ>(m_registers) * camp::get<IDX_SEQ>(x.m_registers))...);

        result.m_length = RAJA::min(m_length, x.m_length);

        return result;
      }

      /*!
       * @brief Element-wise division of two vectors
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      self_type divide(self_type const &x) const {
        self_type result(*this);

        camp::sink((camp::get<IDX_SEQ>(result.m_registers) = camp::get<IDX_SEQ>(m_registers) / camp::get<IDX_SEQ>(x.m_registers))...);

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

        camp::sink((camp::get<IDX_SEQ>(result.m_registers) =
            camp::get<IDX_SEQ>(m_registers).fused_multiply_add(
                camp::get<IDX_SEQ>(b.m_registers),
                camp::get<IDX_SEQ>(c.m_registers)))...);

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

        camp::sink((camp::get<IDX_SEQ>(result.m_registers) =
            camp::get<IDX_SEQ>(m_registers).fused_multiply_subtract(
                camp::get<IDX_SEQ>(b.m_registers),
                camp::get<IDX_SEQ>(c.m_registers)))...);

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
        if(m_length == s_num_elem){
          return RAJA::foldl_sum<element_type>(camp::get<IDX_SEQ>(m_registers).sum()...);
        }
        else{
          element_type result = (element_type)0;
          for(camp::idx_t i = 0;i < m_length;++ i){
            result += (*this)[i];
          }
          return result;
        }
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
        self_type z = (*this) * x;
        return z.sum();
      }


      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type max() const
      {
        if(s_is_fixed || m_length == s_num_elem){
          return RAJA::foldl_max<element_type>(camp::get<IDX_SEQ>(m_registers).max()...);
        }
        else{
          element_type result = (*this)[0];
          for(camp::idx_t i = 1;i < m_length;++ i){
            auto new_val = (*this)[i];
            result = result > new_val ? result : new_val;
          }
          return result;
        }
      }

      /*!
       * @brief Returns the largest element
       * @return The largest scalar element in the register
       */
      RAJA_HOST_DEVICE
      RAJA_INLINE
      element_type min() const
      {
        if(s_is_fixed || m_length == s_num_elem){
          return RAJA::foldl_min<element_type>(camp::get<IDX_SEQ>(m_registers).min()...);
        }
        else{
          element_type result = (*this)[0];
          for(camp::idx_t i = 1;i < m_length;++ i){
            auto new_val = (*this)[i];
            result = result < new_val ? result : new_val;
          }
          return result;
        }
      }

  };

  //
  // Operator Overloads for scalar OP vector
  //

  template<typename ST, typename REGISTER_TUPLE, typename IDX_SEQ, bool FIXED_LENGTH>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH>
  operator+(ST x, VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH> const &y){
    return VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH>(x) + y;
  }

  template<typename ST, typename REGISTER_TUPLE, typename IDX_SEQ, bool FIXED_LENGTH>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH>
  operator-(ST x, VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH> const &y){
    return VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH>(x) - y;
  }

  template<typename ST, typename REGISTER_TUPLE, typename IDX_SEQ, bool FIXED_LENGTH>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH>
  operator*(ST x, VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH> const &y){
    return VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH>(x) * y;
  }

  template<typename ST, typename REGISTER_TUPLE, typename IDX_SEQ, bool FIXED_LENGTH>
  RAJA_HOST_DEVICE
  RAJA_INLINE
  VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH>
  operator/(ST x, VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH> const &y){
    return VectorImpl<REGISTER_TUPLE, IDX_SEQ, FIXED_LENGTH>(x) / y;
  }


  template<typename REGISTER_TYPE, size_t VEC_NUM_ELEM>
  struct FixedVectorTypeHelper;

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, size_t REG_NUM_ELEM, size_t VEC_NUM_ELEM>
  struct FixedVectorTypeHelper<Register<REGISTER_POLICY, ELEMENT_TYPE, REG_NUM_ELEM>, VEC_NUM_ELEM>{


    static constexpr camp::idx_t s_num_full_registers = VEC_NUM_ELEM / REG_NUM_ELEM;

    static constexpr camp::idx_t s_num_full_elem = s_num_full_registers*REG_NUM_ELEM;

    static constexpr camp::idx_t s_num_partial_elem = VEC_NUM_ELEM - s_num_full_elem;

    static constexpr camp::idx_t s_num_partial_registers = s_num_partial_elem > 0 ? 1 : 0;

    using full_register_type = Register<REGISTER_POLICY, ELEMENT_TYPE, REG_NUM_ELEM>;

    using partial_register_type =
        Register<REGISTER_POLICY, ELEMENT_TYPE, s_num_partial_elem ? s_num_partial_elem : 1>;


    // Create lists of registers
    using full_register_list = internal::list_of_n<full_register_type, s_num_full_registers>;
    using partial_register_list = camp::list<partial_register_type>;

    using register_list_t = typename
        std::conditional<s_num_partial_registers == 0,
        full_register_list,
        typename camp::extend<full_register_list, partial_register_list>::type>::type;

    using register_idx_seq_t = camp::make_idx_seq_t<s_num_full_registers+s_num_partial_registers>;

    // Create actual VectorImpl type
    using type = VectorImpl<register_list_t, register_idx_seq_t, true>;

  };


  } //namespace internal






  template<typename REGISTER_TYPE, size_t NUM_ELEM>
  //using FixedVectorExt = internal::VectorImpl<typename internal::FixedVectorTypeHelper<REGISTER_TYPE, NUM_ELEM>::register_list_t, typename internal::FixedVectorTypeHelper<REGISTER_TYPE, NUM_ELEM>::idx_seq_t, true>;
  using FixedVectorExt = typename internal::FixedVectorTypeHelper<REGISTER_TYPE, NUM_ELEM>::type;

  template<typename REGISTER_TYPE, size_t NUM_REGISTERS>
  using StreamVectorExt = internal::VectorImpl<internal::list_of_n<REGISTER_TYPE, NUM_REGISTERS>, camp::make_idx_seq_t<NUM_REGISTERS>, false>;



}  // namespace RAJA


#endif
