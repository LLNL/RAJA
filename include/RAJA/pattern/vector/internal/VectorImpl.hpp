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

#ifndef RAJA_pattern_vector_vectorimpl_HPP
#define RAJA_pattern_vector_vectorimpl_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/vector/internal/VectorProductRef.hpp"
#include "RAJA/pattern/vector/internal/VectorRef.hpp"

#include <array>

namespace RAJA
{

  enum VectorSizeType
  {
    VECTOR_STREAM,
    VECTOR_FIXED
  };

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
  class Vector;

  namespace internal
  {


    template<camp::idx_t DEFAULT, bool IS_STATIC>
    class SemiStaticValue;

    template<camp::idx_t DEFAULT>
    class SemiStaticValue<DEFAULT, false>
    {
      private:
        camp::idx_t m_value;

      public:
        using self_type = SemiStaticValue<DEFAULT, false>;

        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr
        SemiStaticValue() : m_value(DEFAULT) {}

        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr
        SemiStaticValue(self_type const &c) : m_value(c.m_value) {}

        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr camp::idx_t get() const{
          return m_value;
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        void set(camp::idx_t new_value){
          m_value = new_value;
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        void min(self_type const &a, self_type const &b){
          m_value = a.m_value < b.m_value ?
                    a.m_value : b.m_value;
        }
    };

    template<camp::idx_t DEFAULT>
    class SemiStaticValue<DEFAULT, true>
    {
      public:
        using self_type = SemiStaticValue<DEFAULT, true>;

        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr
        SemiStaticValue(){}

        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr
        SemiStaticValue(self_type const &){}

        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr camp::idx_t get() const{
          return DEFAULT;
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        void set(camp::idx_t ){
          // NOP
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        void min(self_type const &, self_type const &){
        }
    };


    template<typename TYPE_LIST>
    struct tuple_from_list;

    template<typename ... TYPES>
    struct tuple_from_list<camp::list<TYPES...>>{
        using type = camp::tuple<TYPES...>;
    };

    template<typename TYPE_LIST>
    using tuple_from_list_t = typename tuple_from_list<TYPE_LIST>::type;


    template<typename T, typename I>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    constexpr
    T const &to_first(T const &value, I const &){
      return value;
    }


    /*
     * Helper that compute template arguments to VectorImpl
     */
    template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t VEC_NUM_ELEM, VectorSizeType VECTOR_TYPE>
    struct VectorTypeHelper{

      using register_type = Register<REGISTER_POLICY, ELEMENT_TYPE>;

      static constexpr camp::idx_t s_num_full_registers =
        VEC_NUM_ELEM / register_type::s_num_elem;

      static constexpr camp::idx_t s_num_registers =
          s_num_full_registers + (( (VEC_NUM_ELEM % register_type::s_num_elem) > 0) ? 1 : 0);


    };





    template<typename REGISTER_POLICY, typename ELEMENT_TYPE, typename REG_SEQ, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
    class VectorBase;

    template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t ... REG_SEQ, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
    class VectorBase<REGISTER_POLICY, ELEMENT_TYPE, camp::idx_seq<REG_SEQ...>, NUM_ELEM, VECTOR_TYPE>
    {
      public:
        using self_type = Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>;
        using register_policy = REGISTER_POLICY;
        using element_type = ELEMENT_TYPE;
        using register_type = Register<REGISTER_POLICY, ELEMENT_TYPE>;

        static constexpr camp::idx_t s_num_elem = NUM_ELEM;
        static constexpr camp::idx_t s_num_reg_elem = register_type::s_num_elem;
        static constexpr camp::idx_t s_num_registers = sizeof...(REG_SEQ);
        static constexpr bool s_is_fixed = VECTOR_TYPE==VECTOR_FIXED;
      private:


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


      private:

        register_type m_registers[sizeof...(REG_SEQ)];

        SemiStaticValue<NUM_ELEM, VECTOR_TYPE==VECTOR_FIXED> m_length;



        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr
        camp::idx_t regNumElem(camp::idx_t reg) const {
          // How many elements of this register are there?
          return (1+reg)*s_num_reg_elem < m_length.get()
            ? s_num_reg_elem                       // Full register
            : m_length.get()-reg*s_num_reg_elem;  // Partial register
        }

      public:


        /*!
         * @brief Default constructor, zeros register contents
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorBase() : m_length(){}

        /*!
         * @brief Copy constructor
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorBase(self_type const &c) :
          m_registers{c.m_registers[REG_SEQ]...},
          m_length(c.m_length)
        {
        }

        /*!
         * @brief Scalar constructor (broadcast)
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorBase(element_type const &c) :
          m_registers{to_first(register_type(c), REG_SEQ)...},
          m_length()
        {
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        constexpr
        bool is_root() {
          return register_type::is_root();
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type get(camp::idx_t i) const
        {
          return m_registers[i/s_num_reg_elem].get(i%s_num_reg_elem);
        }



        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& set(camp::idx_t i, element_type value)
        {
          m_registers[i/s_num_reg_elem].set(i%s_num_reg_elem, value);
          return *getThis();
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr
        camp::idx_t size() const
        {
          return m_length.get();
        }

        /*!
         * Gets the current size of matrix along specified dimension
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr camp::idx_t dim_elem(camp::idx_t dim) const {
          return (dim==0) ? m_length.get() : 0;
        }

        /*!
         * Gets the maximum size of matrix along specified dimension
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        constexpr camp::idx_t s_dim_elem(camp::idx_t dim){
          return (dim==0) ? NUM_ELEM : 0;
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





        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& load(element_type const *ptr, camp::idx_t stride = 1, camp::idx_t length = NUM_ELEM){

          m_length.set(length);
//          printf("load(n=%d)\n", (int)length);

          camp::sink(
              m_registers[REG_SEQ].load(
                  ptr + REG_SEQ*s_num_reg_elem*stride,
                  stride,
                  regNumElem(REG_SEQ)) ...);


          return *getThis();
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type const& store(element_type *ptr, camp::idx_t stride = 1) const {

//          printf("store: length=%d  ", (int)m_length.get());
//
//          camp::sink(printf("REG[%d].store(n=%d) ", (int)REG_SEQ, (int)regNumElem(REG_SEQ))...);

          camp::sink(
              m_registers[REG_SEQ].store(
                  ptr + REG_SEQ*s_num_reg_elem*stride,
                  stride,
                  regNumElem(REG_SEQ)) ...);

//          printf("\n");

          return *getThis();
        }



        RAJA_HOST_DEVICE
        RAJA_INLINE
        void broadcast(element_type const &value){

          m_length.set(NUM_ELEM);

          camp::sink(m_registers[REG_SEQ].broadcast(value)...);

        }


        /*!
         * @brief Copy values of another vector
         * @param x The other vector to copy
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type  &copy(self_type const &x){

          camp::sink( (m_registers[REG_SEQ] = x.m_registers[REG_SEQ])...);

          m_length.set(x.m_length.get());

          return *getThis();
        }



        /*!
         * @brief Element-wise addition of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type add(self_type const &b) const {

          self_type result;

          result.m_length.min(m_length, b.m_length);

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].add(b.m_registers[REG_SEQ]))...);

          return result;
        }

        /*!
         * @brief Element-wise subtraction of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type subtract(self_type const &b) const {

          self_type result;

          result.m_length.min(m_length, b.m_length);

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].subtract(b.m_registers[REG_SEQ]))...);

          return result;
        }

        /*!
         * @brief Element-wise multiplication of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type multiply(self_type const &b) const {

          self_type result;

          result.m_length.min(m_length, b.m_length);

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].multiply(b.m_registers[REG_SEQ]))...);

          return result;
        }

        /*!
         * @brief Element-wise division of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type divide(self_type const &b) const {

          self_type result;

          result.m_length.min(m_length, b.m_length);

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].divide(b.m_registers[REG_SEQ], regNumElem(REG_SEQ)))...);

          return result;
        }


        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type fused_multiply_add(self_type const &b, self_type const &c) const {

          self_type result;

          result.m_length.min(m_length, b.m_length);

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].fused_multiply_add(b.m_registers[REG_SEQ], c.m_registers[REG_SEQ]))...);

          return result;
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type fused_multiply_subtract(self_type const &b, self_type const &c) const {

          self_type result;

          result.m_length.min(m_length, b.m_length);

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].fused_multiply_subtract(b.m_registers[REG_SEQ], c.m_registers[REG_SEQ]))...);

          return result;
        }


        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type vmin(self_type const &b) const {

          self_type result;

          result.m_length.min(m_length, b.m_length);

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].vmin(b.m_registers[REG_SEQ]))...);

          return result;
        }



        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type vmax(self_type const &b) const {

          self_type result;

          result.m_length.min(m_length, b.m_length);

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].vmax(b.m_registers[REG_SEQ]))...);

          return result;
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


        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type sum(camp::idx_t = NUM_ELEM) const {
          return foldl_sum<element_type>(m_registers[REG_SEQ].sum(regNumElem(REG_SEQ))...);
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type max(camp::idx_t = NUM_ELEM) const {
          return foldl_max<element_type>(m_registers[REG_SEQ].max(regNumElem(REG_SEQ))...);
        }


        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type min(camp::idx_t = NUM_ELEM) const {
          return foldl_min<element_type>(m_registers[REG_SEQ].min(regNumElem(REG_SEQ))...);
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
          return *getThis();
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

          return *getThis();
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
          return *getThis();
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
          *getThis() = subtract(x);
          return *getThis();
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
          return VectorProductRef<self_type>(*getThis(), x);
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
          *getThis() = multiply(x);
          return *getThis();
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
          *getThis() = divide(x);
          return *getThis();
        }

    };





  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
  using makeVectorBase = VectorBase<REGISTER_POLICY,
                                    ELEMENT_TYPE,
                                    camp::make_idx_seq_t<internal::VectorTypeHelper<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>::s_num_registers>,
                                    NUM_ELEM,
                                    VECTOR_TYPE>;



  } // namespace internal

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
  class Vector;

  namespace internal
  {


  /*
   * Helper that computes a similar vector to the one provided, but of a
   * different length
   */
  template<typename VECTOR_TYPE, camp::idx_t NEW_LENGTH>
  struct VectorNewLengthHelper;

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE, camp::idx_t NEW_LENGTH>
  struct VectorNewLengthHelper<Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>, NEW_LENGTH> {
      using type = Vector<REGISTER_POLICY, ELEMENT_TYPE, NEW_LENGTH, VECTOR_TYPE>;
  };





  } //namespace internal





}  // namespace RAJA


#endif
