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

#ifndef RAJA_pattern_vector_vectorbase_HPP
#define RAJA_pattern_vector_vectorbase_HPP

#include "RAJA/config.hpp"
#include "RAJA/util/macros.hpp"
#include "RAJA/pattern/vector/internal/stats.hpp"
#include "RAJA/pattern/vector/internal/VectorProductRef.hpp"
#include "RAJA/pattern/vector/internal/VectorRef.hpp"


#include <array>
#include <type_traits>

namespace RAJA
{

  enum VectorSizeType
  {
    VECTOR_STREAM,
    VECTOR_FIXED
  };


  namespace internal
  {

    struct InitSizeOnlyTag{};


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

      static constexpr camp::idx_t s_num_partial_elem =
          s_num_full_registers*register_type::s_num_elem - VEC_NUM_ELEM;

      static constexpr camp::idx_t s_num_registers =
          s_num_full_registers + (( s_num_partial_elem > 0) ? 1 : 0);

      using reg_seq = camp::make_idx_seq_t<s_num_full_registers>;

      using part_reg_seq = typename
          std::conditional<s_num_partial_elem == 0,
                          camp::idx_seq<>,
                          camp::idx_seq<s_num_full_registers> >::type;
    };


    // Forward Declaration
    template<VectorSizeType VECTOR_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, typename REG_SEQ, typename PART_REG_SEQ, camp::idx_t NUM_ELEM>
    class VectorImpl;


    /**
     * Base Vector class
     *
     * This is the base Vector class that implements register storage and
     * length-agnostic operations (add, mul, etc)
     *
     * Assumptions:
     *   - We have a set of full registers
     *   - All operations are implemented on all register elements
     *   - Any subsetting (dynamic or static) is implemented on derived types
     *   - Derived type must be passed in through SELF_TYPE parameter
     */
    //template<typename REGISTER_POLICY, typename ELEMENT_TYPE, typename REG_SEQ, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
    template<typename VECTOR_TYPE>
    class VectorBase;

    template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t ... REG_SEQ, camp::idx_t ... PART_REG_SEQ, camp::idx_t NUM_ELEM, VectorSizeType VECTOR_TYPE>
    class VectorBase<VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, camp::idx_seq<REG_SEQ...>, camp::idx_seq<PART_REG_SEQ...>, NUM_ELEM>>
    {
      public:
        using self_type = VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, camp::idx_seq<REG_SEQ...>, camp::idx_seq<PART_REG_SEQ...>, NUM_ELEM>;
        using vector_base_type = VectorBase<self_type>;
        using element_type = ELEMENT_TYPE;
        using register_type = Register<REGISTER_POLICY, ELEMENT_TYPE>;

        static constexpr camp::idx_t s_num_elem = NUM_ELEM;
        static constexpr camp::idx_t s_num_reg_elem = register_type::s_num_elem;
        static constexpr camp::idx_t s_num_full_registers = sizeof...(REG_SEQ);
        static constexpr camp::idx_t s_num_part_registers = sizeof...(PART_REG_SEQ);
        static constexpr camp::idx_t s_num_registers = sizeof...(REG_SEQ) + sizeof...(PART_REG_SEQ);
        static constexpr camp::idx_t s_num_partial_elem = s_num_reg_elem -
            (s_num_registers*register_type::s_num_elem - NUM_ELEM);
        static constexpr bool s_is_fixed = VECTOR_TYPE==VECTOR_FIXED;
      protected:


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


        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &copy(self_type const &c){
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_copy ++;
#endif
          camp::sink( (m_registers[REG_SEQ] = c.m_registers[REG_SEQ])...);
          camp::sink( (m_registers[PART_REG_SEQ] = c.m_registers[PART_REG_SEQ])...);
          return *getThis();
        }


        register_type m_registers[s_num_registers];



      public:


        /*!
         * @brief Default constructor, zeros register contents
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorBase(){}

        /*!
         * @brief Copy constructor
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorBase(VectorBase const &c) :
          m_registers{c.m_registers[REG_SEQ]..., c.m_registers[PART_REG_SEQ]...}
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_copy_ctor ++;
#endif
        }

        /*!
         * @brief Scalar constructor (broadcast)
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        VectorBase(element_type const &c) :
          m_registers{to_first(register_type(c), REG_SEQ)..., to_first(register_type(c), PART_REG_SEQ)...}
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_broadcast_ctor ++;
#endif
        }

        /*!
         * @brief Are we this vector's root thread of execution?
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        constexpr
        bool is_root() {
          return register_type::is_root();
        }

        RAJA_INLINE
        std::string toString() const {
          std::string s = "[";
          for(int i = 0;i < s_num_registers;++ i){
            s += " " + m_registers[i].toString();
          }

          return s + " ]";
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& load_packed(element_type const *ptr){


#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_load_packed ++;
#endif

          // load full registers
          camp::sink(
              m_registers[REG_SEQ].load_packed(
                  ptr + REG_SEQ*s_num_reg_elem) ...);

          // load partial register
          camp::sink(
              m_registers[PART_REG_SEQ].load_packed_n(
                  ptr + PART_REG_SEQ*s_num_reg_elem,
                  s_num_partial_elem) ...);

          return *getThis();
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& load_packed_n(element_type const *ptr, camp::idx_t N){

          if(N == NUM_ELEM){
            return load_packed(ptr);
          }

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_load_packed_n ++;
#endif

          // load full packed registers
          int elem = 0;
          int reg = 0;
          for(;reg < s_num_full_registers && elem+s_num_reg_elem <= N;reg ++){
            m_registers[reg].load_packed(ptr+elem);
            elem += s_num_reg_elem;
          }

          // load partial register for remaining elements
          if(elem < N){
            m_registers[reg].load_packed_n(ptr+elem, N - elem);
            ++reg;
          }

          // zero fill remaining registers
          while(reg < s_num_registers){
            m_registers[reg].broadcast(element_type(0));
            ++ reg;
          }

          return *getThis();
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& load_strided(element_type const *ptr, camp::idx_t stride){
          if(stride == 1){
            return load_packed(ptr);
          }

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_load_strided ++;
#endif

          // load full registers
          camp::sink(
              m_registers[REG_SEQ].load_strided(
                  ptr + REG_SEQ*s_num_reg_elem*stride,
                  stride) ...);

          // load partial register
          camp::sink(
              m_registers[PART_REG_SEQ].load_strided_n(
                  ptr + PART_REG_SEQ*s_num_reg_elem*stride,
                  stride,
                  s_num_partial_elem) ...);

          return *getThis();
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& load_strided_n(element_type const *ptr, camp::idx_t stride, camp::idx_t N){

          if(N == NUM_ELEM){
            return load_strided(ptr, stride);
          }

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_load_strided_n ++;
#endif

          // load full packed registers
          int elem = 0;
          int reg = 0;
          for(;reg < s_num_full_registers && elem+s_num_reg_elem <= N;reg ++){
            m_registers[reg].load_strided(ptr+elem*stride, stride);
            elem += s_num_reg_elem;
          }

          // load partial register for remaining elements
          if(elem < N){
            m_registers[reg].load_strided_n(ptr+elem*stride, stride, N - elem);
            ++reg;
          }

          // zero fill remaining registers
          while(reg < s_num_registers){
            m_registers[reg].broadcast(element_type(0));
            ++ reg;
          }

          return *getThis();

        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type const& store_packed(element_type *ptr) const {

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_store_packed ++;
#endif

          // store full registers
          camp::sink(
              m_registers[REG_SEQ].store_packed(
                  ptr + REG_SEQ*s_num_reg_elem) ...);

          // store partial register
          camp::sink(
              m_registers[PART_REG_SEQ].store_packed_n(
                  ptr + PART_REG_SEQ*s_num_reg_elem,
                  s_num_partial_elem) ...);

          return *getThis();
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type const & store_packed_n(element_type *ptr, camp::idx_t N) const {

          if(N == NUM_ELEM){
            return store_packed(ptr);
          }

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_store_packed_n ++;
#endif

          // store full packed registers
          int elem = 0;
          int reg = 0;
          for(;reg < s_num_full_registers && elem+s_num_reg_elem <= N;reg ++){
            m_registers[reg].store_packed(ptr+elem);
            elem += s_num_reg_elem;
          }

          // store partial register for remaining elements
          if(elem < N){
            m_registers[reg].store_packed_n(ptr+elem, N - elem);
            ++reg;
          }

          return *getThis();
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type const& store_strided(element_type *ptr, camp::idx_t stride) const {

          if(stride == 1){
            return store_packed(ptr);
          }

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_store_strided ++;
#endif

          // store full registers
          camp::sink(
              m_registers[REG_SEQ].store_strided(
                  ptr + REG_SEQ*s_num_reg_elem*stride,
                  stride) ...);

          // store partial register
          camp::sink(
              m_registers[PART_REG_SEQ].store_strided_n(
                  ptr + PART_REG_SEQ*s_num_reg_elem*stride,
                  stride,
                  s_num_partial_elem) ...);

          return *getThis();
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type const & store_strided_n(element_type *ptr, camp::idx_t stride, camp::idx_t N) const {

          if(N == NUM_ELEM){
            return store_strided(ptr, stride);
          }

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_store_strided_n ++;
#endif

          // store full packed registers
          int elem = 0;
          int reg = 0;
          for(;reg < s_num_full_registers && elem+s_num_reg_elem <= N;reg ++){
            m_registers[reg].store_strided(ptr+elem*stride, stride);
            elem += s_num_reg_elem;
          }

          // store partial register for remaining elements
          if(elem < N){
            m_registers[reg].store_strided_n(ptr+elem*stride, stride, N - elem);
            ++reg;
          }

          return *getThis();
        }


        /*!
         * @brief Set all vector elements to value
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &broadcast(element_type const &value){

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_broadcast ++;
#endif
          camp::sink(m_registers[REG_SEQ].broadcast(value)...);
          camp::sink(m_registers[PART_REG_SEQ].broadcast_n(value, s_num_partial_elem)...);

          return *this;
        }

        /*!
         * @brief Get i'th value of vector by value
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type get(camp::idx_t i) const
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_get ++;
#endif
          camp::idx_t reg = 0;
          camp::idx_t elem = i;
          while(elem >= 0){
            if(elem < s_num_reg_elem){
              return m_registers[reg].get(elem);
            }
            ++ reg;
            elem -= s_num_reg_elem;
          }
          return 0;
          //return m_registers[i/s_num_reg_elem].get(i%s_num_reg_elem);
        }


        /*!
         * @brief Set i'th value of vector
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type& set(camp::idx_t i, element_type value)
        {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_set ++;
#endif
//          m_registers[i/s_num_reg_elem].set(i%s_num_reg_elem, value);

          camp::idx_t reg = 0;
          camp::idx_t elem = i;
          while(elem >= 0){
            if(elem < s_num_reg_elem){
              m_registers[reg].set(elem, value);
              return *getThis();
            }
            ++ reg;
            elem -= s_num_reg_elem;
          }

          return *getThis();
        }



        /*!
         * Gets the current size of matrix along specified dimension
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        constexpr camp::idx_t dim_elem(camp::idx_t dim) const {
          return (dim==0) ? getThis()->size() : 0;
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





        /*!
         * @brief Element-wise addition of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type add(self_type const &b) const {

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_add ++;
#endif

          self_type result(b, InitSizeOnlyTag{});

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].add(b.m_registers[REG_SEQ]))...);
          camp::sink( (result.m_registers[PART_REG_SEQ] = m_registers[PART_REG_SEQ].add(b.m_registers[PART_REG_SEQ]))...);

          return result;
        }

        /*!
         * @brief Element-wise subtraction of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type subtract(self_type const &b) const {

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_subtract ++;
#endif

          self_type result(b, InitSizeOnlyTag{});

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].subtract(b.m_registers[REG_SEQ]))...);
          camp::sink( (result.m_registers[PART_REG_SEQ] = m_registers[PART_REG_SEQ].subtract(b.m_registers[PART_REG_SEQ]))...);

          return result;
        }

        /*!
         * @brief Element-wise multiplication of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type multiply(self_type const &b) const {

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_multiply ++;
#endif

          self_type result(b, InitSizeOnlyTag{});

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].multiply(b.m_registers[REG_SEQ]))...);
          camp::sink( (result.m_registers[PART_REG_SEQ] = m_registers[PART_REG_SEQ].multiply(b.m_registers[PART_REG_SEQ]))...);

          return result;
        }




        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type fused_multiply_add(self_type const &b, self_type const &c) const {

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_fma ++;
#endif

          self_type result(b, InitSizeOnlyTag{});

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].fused_multiply_add(b.m_registers[REG_SEQ], c.m_registers[REG_SEQ]))...);
          camp::sink( (result.m_registers[PART_REG_SEQ] = m_registers[PART_REG_SEQ].fused_multiply_add(b.m_registers[PART_REG_SEQ], c.m_registers[PART_REG_SEQ]))...);


          return result;
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type fused_multiply_subtract(self_type const &b, self_type const &c) const {

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_fms ++;
#endif

          self_type result(b, InitSizeOnlyTag{});

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].fused_multiply_subtract(b.m_registers[REG_SEQ], c.m_registers[REG_SEQ]))...);
          camp::sink( (result.m_registers[PART_REG_SEQ] = m_registers[PART_REG_SEQ].fused_multiply_subtract(b.m_registers[PART_REG_SEQ], c.m_registers[PART_REG_SEQ]))...);

          return result;
        }


        /*!
         * @brief Returns the sum of all elements in the vector
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type sum(camp::idx_t = NUM_ELEM) const {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_sum ++;
#endif
          return foldl_sum<element_type>(m_registers[REG_SEQ].sum(s_num_reg_elem)..., m_registers[PART_REG_SEQ].sum(s_num_partial_elem)...);
        }

        /*!
         * @brief Returns the maximum value of all elements in the vector
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type max(camp::idx_t = NUM_ELEM) const {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_max ++;
#endif
          return foldl_max<element_type>(m_registers[REG_SEQ].max(s_num_reg_elem)..., m_registers[PART_REG_SEQ].max(s_num_partial_elem)...);
        }


        /*!
         * @brief Returns the minimum value of all elements in the vector
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        element_type min(camp::idx_t = NUM_ELEM) const {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_min ++;
#endif
          return foldl_min<element_type>(m_registers[REG_SEQ].min(s_num_reg_elem)..., m_registers[PART_REG_SEQ].min(s_num_partial_elem)...);
        }


        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type vmin(self_type const &b) const {
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_vmin ++;
#endif

          self_type result(b, InitSizeOnlyTag{});

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].vmin(b.m_registers[REG_SEQ]))...);
          camp::sink( (result.m_registers[PART_REG_SEQ] = m_registers[PART_REG_SEQ].vmin(b.m_registers[PART_REG_SEQ]))...);

          return result;
        }



        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type vmax(self_type const &b) const {

#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_vmax ++;
#endif

          self_type result(b, InitSizeOnlyTag{});

          camp::sink( (result.m_registers[REG_SEQ] = m_registers[REG_SEQ].vmax(b.m_registers[REG_SEQ]))...);
          camp::sink( (result.m_registers[PART_REG_SEQ] = m_registers[PART_REG_SEQ].vmax(b.m_registers[PART_REG_SEQ]))...);

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
#ifdef RAJA_ENABLE_VECTOR_STATS
          RAJA::vector_stats::num_vector_dot ++;
#endif
          return getThis()->multiply(x).sum();
        }





        /*!
         * @brief Set entire vector to a single scalar value
         * @param value Value to set all vector elements to
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        self_type &operator=(element_type value)
        {
          getThis()->broadcast(value);
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
          getThis()->copy(x);

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
          return getThis()->add(x);
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
          *getThis() = getThis()->add(x);
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
          return getThis()->subtract(x);
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
          *getThis() = getThis()->subtract(x);
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
          *getThis() = getThis()->multiply(x);
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
          return getThis()->divide(x);
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
          *getThis() = getThis()->divide(x);
          return *getThis();
        }

    };












    template<VectorSizeType VECTOR_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, typename REG_SEQ, typename PART_REG_SEQ, camp::idx_t NUM_ELEM>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>
    operator+(typename VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>::element_type x, VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM> const &y){
      return VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>(x) + y;
    }

    template<VectorSizeType VECTOR_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, typename REG_SEQ, typename PART_REG_SEQ, camp::idx_t NUM_ELEM>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>
    operator-(typename VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>::element_type x, VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM> const &y){
      return VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>(x) - y;
    }

    template<VectorSizeType VECTOR_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, typename REG_SEQ, typename PART_REG_SEQ, camp::idx_t NUM_ELEM>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>
    operator*(typename VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>::element_type x, VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM> const &y){
      return VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>(x) * y;
    }

    template<VectorSizeType VECTOR_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, typename REG_SEQ, typename PART_REG_SEQ, camp::idx_t NUM_ELEM>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>
    operator/(typename VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>::element_type x, VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM> const &y){
      return VectorImpl<VECTOR_TYPE, REGISTER_POLICY, ELEMENT_TYPE, REG_SEQ, PART_REG_SEQ, NUM_ELEM>(x) / y;
    }



    template<VectorSizeType VECTOR_TYPE, typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM>
    using makeVectorImpl = VectorImpl<VECTOR_TYPE,
                                      REGISTER_POLICY,
                                      ELEMENT_TYPE,
                                      typename internal::VectorTypeHelper<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>::reg_seq,
                                      typename internal::VectorTypeHelper<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>::part_reg_seq,
                                      NUM_ELEM>;


    /*
     * Helper that computes a similar vector to the one provided, but of a
     * different length
     */
    template<typename VECTOR, camp::idx_t NEW_LENGTH>
    struct VectorNewLengthHelper;

    template<VectorSizeType VECTOR_TYPE, typename REGISTER_POLICY, typename T, typename REG_SEQ, typename PART_REG_SEQ, camp::idx_t NUM_ELEM, camp::idx_t NEW_LENGTH>
    struct VectorNewLengthHelper<VectorImpl<VECTOR_TYPE, REGISTER_POLICY, T, REG_SEQ, PART_REG_SEQ, NUM_ELEM>, NEW_LENGTH> {

        using type = internal::makeVectorImpl<VECTOR_TYPE,
                                 REGISTER_POLICY,
                                 T,
                                 NEW_LENGTH>;

    };





  } //namespace internal
}  // namespace RAJA


#include "RAJA/pattern/vector/internal/FixedVector.hpp"
#include "RAJA/pattern/vector/internal/StreamVector.hpp"


#endif
