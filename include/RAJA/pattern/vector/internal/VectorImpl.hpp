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

  enum VECTOR_LENGTH_TYPE
  {
    VECTOR_STREAM,
    VECTOR_FIXED
  };

  namespace internal
  {

  namespace detail
  {

    template<typename TYPE_LIST>
    struct tuple_from_list;

    template<typename ... TYPES>
    struct tuple_from_list<camp::list<TYPES...>>{
        using type = camp::tuple<TYPES...>;
    };

    template<typename TYPE_LIST>
    using tuple_from_list_t = typename tuple_from_list<TYPE_LIST>::type;


    /*
     * Helper that compute template arguments to VectorImpl
     */
    template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t VEC_NUM_ELEM, VECTOR_LENGTH_TYPE VECTOR_TYPE>
    struct VectorTypeHelper{

      using register_traits_t = RegisterTraits<REGISTER_POLICY, ELEMENT_TYPE>;

      static constexpr camp::idx_t num_full_registers =
        VEC_NUM_ELEM / register_traits_t::num_elem();


      // number of elements in a partial final register for Fix
      static constexpr camp::idx_t num_partial_elem =
        VEC_NUM_ELEM - num_full_registers*register_traits_t::num_elem();


      static constexpr camp::idx_t num_registers =
        num_full_registers + (num_partial_elem > 0 ? 1 : 0);

      using register_idx_seq_t = camp::make_idx_seq_t<num_registers>;

      using full_register_type = Register<REGISTER_POLICY, ELEMENT_TYPE>;

      using partial_register_type =
          Register<REGISTER_POLICY, ELEMENT_TYPE, num_partial_elem ? num_partial_elem : 1>;


      // Create lists of registers for a fixed vector
      using fixed_full_registers = internal::list_of_n<full_register_type, num_full_registers>;
      using fixed_partial_register_list = camp::list<partial_register_type>;

      using fixed_register_list_t = typename
          std::conditional<num_partial_elem == 0,
          fixed_full_registers,
          typename camp::extend<fixed_full_registers, fixed_partial_register_list>::type>::type;


      using stream_register_list_t = list_of_n<full_register_type, num_registers>;


      using register_list_t = typename
          std::conditional<VECTOR_TYPE == VECTOR_STREAM,
            stream_register_list_t,
            fixed_register_list_t>::type;

      using register_tuple_t = tuple_from_list_t<register_list_t>;


    };





    template<typename ELEMENT_TYPE>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    ELEMENT_TYPE
    VectorGetByIndex(camp::idx_t){
      return ELEMENT_TYPE(); // termination case: this is undefined behavior!
    }

    template<typename ELEMENT_TYPE, typename REGISTER0, typename ... REGISTER_REST>
    RAJA_HOST_DEVICE
    RAJA_INLINE
    ELEMENT_TYPE
    VectorGetByIndex(camp::idx_t i, REGISTER0 const &r0, REGISTER_REST const &...r_rest){

      if(i < REGISTER0::num_elem()){
        return r0.get(i);
      }
      else{
        return VectorGetByIndex<ELEMENT_TYPE>(i-REGISTER0::num_elem(), r_rest...);
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
      if((camp::idx_t)i < (camp::idx_t)REGISTER0::num_elem()){
        r0.set(i, value);
      }
      else{
        VectorSetByIndex((camp::idx_t)i-(camp::idx_t)REGISTER0::num_elem(), value, r_rest...);
      }
    }


    template<typename REGISTER_TUPLE, typename REG_SEQ, camp::idx_t NUM_ELEM, VECTOR_LENGTH_TYPE VECTOR_TYPE>
    struct VectorOpHelper;

    template<typename REGISTER0, typename ... REGISTERS, camp::idx_t ... REG_SEQ, camp::idx_t NUM_ELEM, VECTOR_LENGTH_TYPE VECTOR_TYPE>
    struct VectorOpHelper<camp::tuple<REGISTER0, REGISTERS...>, camp::idx_seq<REG_SEQ...>, NUM_ELEM, VECTOR_TYPE>
    {
        using register_policy = typename REGISTER0::register_policy;
        using element_type = typename REGISTER0::element_type;
        using tuple_t = camp::tuple<REGISTER0, REGISTERS...>;
        using register_traits_t = RegisterTraits<register_policy, element_type>;
        using range_type = TypedRangeSegment<camp::idx_t, camp::idx_t>;

        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        constexpr
        bool has_full_register(camp::idx_t length){
          return length >= register_traits_t::num_elem();
        }


        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        constexpr
        bool is_full_register(camp::idx_t reg_idx, camp::idx_t length){
          return (reg_idx+1)*register_traits_t::num_elem() <= length;
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        range_type remainder(camp::idx_t length){
          return range_type((length / register_traits_t::num_elem()) * register_traits_t::num_elem(), length);
        }



        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        element_type get(tuple_t const &registers, camp::idx_t i)
        {

          return detail::VectorGetByIndex<element_type>(i, camp::get<REG_SEQ>(registers)...);
        }



        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void set(tuple_t &registers, camp::idx_t i, element_type value)
        {
          detail::VectorSetByIndex(i, value, camp::get<REG_SEQ>(registers)...);
        }



        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void load(tuple_t &registers, element_type const *ptr, camp::idx_t stride, camp::idx_t length){

          // Do full vector loads where possible
          camp::sink((

              // Is REG_SEQ getting fully loaded?
              is_full_register(REG_SEQ, length) ?

              // It's a full load:
              camp::get<REG_SEQ>(registers).load(
                  ptr + REG_SEQ*register_traits_t::num_elem()*stride,
                  stride).sink()

              // Not a full load, so NOP (register will get loaded below)
              : camp::get<REG_SEQ>(registers).sink()

            )...);

          // Load remainder of elements
          for(camp::idx_t i : remainder(length)){
            set(registers, i, ptr[i*stride]);
          }


        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void store(tuple_t const &registers, element_type *ptr, camp::idx_t stride, camp::idx_t length){
          // Do full vector stores where possible
          camp::sink((

              // Is REG_SEQ getting fully loaded?
              is_full_register(REG_SEQ, length) ?

              // It's a full load:
              camp::get<REG_SEQ>(registers).store(
                  ptr + REG_SEQ*register_traits_t::num_elem()*stride,
                  stride).sink()

              // Not a full load, so NOP (register will get loaded below)
              : camp::get<REG_SEQ>(registers).sink()

            )...);

          // Store remainder of elements
          for(camp::idx_t i : remainder(length)){
            ptr[i*stride] = get(registers, i);
          }
        }



        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void broadcast(tuple_t &registers, element_type const &value){

          camp::sink(camp::get<REG_SEQ>(registers).broadcast(value)...);

        }



        /*!
         * @brief Element-wise addition of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void add(tuple_t const &registers_a, tuple_t const &registers_b, tuple_t &registers_result) {

          camp::sink((camp::get<REG_SEQ>(registers_result) = camp::get<REG_SEQ>(registers_a) + camp::get<REG_SEQ>(registers_b))...);


        }

        /*!
         * @brief Element-wise subtraction of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void subtract(tuple_t const &registers_a, tuple_t const &registers_b, tuple_t &registers_result) {

          camp::sink((camp::get<REG_SEQ>(registers_result) = camp::get<REG_SEQ>(registers_a) - camp::get<REG_SEQ>(registers_b))...);

        }

        /*!
         * @brief Element-wise multiplication of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void multiply(tuple_t const &registers_a, tuple_t const &registers_b, tuple_t &registers_result) {

          camp::sink((camp::get<REG_SEQ>(registers_result) = camp::get<REG_SEQ>(registers_a) * camp::get<REG_SEQ>(registers_b))...);

        }

        /*!
         * @brief Element-wise division of two vectors
         */
        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void divide(tuple_t const &registers_a, tuple_t const &registers_b, tuple_t &registers_result) {

          camp::sink((camp::get<REG_SEQ>(registers_result) = camp::get<REG_SEQ>(registers_a) / camp::get<REG_SEQ>(registers_b))...);

        }


        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void fused_multiply_add(tuple_t const &registers_a, tuple_t const &registers_b, tuple_t const &registers_c, tuple_t &registers_result) {

          camp::sink((camp::get<REG_SEQ>(registers_result) =
              camp::get<REG_SEQ>(registers_a).fused_multiply_add(
                  camp::get<REG_SEQ>(registers_b),
                  camp::get<REG_SEQ>(registers_c)))...);

        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void fused_multiply_subtract(tuple_t const &registers_a, tuple_t const &registers_b, tuple_t const &registers_c, tuple_t &registers_result) {

          camp::sink((camp::get<REG_SEQ>(registers_result) =
              camp::get<REG_SEQ>(registers_a).fused_multiply_subtract(
                  camp::get<REG_SEQ>(registers_b),
                  camp::get<REG_SEQ>(registers_c)))...);

        }


        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void vmin(tuple_t const &registers_a, tuple_t const &registers_b, tuple_t &registers_result) {

          camp::sink((camp::get<REG_SEQ>(registers_result) = camp::get<REG_SEQ>(registers_a).vmin(camp::get<REG_SEQ>(registers_b)))...);

        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        void vmax(tuple_t const &registers_a, tuple_t const &registers_b, tuple_t &registers_result) {

          camp::sink((camp::get<REG_SEQ>(registers_result) = camp::get<REG_SEQ>(registers_a).vmax(camp::get<REG_SEQ>(registers_b)))...);

        }




        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        element_type sum(tuple_t const &registers, camp::idx_t length){
          // Sum across vectors
          REGISTER0 reg_reduce = camp::get<0>(registers);

          // Reduce all full registers to a single register
          camp::sink((

              // Is REG_SEQ getting fully loaded?
              // And not register 0, since we started with that
              (is_full_register(REG_SEQ, length) && REG_SEQ > 0 )?

              // Add to total
              (reg_reduce = reg_reduce + camp::get<REG_SEQ>(registers)).sink()

              // Not a full load, so NOP (register will get loaded below)
              : camp::get<REG_SEQ>(registers).sink()

            )...);

          // Reduce full registers to scalar
          element_type scalar_reduce =
             has_full_register(length) ?
                 reg_reduce.sum() : element_type(0);

          // Reduce remainder of elements
          for(camp::idx_t i : remainder(length)){
            scalar_reduce += get(registers, i);
          }

          return scalar_reduce;
        }

        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        element_type max(tuple_t const &registers, camp::idx_t length){
          // Sum across vectors
          REGISTER0 reg_reduce = camp::get<0>(registers);

          // Reduce all full registers to a single register
          camp::sink((

              // Is REG_SEQ getting fully loaded?
              // And not register 0, since we started with that
              (is_full_register(REG_SEQ, length) && REG_SEQ > 0 )?

              // Vector Reduce
              (reg_reduce = reg_reduce.vmax(camp::get<REG_SEQ>(registers))).sink()

              // Not a full load, so NOP (register will get loaded below)
              : camp::get<REG_SEQ>(registers).sink()

            )...);

          // Reduce full registers to scalar
          element_type scalar_reduce =
             has_full_register(length) ?
                 reg_reduce.max() : camp::get<0>(registers).get(0);


          // Reduce remainder of elements
          for(camp::idx_t i : remainder(length)){
            element_type v = get(registers, i);
            scalar_reduce = scalar_reduce < v ? v : scalar_reduce;
          }

          return scalar_reduce;
        }


        RAJA_HOST_DEVICE
        RAJA_INLINE
        static
        element_type min(tuple_t const &registers, camp::idx_t length){
          // Sum across vectors
          REGISTER0 reg_reduce = camp::get<0>(registers);

          // Reduce all full registers to a single register
          camp::sink((

              // Is REG_SEQ getting fully loaded?
              // And not register 0, since we started with that
              (is_full_register(REG_SEQ, length) && REG_SEQ > 0 )?

              // Vector Reduce
              (reg_reduce = reg_reduce.vmin(camp::get<REG_SEQ>(registers))).sink()

              // Not a full load, so NOP (register will get loaded below)
              : camp::get<REG_SEQ>(registers).sink()

            )...);

          // Reduce full registers to scalar
          element_type scalar_reduce =
             has_full_register(length) ?
                 reg_reduce.min() : camp::get<0>(registers).get(0);


          // Reduce remainder of elements
          for(camp::idx_t i : remainder(length)){
            element_type v = get(registers, i);
            scalar_reduce = scalar_reduce > v ? v : scalar_reduce;
          }

          return scalar_reduce;
        }

    };




  } // namespace detail




  } // namespace internal

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VECTOR_LENGTH_TYPE VECTOR_TYPE>
  class Vector;

  namespace internal
  {


  /*
   * Helper that computes a similar vector to the one provided, but of a
   * different length
   */
  template<typename VECTOR_TYPE, camp::idx_t NEW_LENGTH>
  struct VectorNewLengthHelper;

  template<typename REGISTER_POLICY, typename ELEMENT_TYPE, camp::idx_t NUM_ELEM, VECTOR_LENGTH_TYPE VECTOR_TYPE, camp::idx_t NEW_LENGTH>
  struct VectorNewLengthHelper<Vector<REGISTER_POLICY, ELEMENT_TYPE, NUM_ELEM, VECTOR_TYPE>, NEW_LENGTH> {
      using type = Vector<REGISTER_POLICY, ELEMENT_TYPE, NEW_LENGTH, VECTOR_TYPE>;
  };



  } //namespace internal



}  // namespace RAJA


#endif
