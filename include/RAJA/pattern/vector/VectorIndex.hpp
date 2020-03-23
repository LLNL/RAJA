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

#ifndef RAJA_pattern_vector_vectorindex_HPP
#define RAJA_pattern_vector_vectorindex_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"


namespace RAJA
{



  template<typename IDX, typename VECTOR_TYPE>
  class VectorIndex {
    public:
      using value_type = strip_index_type_t<IDX>;
      using index_type = IDX;
      using vector_type = VECTOR_TYPE;


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      VectorIndex() : m_index(index_type(0)), m_length(vector_type::s_num_elem) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      explicit VectorIndex(index_type value) : m_index(value), m_length(vector_type::s_num_elem) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      VectorIndex(index_type value, size_t length) : m_index(value), m_length(length) {}


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      index_type const &operator*() const {
        return m_index;
      }

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      size_t size() const {
        return m_length;
      }

    private:
      index_type m_index;
      size_t m_length;
  };


  namespace internal{


    /* Partial specialization for the strip_index_type_t helper in
       IndexValue.hpp
    */
    template<typename IDX, typename VECTOR_TYPE>
    struct StripIndexTypeT<VectorIndex<IDX, VECTOR_TYPE>>
    {
        using type = typename VectorIndex<IDX, VECTOR_TYPE>::value_type;
    };


    // Helper that strips the Vector type from an argument
    template<typename ARG>
    struct VectorIndexTraits {
        using arg_type = ARG;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        bool isVectorIndex(){
          return false;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        arg_type const &stripVector(arg_type const &arg){
          return arg;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        camp::idx_t size(arg_type const &){
          return 1;
        }
    };

    template<typename IDX, typename VECTOR_TYPE>
    struct VectorIndexTraits<VectorIndex<IDX, VECTOR_TYPE>> {
        using arg_type = IDX;

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        bool isVectorIndex(){
          return true;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        arg_type const &stripVector(VectorIndex<IDX, VECTOR_TYPE> const &arg){
          return *arg;
        }

        RAJA_INLINE
        RAJA_HOST_DEVICE
        static
        constexpr
        camp::idx_t size(VectorIndex<IDX, VECTOR_TYPE> const &arg){
          return arg.size();
        }
    };

    /*
     * Returns vector size of argument.
     *
     * For scalars, always returns 1.
     *
     * For VectorIndex types, returns the number of vector lanes.
     */
    template<typename ARG>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    bool isVectorIndex()
    {
      return VectorIndexTraits<ARG>::isVectorIndex();
    }

    template<typename ARG>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    auto stripVectorIndex(ARG const &arg) ->
    typename VectorIndexTraits<ARG>::arg_type const &
    {
      return VectorIndexTraits<ARG>::stripVector(arg);
    }

    /*
     * Returns vector size of argument.
     *
     * For scalars, always returns 1.
     *
     * For VectorIndex types, returns the number of vector lanes.
     */
    template<typename ARG>
    RAJA_INLINE
    RAJA_HOST_DEVICE
    constexpr
    camp::idx_t getVectorSize(ARG const &arg)
    {
      return VectorIndexTraits<ARG>::size(arg);
    }

    /*
     * Lambda<N, Seg<X>>  overload that matches VectorIndex types, and properly
     * includes the vector length with them
     */
    template<typename IDX, typename VECTOR_TYPE, camp::idx_t id>
    struct LambdaSegExtractor<VectorIndex<IDX, VECTOR_TYPE>, id>
    {

      template<typename Data>
      RAJA_HOST_DEVICE
      RAJA_INLINE
      constexpr
      static VectorIndex<IDX, VECTOR_TYPE> extract(Data &&data)
      {
        return VectorIndex<IDX, VECTOR_TYPE>(
            camp::get<id>(data.segment_tuple).begin()[camp::get<id>(data.offset_tuple)],
            camp::get<id>(data.vector_sizes));
      }

    };

  } // namespace internal



}  // namespace RAJA


#endif
