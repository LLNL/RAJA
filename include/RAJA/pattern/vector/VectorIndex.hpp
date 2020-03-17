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

  namespace internal{

    struct VectorIndexBase {};


    template<typename FROM>
    struct StripIndexTypeT<FROM, typename std::enable_if<std::is_base_of<VectorIndexBase, FROM>::value>::type>
    {
        using type = typename FROM::value_type;
    };


  }

  template<typename IDX, typename VECTOR_TYPE>
  class VectorIndex : public internal::VectorIndexBase {
    public:
      using value_type = strip_index_type_t<IDX>;
      using index_type = IDX;
      using vector_type = VECTOR_TYPE;


      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      VectorIndex() : m_index(index_type(0)), m_length(vector_type::s_num_elem) {}

      template<typename T>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      explicit VectorIndex(T value) : m_index(index_type(value)), m_length(vector_type::s_num_elem) {}


      template<typename T>
      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      VectorIndex(T value, size_t length) : m_index(index_type(value)), m_length(length) {}

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



  namespace internal
  {

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
            data.vector_sizes[id]);
      }

    };

  } // namespace internal



}  // namespace RAJA


#endif
