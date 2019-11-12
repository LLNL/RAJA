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
      using index_type = IDX;
      using vector_type = VECTOR_TYPE;

      RAJA_INLINE
      RAJA_HOST_DEVICE
      constexpr
      VectorIndex() : m_index(0), m_length(vector_type::s_num_elem) {}

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

}  // namespace RAJA


#endif
