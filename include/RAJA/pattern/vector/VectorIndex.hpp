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
      explicit VectorIndex(index_type index) : m_index(index), m_length(vector_type::s_num_elem) {}


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


  namespace Iterators
  {

  /*
   * Specialization of the numeric_iterator for VectorIndex type.
   *
   *
   *
   */

  template <typename Type,
            typename VectorType,
            typename DifferenceType,
            typename PointerType>
  class numeric_iterator<RAJA::VectorIndex<Type, VectorType>, DifferenceType, PointerType>
  {
  public:
    using value_type = Type;
    using vector_type = VectorType;
    using difference_type = DifferenceType;
    using pointer = PointerType;
    using reference = value_type&;
    using iterator_category = std::random_access_iterator_tag;

    RAJA_HOST_DEVICE constexpr numeric_iterator() : val(0) {}
    RAJA_HOST_DEVICE constexpr numeric_iterator(const difference_type& rhs)
        : val(rhs)
    {
    }
    RAJA_HOST_DEVICE constexpr numeric_iterator(const numeric_iterator& rhs)
        : val(rhs.val)
    {
    }

    RAJA_HOST_DEVICE inline DifferenceType get_stride() const { return vector_type::s_num_elem; }

    RAJA_HOST_DEVICE inline bool operator==(const numeric_iterator& rhs) const
    {
      return val == rhs.val;
    }
    RAJA_HOST_DEVICE inline bool operator!=(const numeric_iterator& rhs) const
    {
      return val != rhs.val;
    }
    RAJA_HOST_DEVICE inline bool operator>(const numeric_iterator& rhs) const
    {
      return val > rhs.val;
    }
    RAJA_HOST_DEVICE inline bool operator<(const numeric_iterator& rhs) const
    {
      return val < rhs.val;
    }
    RAJA_HOST_DEVICE inline bool operator>=(const numeric_iterator& rhs) const
    {
      return val >= rhs.val;
    }
    RAJA_HOST_DEVICE inline bool operator<=(const numeric_iterator& rhs) const
    {
      return val <= rhs.val;
    }

    RAJA_HOST_DEVICE inline numeric_iterator& operator++()
    {
      ++val;
      return *this;
    }
    RAJA_HOST_DEVICE inline numeric_iterator& operator--()
    {
      --val;
      return *this;
    }
    RAJA_HOST_DEVICE inline numeric_iterator operator++(int)
    {
      numeric_iterator tmp(*this);
      ++val;
      return tmp;
    }
    RAJA_HOST_DEVICE inline numeric_iterator operator--(int)
    {
      numeric_iterator tmp(*this);
      --val;
      return tmp;
    }

    RAJA_HOST_DEVICE inline numeric_iterator& operator+=(
        const difference_type& rhs)
    {
      val += rhs;
      return *this;
    }
    RAJA_HOST_DEVICE inline numeric_iterator& operator-=(
        const difference_type& rhs)
    {
      val -= rhs;
      return *this;
    }
    RAJA_HOST_DEVICE inline numeric_iterator& operator+=(
        const numeric_iterator& rhs)
    {
      val += rhs.val;
      return *this;
    }
    RAJA_HOST_DEVICE inline numeric_iterator& operator-=(
        const numeric_iterator& rhs)
    {
      val -= rhs.val;
      return *this;
    }

    RAJA_HOST_DEVICE inline difference_type operator+(
        const numeric_iterator& rhs) const
    {
      return val + rhs.val;
    }
    RAJA_HOST_DEVICE inline difference_type operator-(
        const numeric_iterator& rhs) const
    {
      return val - rhs.val;
    }
    RAJA_HOST_DEVICE inline numeric_iterator operator+(
        const difference_type& rhs) const
    {
      return numeric_iterator(val + rhs);
    }
    RAJA_HOST_DEVICE inline numeric_iterator operator-(
        const difference_type& rhs) const
    {
      return numeric_iterator(val - rhs);
    }
    RAJA_HOST_DEVICE friend constexpr numeric_iterator operator+(
        difference_type lhs,
        const numeric_iterator& rhs)
    {
      return numeric_iterator(lhs + rhs.val);
    }
    RAJA_HOST_DEVICE friend constexpr numeric_iterator operator-(
        difference_type lhs,
        const numeric_iterator& rhs)
    {
      return numeric_iterator(lhs - rhs.val);
    }

    RAJA_HOST_DEVICE inline value_type operator*() const
    {
      return value_type(val);
    }
    RAJA_HOST_DEVICE inline value_type operator->() const
    {
      return value_type(val);
    }
    RAJA_HOST_DEVICE constexpr value_type operator[](difference_type rhs) const
    {
      return value_type(val + rhs);
    }

  private:
    difference_type val;
  };
  } //namespace Iterators

}  // namespace RAJA


#endif
