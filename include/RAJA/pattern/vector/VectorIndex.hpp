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
    using vector_index_type = RAJA::VectorIndex<Type, VectorType>;
    using difference_type = DifferenceType;
    using pointer = PointerType;
    using reference = value_type&;
    using iterator_category = std::random_access_iterator_tag;

    RAJA_HOST_DEVICE constexpr numeric_iterator() : val(0), length(vector_type::s_num_elem) {}
    RAJA_HOST_DEVICE constexpr numeric_iterator(const difference_type& rhs)
        : val(rhs), length(vector_type::s_num_elem)
    {
    }
    RAJA_HOST_DEVICE constexpr numeric_iterator(const difference_type& rhs, const difference_type& len)
        : val(rhs), length(len)
    {
    }
    RAJA_HOST_DEVICE constexpr numeric_iterator(const numeric_iterator& rhs)
        : val(rhs.val), length(rhs.length)
    {
    }

    RAJA_HOST_DEVICE constexpr inline DifferenceType get_stride() const
    {
      return vector_type::s_num_elem;
    }

    RAJA_HOST_DEVICE inline void set_vector_length(DifferenceType len)
    {
      length = len;
    }

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
      return numeric_iterator(val + rhs, length);
    }
    RAJA_HOST_DEVICE inline numeric_iterator operator-(
        const difference_type& rhs) const
    {
      return numeric_iterator(val - rhs, length);
    }
    RAJA_HOST_DEVICE friend constexpr numeric_iterator operator+(
        difference_type lhs,
        const numeric_iterator& rhs)
    {
      return numeric_iterator(lhs + rhs.val, rhs.length);
    }
    RAJA_HOST_DEVICE friend constexpr numeric_iterator operator-(
        difference_type lhs,
        const numeric_iterator& rhs)
    {
      return numeric_iterator(lhs - rhs.val, rhs.length);
    }

    RAJA_HOST_DEVICE inline vector_index_type operator*() const
    {
      return vector_index_type(val, length);
    }
    RAJA_HOST_DEVICE inline vector_index_type operator->() const
    {
      return vector_index_type(val, length);
    }
    RAJA_HOST_DEVICE constexpr vector_index_type operator[](difference_type rhs) const
    {
      return vector_index_type(val + rhs, length);
    }

  private:
    difference_type val;
    difference_type length;
  };
  } //namespace Iterators

}  // namespace RAJA


#endif
