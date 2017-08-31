/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA iterator constructs.
 *
 ******************************************************************************
 */


#ifndef RAJA_ITERATORS_HPP
#define RAJA_ITERATORS_HPP

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/config.hpp"
#include "RAJA/util/defines.hpp"
#include "RAJA/util/types.hpp"

#include <iterator>
#include <type_traits>
#include <utility>

namespace RAJA
{
namespace Iterators
{

// Containers

template <typename Type,
          typename DifferenceType = std::ptrdiff_t,
          typename PointerType = Type*>
struct base_iterator {

  using value_type = Type;
  using difference_type = DifferenceType;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::random_access_iterator_tag;

  RAJA_HOST_DEVICE constexpr base_iterator() : val(0) {}
  RAJA_HOST_DEVICE constexpr base_iterator(Type rhs) : val(rhs) {}
  RAJA_HOST_DEVICE constexpr base_iterator(const base_iterator& rhs)
      : val(rhs.val)
  {
  }

  RAJA_HOST_DEVICE inline bool operator==(const base_iterator& rhs) const
  {
    return val == rhs.val;
  }
  RAJA_HOST_DEVICE inline bool operator!=(const base_iterator& rhs) const
  {
    return val != rhs.val;
  }
  RAJA_HOST_DEVICE inline bool operator>(const base_iterator& rhs) const
  {
    return val > rhs.val;
  }
  RAJA_HOST_DEVICE inline bool operator<(const base_iterator& rhs) const
  {
    return val < rhs.val;
  }
  RAJA_HOST_DEVICE inline bool operator>=(const base_iterator& rhs) const
  {
    return val >= rhs.val;
  }
  RAJA_HOST_DEVICE inline bool operator<=(const base_iterator& rhs) const
  {
    return val <= rhs.val;
  }

protected:
  Type val;
};

template <typename Type = Index_type,
          typename DifferenceType = Index_type,
          typename PointerType = Type*>
class numeric_iterator : public base_iterator<Type, DifferenceType>
{
  using base = base_iterator<Type, DifferenceType>;

public:
  using value_type = typename base::value_type;
  using difference_type = typename base::difference_type;
  using pointer = typename base::pointer;
  using reference = typename base::reference;
  using iterator_category = typename base::iterator_category;

  RAJA_HOST_DEVICE constexpr numeric_iterator() : base(0) {}
  RAJA_HOST_DEVICE constexpr numeric_iterator(const Type& rhs) : base(rhs) {}
  RAJA_HOST_DEVICE constexpr numeric_iterator(const numeric_iterator& rhs)
      : base(rhs.val)
  {
  }

  RAJA_HOST_DEVICE inline numeric_iterator& operator++()
  {
    ++base::val;
    return *this;
  }
  RAJA_HOST_DEVICE inline numeric_iterator& operator--()
  {
    --base::val;
    return *this;
  }
  RAJA_HOST_DEVICE inline numeric_iterator operator++(int)
  {
    numeric_iterator tmp(*this);
    ++base::val;
    return tmp;
  }
  RAJA_HOST_DEVICE inline numeric_iterator operator--(int)
  {
    numeric_iterator tmp(*this);
    --base::val;
    return tmp;
  }

  RAJA_HOST_DEVICE inline numeric_iterator& operator+=(
      const difference_type& rhs)
  {
    base::val += rhs;
    return *this;
  }
  RAJA_HOST_DEVICE inline numeric_iterator& operator-=(
      const difference_type& rhs)
  {
    base::val -= rhs;
    return *this;
  }
  RAJA_HOST_DEVICE inline numeric_iterator& operator+=(
      const numeric_iterator& rhs)
  {
    base::val += rhs.val;
    return *this;
  }
  RAJA_HOST_DEVICE inline numeric_iterator& operator-=(
      const numeric_iterator& rhs)
  {
    base::val -= rhs.val;
    return *this;
  }

  RAJA_HOST_DEVICE inline difference_type operator+(
      const numeric_iterator& rhs) const
  {
    return static_cast<difference_type>(base::val)
           + static_cast<difference_type>(rhs.val);
  }
  RAJA_HOST_DEVICE inline difference_type operator-(
      const numeric_iterator& rhs) const
  {
    return static_cast<difference_type>(base::val)
           - static_cast<difference_type>(rhs.val);
  }
  RAJA_HOST_DEVICE inline numeric_iterator operator+(
      const difference_type& rhs) const
  {
    return numeric_iterator(base::val + rhs);
  }
  RAJA_HOST_DEVICE inline numeric_iterator operator-(
      const difference_type& rhs) const
  {
    return numeric_iterator(base::val - rhs);
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

  RAJA_HOST_DEVICE inline Type operator*() const { return base::val; }
  RAJA_HOST_DEVICE inline Type operator->() const { return base::val; }
  RAJA_HOST_DEVICE constexpr Type operator[](difference_type rhs) const
  {
    return base::val + rhs;
  }
};

template <typename Type = Index_type,
          typename DifferenceType = Index_type,
          typename PointerType = Type*>
class strided_numeric_iterator : public base_iterator<Type, DifferenceType>
{
  using base = base_iterator<Type, DifferenceType>;

public:
  using value_type = typename base::value_type;
  using difference_type = typename base::difference_type;
  using pointer = typename base::pointer;
  using reference = typename base::reference;
  using iterator_category = typename base::iterator_category;

  RAJA_HOST_DEVICE constexpr strided_numeric_iterator() : base(0), stride(1) {}
  RAJA_HOST_DEVICE constexpr strided_numeric_iterator(const Type& rhs,
                                                      DifferenceType stride = 1)
      : base(rhs), stride(stride)
  {
  }
  RAJA_HOST_DEVICE constexpr strided_numeric_iterator(
      const strided_numeric_iterator& rhs)
      : base(rhs.val), stride(rhs.stride)
  {
  }

  RAJA_HOST_DEVICE inline strided_numeric_iterator& operator++()
  {
    base::val += stride;
    return *this;
  }
  RAJA_HOST_DEVICE inline strided_numeric_iterator& operator--()
  {
    base::val -= stride;
    return *this;
  }

  RAJA_HOST_DEVICE inline strided_numeric_iterator& operator+=(
      const difference_type& rhs)
  {
    base::val += rhs * stride;
    return *this;
  }
  RAJA_HOST_DEVICE inline strided_numeric_iterator& operator-=(
      const difference_type& rhs)
  {
    base::val -= rhs * stride;
    return *this;
  }

  RAJA_HOST_DEVICE inline difference_type operator+(
      const strided_numeric_iterator& rhs) const
  {
    return (static_cast<difference_type>(base::val)
            + (static_cast<difference_type>(rhs.val)))
           / stride;
  }
  RAJA_HOST_DEVICE inline difference_type operator-(
      const strided_numeric_iterator& rhs) const
  {
    difference_type diff = (static_cast<difference_type>(base::val)
                            - (static_cast<difference_type>(rhs.val)));

    return (diff % stride) ? (1 + diff / stride) : diff / stride;
  }
  RAJA_HOST_DEVICE inline strided_numeric_iterator operator+(
      const difference_type& rhs) const
  {
    return strided_numeric_iterator(base::val + rhs * stride);
  }
  RAJA_HOST_DEVICE inline strided_numeric_iterator operator-(
      const difference_type& rhs) const
  {
    return strided_numeric_iterator(base::val - rhs * stride);
  }

  // Specialized comparison to allow normal iteration to work on off-stride
  // multiples by adjusting rhs to the nearest *higher* multiple of stride
  RAJA_HOST_DEVICE inline bool operator!=(
      const strided_numeric_iterator& rhs) const
  {
    return (base::val - rhs.val) / stride;
  }
  RAJA_HOST_DEVICE inline bool operator==(
      const strided_numeric_iterator& rhs) const
  {
    return !((base::val - rhs.val) / stride);
  }

  RAJA_HOST_DEVICE inline bool operator>(
      const strided_numeric_iterator& rhs) const
  {
    return base::val * stride > rhs.val * stride;
  }
  RAJA_HOST_DEVICE inline bool operator<(
      const strided_numeric_iterator& rhs) const
  {
    return base::val * stride < rhs.val * stride;
  }
  RAJA_HOST_DEVICE inline bool operator>=(
      const strided_numeric_iterator& rhs) const
  {
    return base::val * stride >= rhs.val * stride;
  }
  RAJA_HOST_DEVICE inline bool operator<=(
      const strided_numeric_iterator& rhs) const
  {
    return base::val * stride <= rhs.val * stride;
  }


  RAJA_HOST_DEVICE inline Type operator*() const { return base::val; }
  RAJA_HOST_DEVICE inline Type operator->() const { return base::val; }
  RAJA_HOST_DEVICE inline Type operator[](difference_type rhs) const
  {
    return base::val + rhs * stride;
  }

private:
  DifferenceType stride;
};

} // closing brace for namespace Iterators

} // closing brace for namespace RAJA

#endif /* RAJA_ITERATORS_HPP */
