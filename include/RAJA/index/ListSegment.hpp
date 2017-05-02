/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining list segment classes.
 *
 ******************************************************************************
 */

#ifndef RAJA_ListSegment_HXX
#define RAJA_ListSegment_HXX

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

#include "RAJA/index/BaseSegment.hpp"

#if defined(RAJA_ENABLE_CUDA)
#include "RAJA/policy/cuda/raja_cudaerrchk.hpp"
#endif

#include <algorithm>
#include <iosfwd>
#include <type_traits>
#include <vector>

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Class representing an arbitrary collection of indices.
 *
 *         Length indicates number of indices in index array.
 *         Traversal executes as:
 *            for (i = 0; i < getLength(); ++i) {
 *               expression using m_indx[i] as array index.
 *            }
 *
 ******************************************************************************
 */
class ListSegment : public BaseSegment
{
public:
  ///
  /// Construct list segment from given array with specified length.
  ///
  /// By default the ctor performs deep copy of array elements.
  /// If 'Unowned' is passed as last argument, the constructed object
  /// does not own the segment data and will hold a pointer to given data.
  /// In this case, caller must manage object lifetimes properly.
  ///
  ListSegment(const Index_type* indx,
              Index_type len,
              IndexOwnership indx_own = Owned);

  ///
  /// Construct list segment from arbitrary object holding
  /// indices using a deep copy of given data.
  ///
  /// The object must provide methods: begin(), end(), size().
  ///
  template <typename T>
  explicit ListSegment(const T& indx);

  ///
  /// Copy-constructor for list segment.
  ///
  ListSegment(const ListSegment& other);

  ///
  /// Copy-assignment for list segment.
  ///
  ListSegment& operator=(const ListSegment& rhs);

  ///
  /// Move-constructor for list segment.
  ///
  ListSegment(ListSegment&& other)
    : BaseSegment(std::move(other)),
      m_indx(other.m_indx),
      m_len(other.m_len),
      m_indx_own(other.m_indx_own)
  {
    other.m_indx = nullptr;
  }

  ///
  /// Move-assignment for list segment.
  ///
  ListSegment& operator=(ListSegment&& rhs)
  {
    if (this != &rhs) {
      BaseSegment::operator=(std::move(rhs));
      m_indx = rhs.m_indx;
      m_len = rhs.m_len;
      m_indx_own = rhs.m_indx_own;
      rhs.m_indx = nullptr;
    }
    return *this;
  }

  ///
  /// Destroy segment including its contents.
  ///
  ~ListSegment();

  ///
  /// Swap function for copy-and-swap idiom.
  ///
  void swap(ListSegment& other);

  ///
  ///  Return const pointer to array of indices in segment.
  ///
  const Index_type* getIndex() const { return m_indx; }

  ///
  ///  Return length of list segment (# indices).
  ///
  Index_type getLength() const { return m_len; }

  using iterator = Index_type*;

  ///
  /// Get an iterator to the end.
  ///
  iterator end() const { return m_indx + m_len; }

  ///
  /// Get an iterator to the beginning.
  ///
  Index_type* begin() const { return m_indx; }

  ///
  /// Return the number of elements in the range.
  ///
  Index_type size() const { return m_len; }

  ///
  /// Return enum value indicating whether segment object owns the data
  /// representing its indices.
  ///
  IndexOwnership getIndexOwnership() const { return m_indx_own; }

  ///
  /// Return true if given array of indices is same as indices described
  /// by this segment object; else false.
  ///
  bool indicesEqual(const Index_type* indx, Index_type len) const;

  ///
  /// Equality operator returns true if segments are equal; else false.
  ///
  bool operator==(const ListSegment& other) const
  {
    return (indicesEqual(other.m_indx, other.m_len));
  }

  ///
  /// Inequality operator returns true if segments are not equal, else false.
  ///
  bool operator!=(const ListSegment& other) const
  {
    return (!(*this == other));
  }

  ///
  /// Equality operator returns true if segments are equal; else false.
  /// (Implements pure virtual method in BaseSegment class).
  ///
  bool operator==(const BaseSegment& other) const
  {
    const ListSegment* o_ptr = dynamic_cast<const ListSegment*>(&other);
    if (o_ptr) {
      return (*this == *o_ptr);
    } else {
      return false;
    }
  }

  ///
  /// Inequality operator returns true if segments are not equal; else false.
  /// (Implements pure virtual method in BaseSegment class).
  ///
  bool operator!=(const BaseSegment& other) const
  {
    return (!(*this == other));
  }

  ///
  /// Print segment data to given output stream.
  ///
  void print(std::ostream& os) const;

private:
  //
  // The default ctor is not implemented.
  //
  ListSegment();

  //
  // Initialize segment data properly based on whether object
  // owns the index data.
  //
  void initIndexData(const Index_type* indx,
                     Index_type len,
                     IndexOwnership indx_own);

  Index_type* RAJA_RESTRICT m_indx;
  Index_type m_len;
  IndexOwnership m_indx_own;
};

/*!
 ******************************************************************************
 *
 *  \brief Implementation of generic constructor template.
 *
 ******************************************************************************
 */
template <typename T>
ListSegment::ListSegment(const T& indx)
    : BaseSegment(_ListSeg_), m_indx(0), m_len(indx.size()), m_indx_own(Unowned)
{
  if (!indx.empty()) {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaMallocManaged((void**)&m_indx,
                                 m_len * sizeof(Index_type),
                                 cudaMemAttachGlobal));
    using T_TYPE = typename std::remove_reference<decltype(indx[0])>::type;
    if (sizeof(T_TYPE) == sizeof(Index_type) && std::is_integral<T_TYPE>::value
        && std::is_same<std::vector<T_TYPE>, typename std::decay<T>::type>::value) {
      // this will bitwise copy signed or unsigned integral types
      cudaErrchk(cudaMemcpy(m_indx, &indx[0], m_len * sizeof(Index_type), cudaMemcpyDefault));
    } else {
      cudaErrchk(cudaDeviceSynchronize());
      std::copy(indx.begin(), indx.end(), m_indx);
    }
#else
    m_indx = new Index_type[indx.size()];
    std::copy(indx.begin(), indx.end(), m_indx);
#endif
    m_indx_own = Owned;
  }
}

}  // closing brace for RAJA namespace

/*!
 *  Specialization of std swap method.
 */
namespace std
{

template <>
RAJA_INLINE void swap(RAJA::ListSegment& a, RAJA::ListSegment& b)
{
  a.swap(b);
}
}

#endif  // closing endif for header file include guard
