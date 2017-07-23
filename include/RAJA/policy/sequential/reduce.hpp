/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for
 *          sequential execution.
 *
 *          These methods should work on any platform.
 *
 ******************************************************************************
 */

#ifndef RAJA_reduce_sequential_HPP
#define RAJA_reduce_sequential_HPP

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

#include "RAJA/util/types.hpp"
#include "RAJA/pattern/reduce.hpp"
#include "RAJA/policy/sequential/policy.hpp"

namespace RAJA
{

/*!
 **************************************************************************
 *
 * \brief  Min reducer class template for use in sequential execution.
 *
 **************************************************************************
 */
template <typename T> class ReduceMin<seq_reduce, T> {
  static constexpr const RAJA::reduce::min<T> Reduce{};

public:
  //! prohibit compiler-generated default ctor
  ReduceMin() = delete;

  //! prohibit compiler-generated copy assignment
  ReduceMin &operator=(const ReduceMin &) = delete;

  //! compiler-generated move constructor
  RAJA_HOST_DEVICE ReduceMin(ReduceMin &&) = default;

  //! compiler-generated move assignment
  RAJA_HOST_DEVICE ReduceMin &operator=(ReduceMin &&) = default;

  //! constructor requires a default value for the reducer
  RAJA_HOST_DEVICE explicit ReduceMin(T init_val)
      : m_parent(nullptr), m_val(init_val) {}

  //! create a copy of the reducer
  /*!
   * keep parent the same if non-null or set to current
   */
  RAJA_HOST_DEVICE ReduceMin(const ReduceMin &other)
      : m_parent(other.m_parent ? other.m_parent : &other), m_val(other.m_val) {
  }

  //! Destructor folds value into parent object.
  RAJA_HOST_DEVICE ~ReduceMin() {
    if (m_parent) {
      Reduce(m_parent->m_val, m_val);
    }
  }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE operator T() { return m_val; }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE const ReduceMin &min(T rhs) const {
    Reduce(m_val, rhs);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE ReduceMin &min(T rhs) {
    Reduce(m_val, rhs);
    return *this;
  }

private:
  //! pointer to the parent ReduceMin object
  const ReduceMin *m_parent;
  mutable T m_val;
};

/*!
 **************************************************************************
 *
 * \brief  MinLoc reducer class template for use in sequential execution.
 *
 **************************************************************************
 */
template <typename T> class ReduceMinLoc<seq_reduce, T> {
  static constexpr const RAJA::reduce::minloc<T, Index_type> Reduce{};

public:
  //! prohibit compiler-generated default ctor
  ReduceMinLoc() = delete;

  //! prohibit compiler-generated copy assignment
  ReduceMinLoc &operator=(const ReduceMinLoc &) = delete;

  //! compiler-generated move constructor
  RAJA_HOST_DEVICE ReduceMinLoc(ReduceMinLoc &&) = default;

  //! compiler-generated move assignment
  RAJA_HOST_DEVICE ReduceMinLoc &operator=(ReduceMinLoc &&) = default;

  //! constructor requires a default value for the reducer
  RAJA_HOST_DEVICE explicit ReduceMinLoc(T init_val, Index_type init_idx)
      : m_parent(nullptr), m_val(init_val), m_idx(init_idx) {}

  //! create a copy of the reducer
  /*!
   * keep parent the same if non-null or set to current
   */
  RAJA_HOST_DEVICE ReduceMinLoc(const ReduceMinLoc &other)
      : m_parent(other.m_parent ? other.m_parent : &other), m_val(other.m_val),
        m_idx(other.m_idx) {}

  //! Destructor folds value into parent object.
  RAJA_HOST_DEVICE ~ReduceMinLoc() {
    if (m_parent) {
      Reduce(m_parent->m_val, m_parent->m_idx, m_val, m_idx);
    }
  }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE operator T() { return m_val; }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //! return the index location of the minimum value
  /*!
   *  \return the index location
   */
  RAJA_HOST_DEVICE Index_type getLoc() { return m_idx; }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE const ReduceMinLoc &minloc(T rhs, Index_type idx) const {
    Reduce(m_val, m_idx, rhs, idx);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE ReduceMinLoc &minloc(T rhs, Index_type idx) {
    Reduce(m_val, m_idx, rhs, idx);
    return *this;
  }

private:
  //! pointer to the parent ReduceMinLoc object
  const ReduceMinLoc *m_parent;
  mutable T m_val;
  mutable Index_type m_idx;
};

/*!
 **************************************************************************
 *
 * \brief  Max reducer class template for use in sequential execution.
 *
 **************************************************************************
 */
template <typename T> class ReduceMax<seq_reduce, T> {
  static constexpr const RAJA::reduce::max<T> Reduce{};

public:
  //! prohibit compiler-generated default ctor
  ReduceMax() = delete;

  //! prohibit compiler-generated copy assignment
  ReduceMax &operator=(const ReduceMax &) = delete;

  //! compiler-generated move constructor
  RAJA_HOST_DEVICE ReduceMax(ReduceMax &&) = default;

  //! compiler-generated move assignment
  RAJA_HOST_DEVICE ReduceMax &operator=(ReduceMax &&) = default;

  //! constructor requires a default value for the reducer
  RAJA_HOST_DEVICE explicit ReduceMax(T init_val)
      : m_parent(nullptr), m_val(init_val) {}

  //! create a copy of the reducer
  /*!
   * keep parent the same if non-null or set to current
   */
  RAJA_HOST_DEVICE ReduceMax(const ReduceMax &other)
      : m_parent(other.m_parent ? other.m_parent : &other), m_val(other.m_val) {
  }

  //! Destructor folds value into parent object.
  RAJA_HOST_DEVICE ~ReduceMax() {
    if (m_parent) {
      Reduce(m_parent->m_val, m_val);
    }
  }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE operator T() { return m_val; }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE const ReduceMax &max(T rhs) const {
    Reduce(m_val, rhs);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE ReduceMax &max(T rhs) {
    Reduce(m_val, rhs);
    return *this;
  }

private:
  //! pointer to the parent ReduceMax object
  const ReduceMax *m_parent;
  mutable T m_val;
};

/*!
 **************************************************************************
 *
 * \brief  Sum reducer class template for use in sequential execution.
 *
 **************************************************************************
 */
template <typename T> class ReduceSum<seq_reduce, T> {
  static constexpr const RAJA::reduce::sum<T> Reduce{};

public:
  //! prohibit compiler-generated default ctor
  ReduceSum() = delete;

  //! prohibit compiler-generated copy assignment
  ReduceSum &operator=(const ReduceSum &) = delete;

  //! compiler-generated move constructor
  RAJA_HOST_DEVICE ReduceSum(ReduceSum &&) = default;

  //! compiler-generated move assignment
  RAJA_HOST_DEVICE ReduceSum &operator=(ReduceSum &&) = default;

  //! constructor requires a default value for the reducer
  RAJA_HOST_DEVICE explicit ReduceSum(T init_val, T initializer = T())
      : m_parent(nullptr), m_val(init_val), m_custom_init(initializer) {}

  //! create a copy of the reducer
  /*!
   * keep parent the same if non-null or set to current
   */
  RAJA_HOST_DEVICE ReduceSum(const ReduceSum &other)
      : m_parent(other.m_parent ? other.m_parent : &other),
        m_val(other.m_custom_init), m_custom_init(other.m_custom_init) {}

  //! Destructor folds value into parent object.
  RAJA_HOST_DEVICE ~ReduceSum() {
    if (m_parent) {
      Reduce(m_parent->m_val, m_val);
    }
  }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE operator T() { return m_val; }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE const ReduceSum &operator+=(T rhs) const {
    Reduce(m_val, rhs);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE ReduceSum &operator+=(T rhs) {
    Reduce(m_val, rhs);
    return *this;
  }

private:
  //! pointer to the parent ReduceSum object
  const ReduceSum *m_parent;
  mutable T m_val;
  const T m_custom_init;
};

/*!
 **************************************************************************
 *
 * \brief  MaxLoc reducer class template for use in sequential execution.
 *
 **************************************************************************
 */
template <typename T> class ReduceMaxLoc<seq_reduce, T> {
  static constexpr const RAJA::reduce::maxloc<T, Index_type> Reduce{};

public:
  //! prohibit compiler-generated default ctor
  ReduceMaxLoc() = delete;

  //! prohibit compiler-generated copy assignment
  ReduceMaxLoc &operator=(const ReduceMaxLoc &) = delete;

  //! compiler-generated move constructor
  RAJA_HOST_DEVICE ReduceMaxLoc(ReduceMaxLoc &&) = default;

  //! compiler-generated move assignment
  RAJA_HOST_DEVICE ReduceMaxLoc &operator=(ReduceMaxLoc &&) = default;

  //! constructor requires a default value for the reducer
  RAJA_HOST_DEVICE explicit ReduceMaxLoc(T init_val, Index_type init_idx)
      : m_parent(nullptr), m_val(init_val), m_idx(init_idx) {}

  //! create a copy of the reducer
  /*!
   * keep parent the same if non-null or set to current
   */
  RAJA_HOST_DEVICE ReduceMaxLoc(const ReduceMaxLoc &other)
      : m_parent(other.m_parent ? other.m_parent : &other), m_val(other.m_val),
        m_idx(other.m_idx) {}

  //! Destructor folds value into parent object.
  RAJA_HOST_DEVICE ~ReduceMaxLoc() {
    if (m_parent) {
      Reduce(m_parent->m_val, m_parent->m_idx, m_val, m_idx);
    }
  }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE operator T() { return m_val; }

  //! return the reduced min value.
  /*!
   *  \return the calculated reduced value
   */
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //! return the index location of the maximum value
  /*!
   *  \return the index location
   */
  RAJA_HOST_DEVICE Index_type getLoc() { return m_idx; }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE const ReduceMaxLoc &maxloc(T rhs, Index_type idx) const {
    Reduce(m_val, m_idx, rhs, idx);
    return *this;
  }

  //! reducer function; updates the current instance's state
  /*!
   * Assumes each thread has its own copy of the object.
   */
  RAJA_HOST_DEVICE ReduceMaxLoc &maxloc(T rhs, Index_type idx) {
    Reduce(m_val, m_idx, rhs, idx);
    return *this;
  }

private:
  //! pointer to the parent ReduceMaxLoc object
  const ReduceMaxLoc *m_parent;
  mutable T m_val;
  mutable Index_type m_idx;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
