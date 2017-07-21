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

#include "RAJA/internal/MemUtils_CPU.hpp"

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template for use in sequential reduction.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMin<seq_reduce, T>
{
  using my_type = ReduceMin<seq_reduce, T>;

public:


  ReduceMin() = delete;
  //
  // Constructor takes default value (default ctor is disabled).
  //
  RAJA_HOST_DEVICE explicit ReduceMin(T init_m_val) : m_parent(NULL), m_val(init_m_val) {}

  //
  // Copy ctor.
  //
  RAJA_HOST_DEVICE ReduceMin(const ReduceMin<seq_reduce, T>& other)
      : m_parent(other.m_parent ? other.m_parent : &other), m_val(other.m_val)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  RAJA_HOST_DEVICE ~ReduceMin<seq_reduce, T>()
  {
    if (m_parent) {
      m_parent->min(m_val);
    }
  }

  //
  // Operator that returns reduced min value.
  //
  RAJA_HOST_DEVICE operator T() { return m_val; }

  //
  // Method that returns reduced min value.
  //
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //
  // Method that updates min value.
  //
  RAJA_HOST_DEVICE ReduceMin<seq_reduce, T>& min(T rhs)
  {
    m_val = RAJA_MIN(m_val, rhs);
    return *this;
  }

  RAJA_HOST_DEVICE const ReduceMin<seq_reduce, T>& min(T rhs) const
  {
    m_val = RAJA_MIN(m_val, rhs);
    return *this;
  }

private:

  const my_type* m_parent;
  mutable T m_val;
};

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template for use in sequential reduction.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMinLoc<seq_reduce, T>
{
  using my_type = ReduceMinLoc<seq_reduce, T>;

public:

  ReduceMinLoc() = delete;
  //
  // Constructor takes default value (default ctor is disabled).
  //
  RAJA_HOST_DEVICE explicit ReduceMinLoc(T init_m_val, Index_type init_loc)
      : m_parent(NULL), m_val(init_m_val), loc(init_loc)
  {
  }

  //
  // Copy ctor.
  //
  RAJA_HOST_DEVICE ReduceMinLoc(const ReduceMinLoc<seq_reduce, T>& other)
      : m_parent(other.m_parent ? other.m_parent : &other),
        m_val(other.m_val),
        loc(other.loc)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  RAJA_HOST_DEVICE ~ReduceMinLoc<seq_reduce, T>()
  {
    if (m_parent) {
      m_parent->minloc(m_val, loc);
    }
  }

  //
  // Operator that returns reduced min value.
  //
  RAJA_HOST_DEVICE operator T() { return m_val; }

  //
  // Method that returns reduced min value.
  //
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //
  // Method that returns index corresponding to reduced min value.
  //
  RAJA_HOST_DEVICE Index_type getLoc() { return loc; }

  //
  // Method that updates min and index value.
  //
  RAJA_HOST_DEVICE ReduceMinLoc<seq_reduce, T>& minloc(T rhs, Index_type idx)
  {
    if (rhs < m_val) {
      m_val = rhs;
      loc = idx;
    }
    return *this;
  }

  RAJA_HOST_DEVICE const ReduceMinLoc<seq_reduce, T>& minloc(T rhs, Index_type idx) const
  {
    if (rhs < m_val) {
      m_val = rhs;
      loc = idx;
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //

  const my_type* m_parent;

  mutable T m_val;
  mutable Index_type loc;
};

/*!
 ******************************************************************************
 *
 * \brief  Max reducer class template for use in sequential reduction.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMax<seq_reduce, T>
{
  using my_type = ReduceMax<seq_reduce, T>;

public:

  ReduceMax() = delete;

  //
  // Constructor takes default value (default ctor is disabled).
  //
  RAJA_HOST_DEVICE explicit ReduceMax(T init_m_val) : m_parent(NULL), m_val(init_m_val) {}

  //
  // Copy ctor.
  //
  RAJA_HOST_DEVICE ReduceMax(const ReduceMax<seq_reduce, T>& other)
      : m_parent(other.m_parent ? other.m_parent : &other), m_val(other.m_val)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  RAJA_HOST_DEVICE ~ReduceMax<seq_reduce, T>()
  {
    if (m_parent) {
      m_parent->max(m_val);
    }
  }

  //
  // Operator that returns reduced max value.
  //
  RAJA_HOST_DEVICE operator T() { return m_val; }

  //
  // Method that returns reduced max value.
  //
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //
  // Method that updates max value.
  //
  RAJA_HOST_DEVICE ReduceMax<seq_reduce, T>& max(T rhs)
  {
    m_val = RAJA_MAX(rhs, m_val);
    return *this;
  }

  RAJA_HOST_DEVICE const ReduceMax<seq_reduce, T>& max(T rhs) const
  {
    m_val = RAJA_MAX(rhs, m_val);
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  const my_type* m_parent;

  mutable T m_val;
};

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in sequential reduction.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMaxLoc<seq_reduce, T>
{
  using my_type = ReduceMaxLoc<seq_reduce, T>;

public:

  ReduceMaxLoc() = delete;

  //
  // Constructor takes default value (default ctor is disabled).
  //
  RAJA_HOST_DEVICE explicit ReduceMaxLoc(T init_m_val, Index_type init_loc)
      : m_parent(NULL), m_val(init_m_val), loc(init_loc)
  {
  }

  //
  // Copy ctor.
  //
  RAJA_HOST_DEVICE ReduceMaxLoc(const ReduceMaxLoc<seq_reduce, T>& other)
      : m_parent(other.m_parent ? other.m_parent : &other),
        m_val(other.m_val),
        loc(other.loc)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  RAJA_HOST_DEVICE ~ReduceMaxLoc<seq_reduce, T>()
  {
    if (m_parent) {
      m_parent->maxloc(m_val, loc);
    }
  }

  //
  // Operator that returns reduced max value.
  //
  RAJA_HOST_DEVICE operator T() { return m_val; }

  //
  // Method that returns reduced max value.
  //
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //
  // Method that returns index corresponding to reduced max value.
  //
  RAJA_HOST_DEVICE Index_type getLoc() { return loc; }

  //
  // Method that updates max and index value.
  //
  RAJA_HOST_DEVICE ReduceMaxLoc<seq_reduce, T>& maxloc(T rhs, Index_type idx)
  {
    if (rhs > m_val) {
      m_val = rhs;
      loc = idx;
    }
    return *this;
  }

  RAJA_HOST_DEVICE const ReduceMaxLoc<seq_reduce, T>& maxloc(T rhs, Index_type idx) const
  {
    if (rhs > m_val) {
      m_val = rhs;
      loc = idx;
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //

  const my_type* m_parent;

  mutable T m_val;
  mutable Index_type loc;
};

/*!
 ******************************************************************************
 *
 * \brief  Sum reducer class template for use in sequential reduction.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceSum<seq_reduce, T>
{
  using my_type = ReduceSum<seq_reduce, T>;

public:

  ReduceSum() = delete;

  //
  // Constructor takes default value (default ctor is disabled).
  //
  RAJA_HOST_DEVICE explicit ReduceSum(T init_m_val, T initializer = 0)
      : m_parent(NULL), m_val(init_m_val), m_custom_init(initializer)
  {
  }

  //
  // Copy ctor.
  //
  RAJA_HOST_DEVICE ReduceSum(const ReduceSum<seq_reduce, T>& other)
      : m_parent(other.m_parent ? other.m_parent : &other),
        m_val(other.m_custom_init),
        m_custom_init(other.m_custom_init)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  RAJA_HOST_DEVICE ~ReduceSum<seq_reduce, T>()
  {
    if (m_parent) {
      *m_parent += m_val;
    }
  }

  //
  // Operator that returns reduced sum value.
  //
  RAJA_HOST_DEVICE operator T() { return m_val; }

  //
  // Method that returns reduced sum value.
  //
  RAJA_HOST_DEVICE T get() { return operator T(); }

  //
  // += operator that adds value to sum.
  //
  RAJA_HOST_DEVICE ReduceSum<seq_reduce, T>& operator+=(T rhs)
  {
    this->m_val += rhs;
    return *this;
  }

  RAJA_HOST_DEVICE const ReduceSum<seq_reduce, T>& operator+=(T rhs) const
  {
    this->m_val += rhs;
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  const my_type* m_parent;

  mutable T m_val;
  T m_custom_init;
};

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
