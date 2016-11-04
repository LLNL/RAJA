/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA reduction templates for OpenMP
 *          OpenMP execution.
 *
 *          These methods should work on any platform that supports OpenMP.
 *
 ******************************************************************************
 */

#ifndef RAJA_forall_omp_HXX
#define RAJA_forall_omp_HXX

#include "RAJA/config.hxx"

#if defined(RAJA_ENABLE_OPENMP)

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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/int_datatypes.hxx"

#include "RAJA/reducers.hxx"

#include "RAJA/MemUtils_CPU.hxx"

#if defined(_OPENMP)
#include <omp.h>
#endif

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Min reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMin<omp_reduce, T>
{
  using my_type = ReduceMin<omp_reduce, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMin(T init_val):
    parent(NULL), val(init_val)
  {
  }

  //
  // Copy ctor.
  //
  ReduceMin(const ReduceMin<omp_reduce, T>& other):
    parent(other.parent ? other.parent : &other),
    val(other.val)
  {
  }

  //
  // Destruction folds value into parent object.
  //
  ~ReduceMin<omp_reduce, T>()
  {
    if (parent) {
#pragma omp critical
      {
        parent->min(val);
      }
    }
  }

  //
  // Operator that returns reduced min value.
  //
  operator T()
  {
    return val;
  }

  //
  // Method that returns reduced min value.
  //
  T get() { return operator T(); }

  //
  // Method that updates min value for current object, assumes each thread
  // has its own copy of the object.
  //
  const ReduceMin<omp_reduce, T>& min(T rhs) const
  {
    val = RAJA_MIN(val, rhs);
    return *this;
  }

  ReduceMin<omp_reduce, T>& min(T rhs) {
    val = RAJA_MIN(val, rhs);
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMin<omp_reduce, T>();

  const my_type * parent;
  mutable T val;
};

/*!
 ******************************************************************************
 *
 * \brief  Min-loc reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMinLoc<omp_reduce, T>
{
  using my_type = ReduceMinLoc<omp_reduce, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMinLoc(T init_val, Index_type init_loc):
    parent(NULL), val(init_val), idx(init_loc)
  {
  }

  //
  // Copy ctor.
  //
  ReduceMinLoc(const ReduceMinLoc<omp_reduce, T>& other):
    parent(other.parent ? other.parent : &other),
    val(other.val),
    idx(other.idx)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMinLoc<omp_reduce, T>()
  {
    if (parent) {
#pragma omp critical
      {
        parent->minloc(val, idx);
      }
    }
  }

  //
  // Operator that returns reduced min value.
  //
  operator T()
  {
    return val;
  }

  //
  // Method that returns reduced min value.
  //
  T get() { return operator T(); }

  //
  // Method that returns index corresponding to reduced min value.
  //
  Index_type getLoc()
  {
    return idx;
  }

  //
  // Method that updates min and index values for current thread.
  //
  const ReduceMinLoc<omp_reduce, T>& minloc(T rhs, Index_type rhs_idx) const
  {
    if (rhs <= val) {
      val = rhs;
      idx = rhs_idx;
    }
    return *this;
  }

  ReduceMinLoc<omp_reduce, T>& minloc(T rhs, Index_type rhs_idx)
  {
    if (rhs <= val) {
      val = rhs;
      idx = rhs_idx;
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMinLoc<omp_reduce, T>();

  const my_type * parent;

  mutable T val;
  mutable Index_type idx;
};

/*!
 ******************************************************************************
 *
 * \brief  Max reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMax<omp_reduce, T>
{
  using my_type = ReduceMax<omp_reduce, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMax(T init_val):
    parent(NULL), val(init_val)
  {
  }

  //
  // Copy ctor.
  //
  ReduceMax(const ReduceMax<omp_reduce, T>& other) :
    parent(other.parent ? other.parent : &other),
    val(other.val)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMax<omp_reduce, T>()
  {
    if (parent) {
#pragma omp critical
      {
          parent->max(val);
      }
    }
  }

  //
  // Operator that returns reduced max value.
  //
  operator T()
  {
    return val;
  }

  //
  // Method that returns reduced max value.
  //
  T get() { return operator T(); }

  //
  // Method that updates max value for current thread.
  //
  const ReduceMax<omp_reduce, T>& max(T rhs) const
  {
    val = RAJA_MAX(val, rhs);
    return *this;
  }

  ReduceMax<omp_reduce, T>& max(T rhs)
  {
    val = RAJA_MAX(val, rhs);
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMax<omp_reduce, T>();

  const my_type * parent;

  mutable T val;
};

/*!
 ******************************************************************************
 *
 * \brief  Max-loc reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceMaxLoc<omp_reduce, T>
{
  using my_type = ReduceMaxLoc<omp_reduce, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceMaxLoc(T init_val, Index_type init_loc):
    parent(NULL), val(init_val), idx(init_loc)
  {
  }

  //
  // Copy ctor.
  //
  ReduceMaxLoc(const ReduceMaxLoc<omp_reduce, T>& other):
    parent(other.parent ? other.parent : &other),
    val(other.val),
    idx(other.idx)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceMaxLoc<omp_reduce, T>()
  {
    if (parent) {
#pragma omp critical
      {
        parent->maxloc(val, idx);
      }
    }
  }

  //
  // Operator that returns reduced max value.
  //
  operator T()
  {
    return val;
  }

  //
  // Method that returns reduced max value.
  //
  T get() { return operator T(); }

  //
  // Method that returns index corresponding to reduced max value.
  //
  Index_type getLoc()
  {
    return idx;
  }

  //
  // Method that updates max and index values for current thread.
  //
  const ReduceMaxLoc<omp_reduce, T>& maxloc(T rhs, Index_type rhs_idx) const
  {
    if (rhs >= val) {
      val = rhs;
      idx = rhs_idx;
    }
    return *this;
  }

  ReduceMaxLoc<omp_reduce, T>& maxloc(T rhs, Index_type rhs_idx)
  {
    if (rhs >= val) {
      val = rhs;
      idx = rhs_idx;
    }
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceMaxLoc<omp_reduce, T>();

  const my_type * parent;

  mutable T val;
  mutable Index_type idx;
};

/*!
 ******************************************************************************
 *
 * \brief  Sum reducer class template for use in OpenMP execution.
 *
 *         For usage example, see reducers.hxx.
 *
 ******************************************************************************
 */
template <typename T>
class ReduceSum<omp_reduce, T>
{
  using my_type = ReduceSum<omp_reduce, T>;

public:
  //
  // Constructor takes default value (default ctor is disabled).
  //
  explicit ReduceSum(T init_val, T initializer = 0)
    : parent(NULL), val(init_val), custom_init(initializer)
  {
  }

  //
  // Copy ctor.
  //
  ReduceSum(const ReduceSum<omp_reduce, T>& other) :
    parent(other.parent ? other.parent : &other),
    val(other.custom_init),
    custom_init(other.custom_init)
  {
  }

  //
  // Destruction releases the shared memory block chunk for reduction id
  // and id itself for others to use.
  //
  ~ReduceSum<omp_reduce, T>()
  {
    if (parent) {
#pragma omp critical
      {
        *parent += val;
      }
    }
  }

  //
  // Operator that returns reduced sum value.
  //
  operator T()
  {
    return val;
  }

  //
  // Method that returns sum value.
  //
  T get() { return operator T(); }

  //
  // += operator that adds value to sum for current thread.
  //
  const ReduceSum<omp_reduce, T>& operator+=(T rhs) const
  {
    this->val += rhs;
    return *this;
  }

  ReduceSum<omp_reduce, T>& operator+=(T rhs)
  {
    this->val += rhs;
    return *this;
  }

private:
  //
  // Default ctor is declared private and not implemented.
  //
  ReduceSum<omp_reduce, T>();

  const my_type * parent;

  mutable T val;
  T custom_init;

};

}  // closing brace for RAJA namespace

#endif  // closing endif for RAJA_ENABLE_CUDA guard

#endif  // closing endif for header file include guard
