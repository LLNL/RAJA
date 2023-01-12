/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA type definitions.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_Types_HPP
#define RAJA_Types_HPP

#include "RAJA/config.hpp"

#include <cstddef>

#if defined(RAJA_USE_COMPLEX)
#include <complex>
#endif

#include "camp/helpers.hpp"

namespace RAJA
{

///
/// Enumeration used to indicate whether ListSegment object owns data
/// representing its indices.
///
enum IndexOwnership { Unowned, Owned };

///
/// Type use for all loop indexing in RAJA constructs.
///
using Index_type = std::ptrdiff_t;

///
/// Integer value for undefined indices and other integer values.
/// Although this is a magic value, it avoids sprinkling them throughout code.
///
const int UndefinedValue = -9999999;


///
/// Template list of sizes
///
template <Index_type... Sizes>
struct SizeList {
};


/*!
 ******************************************************************************
 *
 * \brief RAJA scalar type definitions.
 *
 ******************************************************************************
 */

#if defined(RAJA_USE_DOUBLE)
///
using Real_type = double;

#elif defined(RAJA_USE_FLOAT)
///
using Real_type = float;

#else
#error RAJA Real_type is undefined!

#endif

#if defined(RAJA_USE_COMPLEX)
///
using Complex_type = std::complex<Real_type>;
#endif

/*
 ******************************************************************************
 *
 * The following items include some setup items for definitions that follow.
 *
 ******************************************************************************
 */

#if defined(RAJA_COMPILER_ICC)
//
// alignment attribute supported for versions > 12
//
#if __ICC >= 1300
using TDRAReal_ptr = Real_type* RAJA_RESTRICT __attribute__((align_value(RAJA::DATA_ALIGN)));

using const_TDRAReal_ptr = const TDRAReal_ptr;
#endif

#elif defined(RAJA_COMPILER_GNU)

#elif defined(RAJA_COMPILER_CLANG)
using TDRAReal_ptr = Real_type* RAJA_RESTRICT __attribute__((aligned(RAJA::DATA_ALIGN)));

using const_TDRAReal_ptr = const TDRAReal_ptr;

#else

using TDRAReal_ptr = Real_type* RAJA_RESTRICT;

using const_TDRAReal_ptr = const TDRAReal_ptr;

#endif

#if defined(RAJA_USE_PTR_CLASS)
/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted Real_type const pointer.
 *
 ******************************************************************************
 */
class ConstRestrictRealPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  ConstRestrictRealPtr() : dptr(0) { ; }

  ConstRestrictRealPtr(const Real_type* d) : dptr(d) { ; }

  ConstRestrictRealPtr& operator=(const Real_type* d)
  {
    ConstRestrictRealPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to bare const pointer.
  ///
  operator const Real_type*() { return dptr; }

  ///
  ///  "Explicit conversion operator" to bare const pointer,
  ///  consistent with boost shared ptr.
  ///
  const Real_type* get() const { return dptr; }

  ///
  /// Bracket operator.
  ///
  const Real_type& operator[](Index_type i) const
  {
    return ((const Real_type* RAJA_RESTRICT)dptr)[i];
  }

  ///
  /// + operator for pointer arithmetic.
  ///
  const Real_type* operator+(Index_type i) const { return dptr + i; }

private:
  const Real_type* dptr;
};

/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted Real_type (non-const) pointer.
 *
 ******************************************************************************
 */
class RestrictRealPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  RestrictRealPtr() : dptr(0) { ; }

  RestrictRealPtr(Real_type* d) : dptr(d) { ; }

  RestrictRealPtr& operator=(Real_type* d)
  {
    RestrictRealPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to (non-const) bare pointer.
  ///
  operator Real_type*() { return dptr; }

  ///
  ///  Implicit conversion operator to const bare pointer.
  ///
  operator const Real_type*() const { return dptr; }

  ///
  ///  "Explicit conversion operator" to (non-const) bare pointer,
  ///  consistent with boost shared ptr.
  ///
  Real_type* get() { return dptr; }

  ///
  ///  "Explicit conversion operator" to const bare pointer,
  ///  consistent with boost shared ptr.
  ///
  const Real_type* get() const { return dptr; }

  ///
  ///  Operator that enables implicit conversion from RestrictRealPtr to
  ///  RestrictRealConstPtr.
  ///
  operator ConstRestrictRealPtr() { return ConstRestrictRealPtr(dptr); }

  ///
  /// Bracket operator.
  ///
  Real_type& operator[](Index_type i)
  {
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
  }

  ///
  /// + operator for (non-const) pointer arithmetic.
  ///
  Real_type* operator+(Index_type i) { return dptr + i; }

  ///
  /// + operator for const pointer arithmetic.
  ///
  const Real_type* operator+(Index_type i) const { return dptr + i; }

private:
  Real_type* dptr;
};

/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted aligned Real_type const pointer.
 *
 ******************************************************************************
 */
class ConstRestrictAlignedRealPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  ConstRestrictAlignedRealPtr() : dptr(0) { ; }

  ConstRestrictAlignedRealPtr(const Real_type* d) : dptr(d) { ; }

  ConstRestrictAlignedRealPtr& operator=(const Real_type* d)
  {
    ConstRestrictAlignedRealPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to bare const pointer.
  ///
  operator const Real_type*() { return dptr; }

  ///
  ///  "Explicit conversion operator" to bare const pointer,
  ///  consistent with boost shared ptr.
  ///
  const Real_type* get() const { return dptr; }

  ///
  /// Compiler-specific bracket operators.
  ///

#if defined(RAJA_COMPILER_ICC)
  ///
  const Real_type& operator[](Index_type i) const
  {
#if __ICC < 1300  // use alignment intrinsic
    RAJA_ALIGN_DATA(dptr);
    return ((const Real_type* RAJA_RESTRICT)dptr)[i];
#else  // use alignment attribute
    return ((const_TDRAReal_ptr)dptr)[i];
#endif
  }

#elif defined(RAJA_COMPILER_GNU)
  ///
  const Real_type& operator[](Index_type i) const
  {
#if 1  // NOTE: alignment instrinsic not available for older GNU compilers
    return ((const Real_type* RAJA_RESTRICT)RAJA_ALIGN_DATA(dptr))[i];
#else
    return ((const Real_type* RAJA_RESTRICT)dptr)[i];
#endif
  }

#elif defined(RAJA_COMPILER_XLC)
  const Real_type& operator[](Index_type i) const
  {
    RAJA_ALIGN_DATA(dptr);
    return ((const Real_type* RAJA_RESTRICT)dptr)[i];
  }

#elif defined(RAJA_COMPILER_CLANG)
  const Real_type& operator[](Index_type i) const
  {
    return ((const_TDRAReal_ptr)dptr)[i];
  }

#else
#error RAJA compiler macro is undefined!

#endif

  ///
  /// + operator for pointer arithmetic.
  ///
  const Real_type* operator+(Index_type i) const { return dptr + i; }

private:
  const Real_type* dptr;
};

/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted aligned Real_type (non-const) pointer.
 *
 ******************************************************************************
 */
class RestrictAlignedRealPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  RestrictAlignedRealPtr() : dptr(0) { ; }

  RestrictAlignedRealPtr(Real_type* d) : dptr(d) { ; }

  RestrictAlignedRealPtr& operator=(Real_type* d)
  {
    RestrictAlignedRealPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to (non-const) bare pointer.
  ///
  operator Real_type*() { return dptr; }

  ///
  ///  Implicit conversion operator to const bare pointer.
  ///
  operator const Real_type*() const { return dptr; }

  ///
  ///  "Explicit conversion operator" to (non-const) bare pointer,
  ///  consistent with boost shared ptr.
  ///
  Real_type* get() { return dptr; }

  ///
  ///  "Explicit conversion operator" to const bare pointer,
  ///  consistent with boost shared ptr.
  ///
  const Real_type* get() const { return dptr; }

  ///
  ///  Operator that enables implicit conversion from
  ///  RestrictAlignedRealPtr to RestrictAlignedRealConstPtr.
  ///
  operator ConstRestrictAlignedRealPtr()
  {
    return ConstRestrictAlignedRealPtr(dptr);
  }

  ///
  /// Compiler-specific bracket operators.
  ///

#if defined(RAJA_COMPILER_ICC)
  ///
  Real_type& operator[](Index_type i)
  {
#if __ICC < 1300  // use alignment intrinsic
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
#else  // use alignment attribute
    return ((TDRAReal_ptr)dptr)[i];
#endif
  }

  ///
  const Real_type& operator[](Index_type i) const
  {
#if __ICC < 1300  // use alignment intrinsic
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
#else  // use alignment attribute
    return ((TDRAReal_ptr)dptr)[i];
#endif
  }

#elif defined(RAJA_COMPILER_GNU)
  ///
  Real_type& operator[](Index_type i)
  {
#if 1  // NOTE: alignment instrinsic not available for older GNU compilers
    return ((Real_type * RAJA_RESTRICT) RAJA_ALIGN_DATA(dptr))[i];
#else
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
#endif
  }

  ///
  const Real_type& operator[](Index_type i) const
  {
#if 1  // NOTE: alignment instrinsic not available for older GNU compilers
    return ((Real_type * RAJA_RESTRICT) RAJA_ALIGN_DATA(dptr))[i];
#else
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
#endif
  }

#elif defined(RAJA_COMPILER_XLC)
  ///
  Real_type& operator[](Index_type i)
  {
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
  }

  ///
  const Real_type& operator[](Index_type i) const
  {
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT) dptr)[i];
  }

#elif defined(RAJA_COMPILER_CLANG)
  ///
  Real_type& operator[](Index_type i) { return ((TDRAReal_ptr)dptr)[i]; }

  ///
  const Real_type& operator[](Index_type i) const
  {
    return ((TDRAReal_ptr)dptr)[i];
  }

#else
#error RAJA compiler macro is undefined!

#endif

  ///
  /// + operator for (non-const) pointer arithmetic.
  ///
  Real_type* operator+(Index_type i) { return dptr + i; }

  ///
  /// + operator for const pointer arithmetic.
  ///
  const Real_type* operator+(Index_type i) const { return dptr + i; }

private:
  Real_type* dptr;
};

#if defined(RAJA_USE_COMPLEX)
/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted Complex_type const pointer.
 *
 ******************************************************************************
 */
class ConstRestrictComplexPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  ConstRestrictComplexPtr() : dptr(0) { ; }

  ConstRestrictComplexPtr(const Complex_type* d) : dptr(d) { ; }

  ConstRestrictComplexPtr& operator=(const Complex_type* d)
  {
    ConstRestrictComplexPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to bare const pointer.
  ///
  operator const Complex_type*() const { return dptr; }

  ///
  ///  "Explicit conversion operator" to bare const pointer,
  ///  consistent with boost shared ptr.
  ///
  const Complex_type* get() const { return dptr; }

  ///
  ///  Bracket operator.
  ///
  const Complex_type& operator[](Index_type i) const
  {
    return ((const Complex_type* RAJA_RESTRICT)dptr)[i];
  }

  ///
  /// + operator for pointer arithmetic.
  ///
  const Complex_type* operator+(Index_type i) const { return dptr + i; }

private:
  const Complex_type* dptr;
};

/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted Complex_type (non-const) pointer.
 *
 ******************************************************************************
 */
class RestrictComplexPtr
{
public:
  ///
  /// Ctors and assignment op.
  ///

  RestrictComplexPtr() : dptr(0) { ; }

  RestrictComplexPtr(Complex_type* d) : dptr(d) { ; }

  RestrictComplexPtr& operator=(Complex_type* d)
  {
    RestrictComplexPtr copy(d);
    std::swap(dptr, copy.dptr);
    return *this;
  }

  ///
  /// NOTE: Using compiler-generated copy ctor, dtor, and copy assignment op.
  ///

  ///
  ///  Implicit conversion operator to (non-const) bare pointer.
  ///
  operator Complex_type*() { return dptr; }

  ///
  ///  Implicit conversion operator to const bare pointer.
  ///
  operator const Complex_type*() const { return dptr; }

  ///
  ///  "Explicit conversion operator" to (non-const) bare pointer,
  ///  consistent with boost shared ptr.
  ///
  Complex_type* get() { return dptr; }

  ///
  ///  "Explicit conversion operator" to const bare pointer,
  ///  consistent with boost shared ptr.
  ///
  const Complex_type* get() const { return dptr; }

  ///
  ///  Operator that enables implicit conversion from RestrictComplexPtr to
  ///  RestrictComplexConstPtr.
  ///
  operator ConstRestrictComplexPtr() { return ConstRestrictComplexPtr(dptr); }

  ///
  ///  (Non-const) bracket operator.
  ///
  Complex_type& operator[](Index_type i)
  {
    return ((Complex_type * RAJA_RESTRICT) dptr)[i];
  }

  ///
  ///  Const bracket operator.
  ///
  const Complex_type& operator[](Index_type i) const
  {
    return ((Complex_type * RAJA_RESTRICT) dptr)[i];
  }

  ///
  /// + operator for (non-const) pointer arithmetic.
  ///
  Complex_type* operator+(Index_type i) { return dptr + i; }

  ///
  /// + operator for const pointer arithmetic.
  ///
  const Complex_type* operator+(Index_type i) const { return dptr + i; }

private:
  Complex_type* dptr;
};
#endif  // defined(RAJA_USE_COMPLEX)

#endif  // defined(RAJA_USE_PTR_CLASS)

/*
 ******************************************************************************
 *
 * Finally, we define data pointer types based on definitions above and
 * -D value given at compile time.
 *
 ******************************************************************************
 */
#if defined(RAJA_USE_BARE_PTR)
using Real_ptr = Real_type*;
using const_Real_ptr = const Real_type*;

#if defined(RAJA_USE_COMPLEX)
using Complex_ptr = Complex_type*;
using const_Complex_ptr = const Complex_type*;
#endif

using UnalignedReal_ptr = Real_type*;
using const_UnalignedReal_ptr = const Real_type*;

#elif defined(RAJA_USE_RESTRICT_PTR)
using Real_ptr = Real_type* RAJA_RESTRICT;
using const_Real_ptr = const Real_type* RAJA_RESTRICT;

#if defined(RAJA_USE_COMPLEX)
using Complex_ptr = Complex_type* RAJA_RESTRICT;
using const_Complex_ptr = const Complex_type* RAJA_RESTRICT;
#endif

using UnalignedReal_ptr = Real_type* RAJA_RESTRICT;
using const_UnalignedReal_ptr = const Real_type* RAJA_RESTRICT;

#elif defined(RAJA_USE_RESTRICT_ALIGNED_PTR)
using Real_ptr = TDRAReal_ptr;
using const_Real_ptr = const_TDRAReal_ptr;

#if defined(RAJA_USE_COMPLEX)
using Complex_ptr = Complex_type* RAJA_RESTRICT;
using const_Complex_ptr = const Complex_type* RAJA_RESTRICT;
#endif

using UnalignedReal_ptr = Real_type* RAJA_RESTRICT;
using const_UnalignedReal_ptr = const Real_type* RAJA_RESTRICT;

#elif defined(RAJA_USE_PTR_CLASS)
using Real_ptr = RestrictAlignedRealPtr;
using const_Real_ptr = ConstRestrictAlignedRealPtr;

#if defined(RAJA_USE_COMPLEX)
using Complex_ptr = RestrictComplexPtr;
using const_Complex_ptr = ConstRestrictComplexPtr;
#endif

using UnalignedReal_ptr = RestrictRealPtr;
using const_UnalignedReal_ptr = ConstRestrictRealPtr;

#else
#error RAJA pointer type is undefined!

#endif

}  // namespace RAJA

#endif  // closing endif for header file include guard
