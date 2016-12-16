/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA floating point scalar and pointer types.
 *
 *          This file contains compiler-specific type definitions for scalars
 *          and data pointers used in RAJA code.
 *
 *          Note that some of these things depend on the contents of the
 *          RAJA_config.hxx header.  Others can be controlled by editing
 *          the RAJA_rules.mk file or specified using "-D" definitionsa
 *          directly on the compile-line.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

#ifndef RAJA_real_datatypes_HXX
#define RAJA_real_datatypes_HXX

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

#include "RAJA/config.hxx"

#if defined(RAJA_USE_COMPLEX)
#include <complex>
#endif

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief RAJA scalar type definitions.
 *
 ******************************************************************************
 */

#if defined(RAJA_USE_DOUBLE)
///
typedef double Real_type;

#elif defined(RAJA_USE_FLOAT)
///
typedef float Real_type;

#else
#error RAJA Real_type is undefined!

#endif

#if defined(RAJA_USE_COMPLEX)
///
typedef std::complex<Real_type> Complex_type;
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
typedef Real_type* RAJA_RESTRICT __attribute__((align_value(RAJA::DATA_ALIGN)))
TDRAReal_ptr;

typedef const Real_type* RAJA_RESTRICT
    __attribute__((align_value(RAJA::DATA_ALIGN))) const_TDRAReal_ptr;
#endif

#elif defined(RAJA_COMPILER_GNU)

#elif defined(RAJA_COMPILER_CLANG)
typedef Real_type aligned_real_type __attribute__((aligned(RAJA::DATA_ALIGN)));

typedef aligned_real_type* RAJA_RESTRICT TDRAReal_ptr;

typedef const aligned_real_type* RAJA_RESTRICT const_TDRAReal_ptr;

#else

typedef Real_type aligned_real_type;

typedef aligned_real_type* RAJA_RESTRICT TDRAReal_ptr;

typedef const aligned_real_type* RAJA_RESTRICT const_TDRAReal_ptr;

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
    return ((Real_type * RAJA_RESTRICT)dptr)[i];
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
    return ((Real_type * RAJA_RESTRICT)dptr)[i];
#else  // use alignment attribute
    return ((TDRAReal_ptr)dptr)[i];
#endif
  }

  ///
  const Real_type& operator[](Index_type i) const
  {
#if __ICC < 1300  // use alignment intrinsic
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT)dptr)[i];
#else  // use alignment attribute
    return ((TDRAReal_ptr)dptr)[i];
#endif
  }

#elif defined(RAJA_COMPILER_GNU)
  ///
  Real_type& operator[](Index_type i)
  {
#if 1  // NOTE: alignment instrinsic not available for older GNU compilers
    return ((Real_type * RAJA_RESTRICT)RAJA_ALIGN_DATA(dptr))[i];
#else
    return ((Real_type * RAJA_RESTRICT)dptr)[i];
#endif
  }

  ///
  const Real_type& operator[](Index_type i) const
  {
#if 1  // NOTE: alignment instrinsic not available for older GNU compilers
    return ((Real_type * RAJA_RESTRICT)RAJA_ALIGN_DATA(dptr))[i];
#else
    return ((Real_type * RAJA_RESTRICT)dptr)[i];
#endif
  }

#elif defined(RAJA_COMPILER_XLC)
  ///
  Real_type& operator[](Index_type i)
  {
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT)dptr)[i];
  }

  ///
  const Real_type& operator[](Index_type i) const
  {
    RAJA_ALIGN_DATA(dptr);
    return ((Real_type * RAJA_RESTRICT)dptr)[i];
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
    return ((Complex_type * RAJA_RESTRICT)dptr)[i];
  }

  ///
  ///  Const bracket operator.
  ///
  const Complex_type& operator[](Index_type i) const
  {
    return ((Complex_type * RAJA_RESTRICT)dptr)[i];
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
typedef Real_type* Real_ptr;
typedef const Real_type* const_Real_ptr;

#if defined(RAJA_USE_COMPLEX)
typedef Complex_type* Complex_ptr;
typedef const Complex_type* const_Complex_ptr;
#endif

typedef Real_type* UnalignedReal_ptr;
typedef const Real_type* const_UnalignedReal_ptr;

#elif defined(RAJA_USE_RESTRICT_PTR)
typedef Real_type* RAJA_RESTRICT Real_ptr;
typedef const Real_type* RAJA_RESTRICT const_Real_ptr;

#if defined(RAJA_USE_COMPLEX)
typedef Complex_type* RAJA_RESTRICT Complex_ptr;
typedef const Complex_type* RAJA_RESTRICT const_Complex_ptr;
#endif

typedef Real_type* RAJA_RESTRICT UnalignedReal_ptr;
typedef const Real_type* RAJA_RESTRICT const_UnalignedReal_ptr;

#elif defined(RAJA_USE_RESTRICT_ALIGNED_PTR)
typedef TDRAReal_ptr Real_ptr;
typedef const_TDRAReal_ptr const_Real_ptr;

#if defined(RAJA_USE_COMPLEX)
typedef Complex_type* RAJA_RESTRICT Complex_ptr;
typedef const Complex_type* RAJA_RESTRICT const_Complex_ptr;
#endif

typedef Real_type* RAJA_RESTRICT UnalignedReal_ptr;
typedef const Real_type* RAJA_RESTRICT const_UnalignedReal_ptr;

#elif defined(RAJA_USE_PTR_CLASS)
typedef RestrictAlignedRealPtr Real_ptr;
typedef ConstRestrictAlignedRealPtr const_Real_ptr;

#if defined(RAJA_USE_COMPLEX)
typedef RestrictComplexPtr Complex_ptr;
typedef ConstRestrictComplexPtr const_Complex_ptr;
#endif

typedef RestrictRealPtr UnalignedReal_ptr;
typedef ConstRestrictRealPtr const_UnalignedReal_ptr;

#else
#error RAJA pointer type is undefined!

#endif

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
