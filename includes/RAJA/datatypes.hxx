/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA scalar and pointer type definitions.
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
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL 
 *
 ******************************************************************************
 */

#ifndef RAJA_datatypes_HXX
#define RAJA_datatypes_HXX

#include "config.hxx"

#include<complex>

namespace RAJA {


/*!
 ******************************************************************************
 *
 * \brief RAJA scalar type definitions.
 *
 ******************************************************************************
 */
///
typedef int     Index_type;

#if defined(RAJA_USE_DOUBLE)
///
typedef double  Real_type;


#elif defined(RAJA_USE_FLOAT)
///
typedef float  Real_type;


#else
#error RAJA Real_type is undefined!

#endif

///
typedef std::complex<Real_type> Complex_type;


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
typedef Real_type* __restrict__ __attribute__((align_value(RAJA::DATA_ALIGN))) TDRAReal_ptr;
#endif


#elif defined(RAJA_COMPILER_GNU) 
//
// Nothing here for now because alignment attribute is not working...
//


#elif defined(RAJA_COMPILER_XLC12)
extern
#ifdef __cplusplus
"builtin"
#endif
void __alignx(int n, const void* addr);


#elif defined(RAJA_COMPILER_CLANG)
typedef Real_type aligned_real_type __attribute__((aligned (RAJA::DATA_ALIGN)));
typedef aligned_real_type* __restrict__ TDRAReal_ptr;


#else
#error RAJA compiler is undefined!

#endif


#if defined(RAJA_USE_PTR_CLASS)
/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted aligned Real_type pointer.
 *
 ******************************************************************************
 */
class RestrictAlignRealPtr
{
public:

   ///
   /// Default ctor.
   ///
   RestrictAlignRealPtr() : dptr(0) { ; }

   ///
   /// Copy ctor.
   ///
   RestrictAlignRealPtr(Real_type* d) : dptr(d) { ; }

   ///
   /// Compiler-specific bracket operator.
   ///

#if defined(RAJA_COMPILER_ICC)
   ///
   Real_type& operator [] (Index_type i)
   {
#if __ICC < 1300 // use alignment intrinsic
      RAJA_ALIGN_DATA(dptr);
      return( (Real_type* __restrict__) dptr)[i];
#else // use alignment attribute
      return( (TDRAReal_ptr) dptr)[i];
#endif
   }


#elif defined(RAJA_COMPILER_GNU)
   ///
   Real_type& operator [] (Index_type i)
   {
#if 1 // NOTE: alignment instrinsic not available for older GNU compilers
      return( (Real_type* __restrict__) RAJA_ALIGN_DATA(dptr) )[i];
#else
      return( (Real_type* __restrict__) dptr)[i];
#endif
   }


#elif defined(RAJA_COMPILER_XLC12)
   Real_type& operator [] (Index_type i)
   {
      RAJA_ALIGN_DATA(dptr);
      return( (Real_type* __restrict__) dptr)[i];
   }


#elif defined(RAJA_COMPILER_CLANG)
   Real_type& operator [] (Index_type i)
   {
      return( (TDRAReal_ptr) dptr)[i];
   }


#else
#error RAJA compiler macro is undefined!

#endif

   ///
   ///  Pointer access operator.
   ///
   operator Real_type*() { return dptr; }

   ///
   ///  Pointer access operator, consistent with boost shared ptr.
   ///
   Real_type* get() { return dptr; }

   ///
   /// + operator for pointer arithmetic.
   ///
   Real_type* operator+ (Index_type i) { return dptr+i; }

private:
   Real_type* dptr;
};


/*!
 ******************************************************************************
 *
 * \brief Class representing a restricted Complex_type pointer.
 *
 ******************************************************************************
 */
class RestrictComplexPtr
{
public:

   ///
   /// Default ctor.
   ///
   RestrictComplexPtr() : dptr(0) { ; }

   ///
   /// Copy ctor.
   ///
   RestrictComplexPtr(Complex_type* d) : dptr(d) { ; }

   ///
   /// Bracket operator.
   ///
   Complex_type& operator [] (Index_type i)
   {
      return( (Complex_type* __restrict__) dptr)[i];
   }

   ///
   ///  Pointer access operator.
   ///
   operator Complex_type*() { return dptr; }

   ///
   ///  Pointer access operator, consistent with boost shared ptr.
   ///
   Complex_type* get() { return dptr; }

   ///
   /// + operator for pointer arithmetic.
   ///
   Complex_type* operator+ (Index_type i) { return dptr+i; }

private:
   Complex_type* dptr;
};
#endif  // defined(RAJA_USE_PTR_CLASS)


/*
 ******************************************************************************
 *
 * Finally, we define data pointer types based on definitions above.
 *
 ******************************************************************************
 */
#if defined(RAJA_USE_BARE_PTR)
typedef Real_type* Real_ptr;
typedef Complex_type* Complex_ptr;


#elif defined(RAJA_USE_RESTRICT_PTR)
typedef Real_type* __restrict__ Real_ptr;
typedef Complex_type* __restrict__ Complex_ptr;


#elif defined(RAJA_USE_RESTRICT_ALIGNED_PTR)
typedef TDRAReal_ptr Real_ptr;
typedef Complex_type* __restrict__ Complex_ptr;


#elif defined(RAJA_USE_PTR_CLASS)
typedef RestrictAlignRealPtr Real_ptr;
typedef RestrictComplexPtr Complex_ptr;


#else
#error RAJA pointer type is undefined!

#endif


}  // closing brace for namespace statement


#endif  // closing endif for header file include guard
