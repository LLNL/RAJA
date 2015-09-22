/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file for simple comparison operations used in tests.
 *     
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_Compare_HXX
#define RAJA_Compare_HXX

namespace RAJA {

//
// Test approximate equality of floating point numbers. Borrowed from Knuth.
//
template <typename T>
bool equal(T a, T b)
{
   return (abs(a-b) <= ( ( abs(a) < abs(b) ? abs(a) : abs(b) ) * 10e-12 ) );
}

//
// Equality for integers.  Mainly here for consistent usage with above.
//
bool equal(int a, int b)
{
   return a == b ;
}

template <typename T>
bool array_equal(T ref_result,
                 T to_check,
                 Index_type alen)
{
   bool is_correct = true;
   for (Index_type i = 0 ; i < alen && is_correct; ++i) {
      is_correct &= equal(ref_result[i], to_check[i]);
   }

   return is_correct;
}


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
