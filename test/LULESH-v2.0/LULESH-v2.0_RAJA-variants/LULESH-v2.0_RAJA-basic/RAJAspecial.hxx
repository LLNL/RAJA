#ifndef RAJAspecial_HXX
#define RAJAspecial_HXX

#include <omp.h>

namespace RAJA {

/*!
 ***************************************************************************** 
 *
 * \brief  Traverse contiguous range of indices using OpenMP for with
 *         nowait clause (assumes loop appears in a parallel region).
 *
 *****************************************************************************
 */

struct omp_for_nowait_exec {};

template <typename LOOP_BODY>
inline  __attribute__((always_inline))
void forall(omp_for_nowait_exec,
            const int begin, const int end,
            LOOP_BODY loop_body)
{
//#pragma omp for nowait schedule(static)
#pragma omp for nowait
   for ( int ii = begin ; ii < end ; ++ii ) {
      loop_body( ii );
   }
}


}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard

