/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for various index set builder methods.
 *
 * \author  Rich Hornung, Center for Applied Scientific Computing, LLNL
 * \author  Jeff Keasler, Applications, Simulations And Quality, LLNL
 *
 ******************************************************************************
 */

#ifndef RAJA_IndexSetBuilders_HXX
#define RAJA_IndexSetBuilders_HXX

#include "config.hxx"

#include "int_datatypes.hxx"

namespace RAJA {

class IndexSet;

/*!
 ******************************************************************************
 *
 * \brief Initialize index set with aligned Ranges and List segments from 
 *        array of indices with given length.
 *
 *        Specifically, Range segments will be greater than RANGE_MIN_LENGTH
 *        and starting index and length of each range segment will be
 *        multiples of RANGE_ALIGN. These constants are defined in the 
 *        RAJA config.hxx header file.
 *
 *        Note: given index set object is assumed to be empty.
 *
 *        Routine does no error-checking on argements and assumes Index_type
 *        array contains valid indices.
 *
 ******************************************************************************
 */
void buildIndexSetAligned(IndexSet& hiset,
                          const Index_type* const indices_in,
                          Index_type length);


}  // closing brace for RAJA namespace


#endif  // closing endif for header file include guard
