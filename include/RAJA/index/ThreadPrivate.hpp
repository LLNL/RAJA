/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining thread private variables for RAJA::nested
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_ThreadPrivate_HPP
#define RAJA_ThreadPrivate_HPP

#include "RAJA/config.hpp"

#include "RAJA/internal/Iterators.hpp"

#include "RAJA/util/concepts.hpp"

#include <iostream>

namespace RAJA
{

/*!
 ******************************************************************************
 *
 * \brief  Pseudo-Segment class that provides a thread-private value
 *
 * \tparam StorageT the underlying data type of thread-private value
 *
 * A ThreadPrivate variable masquerades as a segment for RAJA::nested::forall,
 * allows for a thread-private value to be passed in to each lambda, and
 * provides no iteration space.
 *
 * Attempting to iterate over a ThreadPrivate is invalid.
 *
 *
 *
 ******************************************************************************
 */


template <typename StorageT>
struct ThreadPrivate {

  //! the underlying iterator type (this is bogus, to make it look like a segment)
  using iterator = Iterators::numeric_iterator<StorageT, StorageT>;
  //! the underlying value_type type
  /*!
   * this corresponds to the template parameter
   */
  using value_type = StorageT;

  //! obtain an iterator to the beginning of this ThreadPrivate
  /*!
   * \return an iterator corresponding to the beginning of the Segment
   */
  RAJA_HOST_DEVICE RAJA_INLINE iterator begin() const { return iterator{0}; }

  //! obtain an iterator to the end of this ThreadPrivate
  /*!
   * \return an iterator corresponding to the end of the Segment
   */
  RAJA_HOST_DEVICE RAJA_INLINE iterator end() const { return iterator{0}; }

  //! obtain the size of this ThreadPrivate
  /*!
   * \return Always 1
   */
  RAJA_HOST_DEVICE RAJA_INLINE StorageT size() const { return 1; }

  //! Create a slice of this instance as a new instance
  /*!
   * \return Returns a copy of *this, since ThreadPrivate has no iteration
   *         space.
   */
  RAJA_HOST_DEVICE RAJA_INLINE ThreadPrivate slice(Index_type ,
                                           Index_type ) const
  {
    return ThreadPrivate{};
  }

  //! equality comparison
  RAJA_HOST_DEVICE RAJA_INLINE bool operator==(ThreadPrivate const& o) const
  {
    return false;
  }

};



}  // namespace RAJA


#endif  // closing endif for header file include guard
