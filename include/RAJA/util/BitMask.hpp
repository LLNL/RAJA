/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining a bit masking operator
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_util_BitMask_HPP
#define RAJA_util_BitMask_HPP

#include "RAJA/config.hpp"


namespace RAJA
{


  /*!
   * A bit-masking operator
   *
   * Provides an operator that shifts and masks in input value to extract
   * a value contiguous set of bits.
   *
   * result = (input >> Shift) & (Mask)
   *
   * Where mask is (1<<Width)-1, or the number of bits defined by Width.
   *
   *
   */
  template<size_t Width, size_t Shift>
  struct BitMask {
    static constexpr size_t shift = Shift;
    static constexpr size_t width = Width;
    static constexpr size_t max_input_size = 1<<(Shift+Width);
    static constexpr size_t max_masked_size = 1<<Width;
    static constexpr size_t max_shifted_size = 1<<Shift;

    template<typename T>
    RAJA_HOST_DEVICE
    static constexpr T maskValue(T input) {
      return( (input>>((T) Shift)) & (T) ((1<<(Width))-1));
    }
  };

}  // namespace RAJA

#endif //RAJA_util_BitMask_HPP
