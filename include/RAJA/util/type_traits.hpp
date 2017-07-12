/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA type trait definitions.
 *
 *          Definitions in this file will propagate to all RAJA header files.
 *
 ******************************************************************************
 */

#ifndef RAJA_type_traits_HPP
#define RAJA_type_traits_HPP

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

#include "RAJA/util/concepts.hpp"

namespace RAJA
{

namespace type_traits
{

  template <typename T>
  using IterableValue = decltype(*std::begin(RAJA::concepts::val<T>()));

  template <typename T>
  using IteratorValue = decltype(*RAJA::concepts::val<T>());

  template <typename T>
  using is_bidirectional_iterator = concepts::requires_<concepts::BidirectionalIterator, T>;

  template <typename T>
  using is_bidirectional_range = concepts::requires_<concepts::BidirectionalRange, T>;

  template <typename T>
  using is_forward_iterator = concepts::requires_<concepts::ForwardIterator, T>;

  template <typename T>
  using is_forward_range = concepts::requires_<concepts::ForwardRange, T>;

  template <typename T>
  using is_random_access_iterator = concepts::requires_<concepts::RandomAccessIterator, T>;

  template <typename T>
  using is_random_access_range = concepts::requires_<concepts::RandomAccessRange, T>;

  template <typename T>
  using is_iterator = concepts::requires_<concepts::Iterator, T>;

  template <typename T>
  using is_range = concepts::requires_<concepts::Range, T>;

  template <typename T>
  using is_comparable = concepts::requires_<concepts::Comparable, T>;

  template <typename T, typename U>
  using is_comparable_to = concepts::requires_<concepts::ComparableTo, T, U>;

  template <typename T>
  using is_integral = concepts::requires_<concepts::Integral, T>;

  template <typename T>
  using is_signed = concepts::requires_<concepts::Signed, T>;

  template <typename T>
  using is_unsigned = concepts::requires_<concepts::Unsigned, T>;

}  // end namespace type_traits

}  // end namespace RAJA

#endif
