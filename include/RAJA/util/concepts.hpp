/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for RAJA concept definitions.
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

#ifndef RAJA_concepts_HPP
#define RAJA_concepts_HPP

#include <iterator>
#include <type_traits>

#include "camp/concepts.hpp"

namespace RAJA
{

namespace concepts
{
using namespace camp::concepts;

template <typename From, typename To>
struct ConvertibleTo
  : DefineConcept(::RAJA::concepts::convertible_to<To>(camp::val<From>())) {
};

}

namespace type_traits
{
using namespace camp::type_traits;

DefineTypeTraitFromConcept(convertible_to, concepts::ConvertibleTo);
}

}  // end namespace RAJA

#endif
