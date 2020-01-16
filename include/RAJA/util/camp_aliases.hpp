/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file with aliases to camp types.
 *
 * The aliases included here are the ones that may be exposed through the
 * RAJA API based on our unit tests and examples. As you build new tests
 * and examples and you find that other camp types are exposed, please
 * add them to this file.
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_CAMP_ALIASES_HPP
#define RAJA_CAMP_ALIASES_HPP

#include "camp/defines.hpp"
#include "camp/list/list.hpp"
#include "camp/tuple.hpp"

namespace RAJA
{

using ::camp::at_v;

using ::camp::get;

using ::camp::list;

using ::camp::idx_t;

using ::camp::make_tuple;

using ::camp::tuple;

}  // end namespace RAJA

#endif /* RAJA_CAMP_ALIASES_HPP */
