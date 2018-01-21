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
// Copyright (c) 2016-17, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_CAMP_ALIASES_HPP
#define RAJA_CAMP_ALIASES_HPP

#include "camp/defines.hpp"
#include "camp/tuple.hpp"
#include "camp/list/list.hpp"

namespace RAJA
{

  using ::camp::make_tuple;

  using ::camp::tuple;

  using ::camp::list;

  using ::camp::at_v;

  using ::camp::idx_t;

}  // end namespace RAJA

#endif /* RAJA_CAMP_ALIASES_HPP */
