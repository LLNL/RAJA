/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining typed rangesegment classes.
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

#ifndef RAJA_RangeTuple_HPP
#define RAJA_RangeTuple_HPP

#include "RAJA/config.hpp"

#include "camp/camp.hpp"

namespace RAJA
{

template <typename... Ranges>
auto make_range_tuple(Ranges... ranges)
    -> camp::tagged_tuple<camp::list<typename Ranges::value_type...>, Ranges...>
{
  return camp::make_tagged_tuple<camp::list<typename Ranges::value_type...>>(ranges...);
}

}  // namespace RAJA

#endif  // closing endif for header file include guard
