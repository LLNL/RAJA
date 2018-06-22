/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA index used to index SIMD vectors in
 *          RAJA::View objects.
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

#ifndef RAJA_policy_vec_vectorindex_HPP
#define RAJA_policy_vec_vectorindex_HPP

#include <iterator>
#include <type_traits>

#include "RAJA/config.hpp"

#include "RAJA/util/types.hpp"

namespace RAJA
{

namespace vec
{


template<typename IndexType, typename VecType>
struct VectorIndex
{
    using vector_type = VecType;
    using index_type = IndexType;
    IndexType value;
};


}  // closing brace for vec namespace

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
