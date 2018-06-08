/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining layout operations for forallN templates.
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

#ifndef RAJA_ITERABLE_WRAPPER_HPP
#define RAJA_ITERABLE_WRAPPER_HPP

#include "RAJA/config.hpp"
#include <iterator>

namespace RAJA
{

namespace detail
{
/*!
 ******************************************************************************
 *
 * \brief  IterableWrapper struct holds iterators for any iterable type.
 *         Introduced to avoid deep copying data in iterables.
 *
 * \tparam T is the underlying data type for the iterable.
 *
 ******************************************************************************
 */
template<typename T>
struct IterableWrapper{
  
  //Declares iterator
  using iterator = typename T::iterator;

  //Constructor 
  IterableWrapper(T const &it)
    : m_begin{std::begin(it)}, m_end{std::end(it)}{}
 
  //Constructor which takes a begin and end iterator
  RAJA_HOST_DEVICE IterableWrapper(iterator begin, iterator end)
  	: m_begin{begin}, m_end{end}{}

  //Method to return begin iterator
  RAJA_HOST_DEVICE RAJA_INLINE iterator begin() const {return m_begin;};

  //Method to return end iterator
  RAJA_HOST_DEVICE RAJA_INLINE iterator end() const {return m_end;};

 //Returns IterableWrapper starting at begin with length ``length"
  RAJA_HOST_DEVICE RAJA_INLINE IterableWrapper slice(Index_type begin,
                                                      Index_type length) const
  {     
    auto start = m_begin + begin;
    auto end = start + length > m_end ? m_end : start + length;
    return IterableWrapper(start, end);
  }

private:
  iterator m_begin; 
  iterator m_end;
};

}

}  // namespace RAJA

#endif
