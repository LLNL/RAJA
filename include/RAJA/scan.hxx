/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

#ifndef RAJA_scan_HXX
#define RAJA_scan_HXX

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
// For additional details, please also read raja/README-license.txt.
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

#include "RAJA/config.hxx"

#include <functional>
#include <type_traits>

namespace RAJA
{

/*!
******************************************************************************
*
* \brief  inclusive in-place scan execution policy
*
******************************************************************************
*/

template <typename ExecPolicy, typename T, typename BinaryFunction>
void inclusive_scan_inplace(T* begin, T* end, BinaryFunction f, T v)
{
  // TODO: add check for begin < end, Iter must be random-access
  const size_t size = end - begin;
  inclusive_scan_inplace(ExecPolicy{}, begin, end, size, f, v);
}

template <typename ExecPolicy, typename T>
void inclusive_scan_inplace(T* begin, T* end)
{
  inclusive_scan_inplace<ExecPolicy>(begin, end, std::plus<T>{}, T{0});
}

template <typename ExecPolicy,
          typename Iter,
          typename T = typename Iter::value_type,
          typename BinaryFunction>
void inclusive_scan_inplace(Iter begin, Iter end, BinaryFunction f, T v)
{
  // TODO: add check for begin < end, Iter must be random-access
  const size_t size = end - begin;
  inclusive_scan_inplace(ExecPolicy{}, begin, end, size, f, v);
}

template <typename ExecPolicy,
          typename Iter,
          typename T = typename Iter::value_type>
void inclusive_scan_inplace(Iter begin, Iter end)
{
  inclusive_scan_inplace<ExecPolicy>(begin, end, std::plus<T>{}, T{0});
}

/*!
******************************************************************************
*
* \brief  inclusive in-place scan execution policy
*
******************************************************************************
*/

template <typename ExecPolicy, typename T, typename BinaryFunction>
void exclusive_scan_inplace(T* begin, T* end, BinaryFunction f, T v)
{
  // TODO: add check for begin < end, Iter must be random-access
  const size_t size = end - begin;
  exclusive_scan_inplace(ExecPolicy{}, begin, end, size, f, v);
}

template <typename ExecPolicy, typename T>
void exclusive_scan_inplace(T* begin, T* end)
{
  exclusive_scan_inplace<ExecPolicy>(begin, end, std::plus<T>{}, T{0});
}

template <typename ExecPolicy,
          typename Iter,
          typename T = typename Iter::value_type,
          typename BinaryFunction>
void exclusive_scan_inplace(Iter begin, Iter end, BinaryFunction f, T v)
{
  // TODO: add check for begin < end, Iter must be random-access
  const size_t size = end - begin;
  exclusive_scan_inplace(ExecPolicy{}, begin, end, size, f, v);
}

template <typename ExecPolicy,
          typename Iter,
          typename T = typename Iter::value_type>
void exclusive_scan_inplace(Iter begin, Iter end)
{
  exclusive_scan_inplace(ExecPolicy{}, begin, end, std::plus<T>{}, T{0});
}

/*!
******************************************************************************
*
* \brief  inclusive scan execution policy
*
******************************************************************************
*/

template <typename ExecPolicy, typename T, typename BinaryFunction>
RAJA_INLINE void inclusive_scan(const T* begin,
                                const T* end,
                                T* out,
                                BinaryFunction f,
                                T v)
{
  inclusive_scan(ExecPolicy{}, begin, end, out, end - begin, f, v);
}

template <typename ExecPolicy, typename T>
RAJA_INLINE void inclusive_scan(const T* begin, const T* end, T* out)
{
  inclusive_scan(
      ExecPolicy{}, begin, end, out, end - begin, std::plus<T>{}, T{0});
}

template <typename ExecPolicy,
          typename Iter,
          typename T = typename Iter::value_type,
          typename IterOut,
          typename BinaryFunction>
RAJA_INLINE void inclusive_scan(const Iter begin,
                                const Iter end,
                                IterOut out,
                                BinaryFunction f,
                                T v)
{
  inclusive_scan(ExecPolicy{}, begin, end, out, end - begin, f, v);
}

template <typename ExecPolicy,
          typename Iter,
          typename T = typename Iter::value_type,
          typename IterOut>
RAJA_INLINE void inclusive_scan(const Iter begin, const Iter end, IterOut out)
{
  inclusive_scan(
      ExecPolicy{}, begin, end, out, end - begin, std::plus<T>{}, T{0});
}


/*!
******************************************************************************
*
* \brief  exclusive scan execution policy
*
******************************************************************************
*/

template <typename ExecPolicy, typename T>
RAJA_INLINE void exclusive_scan(const T* begin, const T* end, T* out)
{
  exclusive_scan(
      ExecPolicy{}, begin, end, out, end - begin, std::plus<T>{}, T{0});
}

template <typename ExecPolicy, typename T, typename BinaryFunction>
RAJA_INLINE void exclusive_scan(const T* begin,
                                const T* end,
                                T* out,
                                BinaryFunction f,
                                T v)
{
  exclusive_scan(ExecPolicy{}, begin, end, out, end - begin, f, v);
}

template <typename ExecPolicy,
          typename Iter,
          typename T = typename Iter::value_type,
          typename IterOut>
RAJA_INLINE void exclusive_scan(const Iter begin, const Iter end, IterOut out)
{
  exclusive_scan(
      ExecPolicy{}, begin, end, out, end - begin, std::plus<T>{}, T{0});
}

template <typename ExecPolicy,
          typename Iter,
          typename T = typename Iter::value_type,
          typename IterOut,
          typename BinaryFunction>
RAJA_INLINE void exclusive_scan(const Iter begin,
                                const Iter end,
                                IterOut out,
                                BinaryFunction f,
                                T v)
{
  exclusive_scan(ExecPolicy{}, begin, end, out, end - begin, f, v);
}

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
