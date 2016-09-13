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
* Usage example:
*
* \verbatim

inclusive_scan_inplace <upsweep_policy, downsweep_policy> (a, a + n, f, 0);

// or

inclusive_scan_inplace <exec_pol> (a, a + n, f, 0);

* \endverbatim
*
******************************************************************************
*/

template <typename ExecPolicy,
          typename Iter,
          typename Value = typename std::decay<
              typename std::remove_pointer<Iter>::type>::type,
          typename BinaryFn>
void inclusive_scan_inplace(Iter begin,
                            Iter end,
                            BinaryFn f = std::plus<Value, Value>{},
                            Value v = Value{0})
{
  // TODO: add check for begin < end, Iter must be random-access
  const size_t size{end - begin};
  RAJA::internal::inclusive_scan_inplace(ExecPolicy{}, begin, end, size, f, v);
}

/*!
******************************************************************************
*
* \brief  exclusive in-place scan execution policy
*
* Usage example:
*
* \verbatim

exclusive_scan_inplace <upsweep_policy, downsweep_policy> (a, a + n, f, 0);

// or

exclusive_scan_inplace <exec_pol> (a, a + n, f, 0);

* \endverbatim
*
******************************************************************************
*/

template <typename ExecPolicy,
          typename Iter,
          typename Value = typename std::decay<
              typename std::remove_pointer<Iter>::type>::type,
          typename BinaryFn>
void exclusive_scan_inplace(Iter begin,
                            Iter end,
                            BinaryFn f = std::plus<Value, Value>{},
                            Value v = Value{0})
{
  // TODO: add check for begin < end, Iter must be random-access
  const size_t size{end - begin};
  RAJA::internal::exclusive_scan_inplace(ExecPolicy{}, begin, end, size, f, v);
}

/*!
******************************************************************************
*
* \brief  inclusive scan execution policy
*
* Usage example:
*
* \verbatim

inclusive_scan <upsweep_policy, downsweep_policy> (a, a + n, b, f, 0);

// or

inclusive_scan <exec_pol> (a, a + n, b, f, 0);

* \endverbatim
*
******************************************************************************
*/

template <typename ExecPolicy typename Iter,
          typename IterOut,
          typename Value = typename std::decay<
              typename std::remove_pointer<Iter>::type>::type,
          typename BinaryFn>
void inclusive_scan(const Iter begin,
                    const Iter end,
                    IterOut out,
                    BinaryFn f = std::plus<Value, Value>{},
                    Value v = Value{0})
{

  // TODO: add check for begin < end, Iter must be random-access
  const size_t size{end - begin};
  RAJA::internal::inclusive_scan(ExecPolicy{}, begin, end, out, size, f, v);
}

/*!
******************************************************************************
*
* \brief  exclusive scan execution policy
*
* Usage example:
*
* \verbatim

exclusive_scan <upsweep_policy, downsweep_policy> (a, a + n, b, f, 0);

// or

exclusive_scan <exec_pol> (a, a + n, b, f, 0);

* \endverbatim
*
******************************************************************************
*/


template <typename ExecPolicy typename Iter,
          typename IterOut,
          typename Value = typename std::decay<
              typename std::remove_pointer<Iter>::type>::type,
          typename BinaryFn>
void exclusive_scan(const Iter begin,
                    const Iter end,
                    IterOut out,
                    BinaryFn f = std::plus<Value, Value>{},
                    Value v = Value{0})
{

  // TODO: add check for begin < end, Iter must be random-access
  const size_t size{end - begin};
  RAJA::internal::exclusive_scan(ExecPolicy{}, begin, end, out, size, f, v);
}

}  // closing brace for RAJA namespace

#endif  // closing endif for header file include guard
