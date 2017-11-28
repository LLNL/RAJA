/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for common RAJA internal definitions.
 *
 ******************************************************************************
 */

#ifndef RAJA_SOA_PTR_HPP
#define RAJA_SOA_PTR_HPP

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

#include "RAJA/config.hpp"

// for RAJA::reduce::detail::ValueLoc
#include "RAJA/pattern/detail/reduce.hpp"

namespace RAJA
{

namespace detail
{

/*!
 * @brief Pointer class specialized for Struct of Array data layout allocated
 *        via RAJA basic_mempools.
 *
 * This is useful for creating a vectorizable data layout and getting
 * coalesced memory accesses or avoiding shared memory bank conflicts in cuda.
 */
template <
    typename T,
    typename mempool =
        RAJA::basic_mempool::MemPool<RAJA::basic_mempool::generic_allocator> >
class SoAPtr
{
  using value_type = T;

public:
  SoAPtr() = default;
  explicit SoAPtr(size_t size)
      : mem(mempool::getInstance().template malloc<value_type>(size))
  {
  }

  SoAPtr& allocate(size_t size)
  {
    mem = mempool::getInstance().template malloc<value_type>(size);
    return *this;
  }

  SoAPtr& deallocate()
  {
    mempool::getInstance().free(mem);
    mem = nullptr;
    return *this;
  }

  RAJA_HOST_DEVICE bool allocated() const { return mem != nullptr; }

  RAJA_HOST_DEVICE value_type get(size_t i) const { return mem[i]; }
  RAJA_HOST_DEVICE void set(size_t i, value_type val) { mem[i] = val; }

private:
  value_type* mem = nullptr;
};

/*!
 * @brief Specialization for RAJA::reduce::detail::ValueLoc.
 */
template <typename T, bool doing_min, typename mempool>
class SoAPtr<RAJA::reduce::detail::ValueLoc<T, doing_min>, mempool>
{
  using value_type = RAJA::reduce::detail::ValueLoc<T, doing_min>;
  using first_type = T;
  using second_type = Index_type;

public:
  SoAPtr() = default;
  explicit SoAPtr(size_t size)
      : mem(mempool::getInstance().template malloc<first_type>(size)),
        mem_idx(mempool::getInstance().template malloc<second_type>(size))
  {
  }

  SoAPtr& allocate(size_t size)
  {
    mem = mempool::getInstance().template malloc<first_type>(size);
    mem_idx = mempool::getInstance().template malloc<second_type>(size);
    return *this;
  }

  SoAPtr& deallocate()
  {
    mempool::getInstance().free(mem);
    mem = nullptr;
    mempool::getInstance().free(mem_idx);
    mem_idx = nullptr;
    return *this;
  }

  RAJA_HOST_DEVICE bool allocated() const { return mem != nullptr; }

  RAJA_HOST_DEVICE value_type get(size_t i) const
  {
    return value_type(mem[i], mem_idx[i]);
  }
  RAJA_HOST_DEVICE void set(size_t i, value_type val)
  {
    mem[i] = val;
    mem_idx[i] = val.getLoc();
  }

private:
  first_type* mem = nullptr;
  second_type* mem_idx = nullptr;
};

}  // closing brace for detail namespace

}  // closing brace for RAJA namespace

#endif /* RAJA_SOA_PTR_HPP */
