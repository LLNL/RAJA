//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-24, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_policy_desul_HPP
#define RAJA_policy_desul_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_DESUL_ATOMICS)

#include "desul/atomics.hpp"

namespace RAJA
{

// Policy to perform an atomic operation with a given memory ordering.
template <typename OrderingPolicy, typename ScopePolicy>
struct detail_atomic_t {
};

using atomic_seq_cst =
    detail_atomic_t<desul::MemoryOrderSeqCst, desul::MemoryScopeDevice>;
using atomic_acq_rel =
    detail_atomic_t<desul::MemoryOrderAcqRel, desul::MemoryScopeDevice>;
using atomic_acquire =
    detail_atomic_t<desul::MemoryOrderAcquire, desul::MemoryScopeDevice>;
using atomic_release =
    detail_atomic_t<desul::MemoryOrderRelease, desul::MemoryScopeDevice>;
using atomic_relaxed =
    detail_atomic_t<desul::MemoryOrderRelaxed, desul::MemoryScopeDevice>;

using atomic_seq_cst_block =
    detail_atomic_t<desul::MemoryOrderSeqCst, desul::MemoryScopeCore>;
using atomic_acq_rel_block =
    detail_atomic_t<desul::MemoryOrderAcqRel, desul::MemoryScopeCore>;
using atomic_acquire_block =
    detail_atomic_t<desul::MemoryOrderAcquire, desul::MemoryScopeCore>;
using atomic_release_block =
    detail_atomic_t<desul::MemoryOrderRelease, desul::MemoryScopeCore>;
using atomic_relaxed_block =
    detail_atomic_t<desul::MemoryOrderRelaxed, desul::MemoryScopeCore>;

using atomic_seq_cst_sys =
    detail_atomic_t<desul::MemoryOrderSeqCst, desul::MemoryScopeSystem>;
using atomic_acq_rel_sys =
    detail_atomic_t<desul::MemoryOrderAcqRel, desul::MemoryScopeSystem>;
using atomic_acquire_sys =
    detail_atomic_t<desul::MemoryOrderAcquire, desul::MemoryScopeSystem>;
using atomic_release_sys =
    detail_atomic_t<desul::MemoryOrderRelease, desul::MemoryScopeSystem>;
using atomic_relaxed_sys =
    detail_atomic_t<desul::MemoryOrderRelaxed, desul::MemoryScopeSystem>;

}  // namespace RAJA

#endif  // RAJA_ENABLE_DESUL_ATOMICS

#endif
