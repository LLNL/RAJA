//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"

#include <exception>

extern "C" void kokkosp_init_library(const int RAJA_UNUSED_ARG(loadSeq),
	const uint64_t RAJA_UNUSED_ARG(interfaceVer),
	const uint32_t RAJA_UNUSED_ARG(devInfoCount),
	void* RAJA_UNUSED_ARG(deviceInfo)) {}

extern "C" void kokkosp_begin_parallel_for(const char* RAJA_UNUSED_ARG(name),
    const uint32_t RAJA_UNUSED_ARG(devID),
    uint64_t* RAJA_UNUSED_ARG(kID)) {
    throw std::runtime_error("preLaunch");
}

extern "C" void kokkosp_end_parallel_for(const uint64_t RAJA_UNUSED_ARG(kID)) {}

extern "C" void kokkosp_finalize_library() {}
