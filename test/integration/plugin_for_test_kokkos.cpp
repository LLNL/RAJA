
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"

#include <exception>

extern "C" void kokkosp_init_library(const int loadSeq,
	const uint64_t interfaceVer,
	const uint32_t devInfoCount,
	void* deviceInfo) {
    (void)loadSeq;
    (void)interfaceVer;
    (void)devInfoCount;
    (void)deviceInfo;
}

extern "C" void kokkosp_begin_parallel_for(const char* name, const uint32_t devID, uint64_t* kID) {
    throw std::runtime_error("preLaunch");
    (void)name;
    (void)devID;
    (void)kID;
}

extern "C" void kokkosp_end_parallel_for(const uint64_t kID) {
    (void)kID;
}

extern "C" void kokkosp_finalize_library() {}