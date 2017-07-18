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
// For additional details, please also read RAJA/README.
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

///
/// Source file containing tests for basic multipolicy operation
///

#include "gtest/gtest.h"

#include "RAJA/RAJA.hpp"

TEST(multipolicy, basic)
{
  auto mp =
      RAJA::make_multi_policy<RAJA::seq_exec, RAJA::omp_parallel_for_exec>(
          [](const RAJA::RangeSegment& r) {
            if (r.size() < 100) {
              return 0;
            } else {
              return 1;
            }
          });
  RAJA::forall(mp, RAJA::RangeSegment(0, 5), [](RAJA::Index_type) {
    ASSERT_EQ(omp_get_num_threads(), 1);
  });
  RAJA::forall(mp, RAJA::RangeSegment(0, 101), [](RAJA::Index_type) {
    ASSERT_TRUE(omp_get_num_threads() > 1);
  });
  // Nest a multipolicy to ensure value-based policies are preserved
  auto mp2 =
      RAJA::make_multi_policy(std::make_tuple(RAJA::omp_parallel_for_exec{},
                                              mp),
                              [](const RAJA::RangeSegment& r) {
                                if (r.size() > 10 && r.size() < 90) {
                                  return 0;
                                } else {
                                  return 1;
                                }
                              });
  RAJA::forall(mp2, RAJA::RangeSegment(0, 5), [](RAJA::Index_type) {
    ASSERT_EQ(omp_get_num_threads(), 1);
  });
  RAJA::forall(mp2, RAJA::RangeSegment(0, 91), [](RAJA::Index_type) {
    ASSERT_EQ(omp_get_num_threads(), 1);
  });
  RAJA::forall(mp2, RAJA::RangeSegment(0, 50), [](RAJA::Index_type) {
    ASSERT_TRUE(omp_get_num_threads() > 1);
  });
}
