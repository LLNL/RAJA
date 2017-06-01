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
/// Source file containing tests for all basic RAJA CPU forall patterns.
///

#include <cstdlib>

#include <string>

#include "RAJA/RAJA.hpp"
#include "gtest/gtest.h"

using namespace RAJA;
using namespace std;

#include "Compare.hpp"
#include "buildIndexSet.hpp"

template <typename ISET_POLICY_T>
class ForallTest : public ::testing::Test
{
protected:
  Real_ptr in_array;
  Index_type alen;
  IndexSet iset;
  RAJAVec<Index_type> is_indices;
  Real_ptr test_array;
  Real_ptr ref_icount_array;
  Real_ptr ref_forall_array;

  virtual void SetUp()
  {
    // AddSegments chosen arbitrarily; index set equivalence is tested elsewhere
    alen = buildIndexSet(&iset, IndexSetBuildMethod::AddSegments) + 1;

    in_array = (Real_ptr)allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));

    for (Index_type i = 0; i < alen; ++i) {
      in_array[i] = Real_type(rand() % 65536);
    }

    getIndices(is_indices, iset);

    test_array =
        (Real_ptr)allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));
    ref_icount_array =
        (Real_ptr)allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));
    ref_forall_array =
        (Real_ptr)allocate_aligned(DATA_ALIGN, alen * sizeof(Real_type));

    for (Index_type i = 0; i < alen; ++i) {
      test_array[i] = 0.0;
      ref_forall_array[i] = 0.0;
      ref_icount_array[i] = 0.0;
    }

    for (size_t i = 0; i < is_indices.size(); ++i) {
      ref_forall_array[is_indices[i]] =
          in_array[is_indices[i]] * in_array[is_indices[i]];
    }

    for (size_t i = 0; i < is_indices.size(); ++i) {
      ref_icount_array[i] = in_array[is_indices[i]] * in_array[is_indices[i]];
    }
  }

  virtual void TearDown()
  {
    free_aligned(in_array);
    free_aligned(test_array);
    free_aligned(ref_icount_array);
    free_aligned(ref_forall_array);
  }
};

TYPED_TEST_CASE_P(ForallTest);

TYPED_TEST_P(ForallTest, BasicForall)
{
  forall<TypeParam>(this->iset, [=](Index_type idx) {
    this->test_array[idx] = this->in_array[idx] * this->in_array[idx];
  });

  for (Index_type i = 0; i < this->alen; ++i) {
    EXPECT_EQ(this->ref_forall_array[i], this->test_array[i]);
  }
}

TYPED_TEST_P(ForallTest, BasicForallIcount)
{
  forall_Icount<TypeParam>(this->iset, [=](Index_type icount, Index_type idx) {
    this->test_array[icount] = this->in_array[idx] * this->in_array[idx];
  });

  for (Index_type i = 0; i < this->alen; ++i) {
    EXPECT_EQ(this->ref_icount_array[i], this->test_array[i]);
  }
}

REGISTER_TYPED_TEST_CASE_P(ForallTest, BasicForall, BasicForallIcount);

using SequentialTypes =
    ::testing::Types<IndexSet::ExecPolicy<seq_segit, seq_exec>,
                     IndexSet::ExecPolicy<seq_segit, simd_exec> >;

INSTANTIATE_TYPED_TEST_CASE_P(Sequential, ForallTest, SequentialTypes);


#if defined(RAJA_ENABLE_OPENMP)
using OpenMPTypes =
    ::testing::Types<IndexSet::ExecPolicy<seq_segit, omp_parallel_for_exec>,
                     IndexSet::ExecPolicy<omp_parallel_for_segit, seq_exec>,
                     IndexSet::ExecPolicy<omp_parallel_for_segit, simd_exec> >;

INSTANTIATE_TYPED_TEST_CASE_P(OpenMP, ForallTest, OpenMPTypes);
#endif

#if defined(RAJA_ENABLE_CILK)
using CilkTypes =
    ::testing::Types<IndexSet::ExecPolicy<seq_segit, cilk_for_exec>,
                     IndexSet::ExecPolicy<cilk_for_segit, seq_exec>,
                     IndexSet::ExecPolicy<cilk_for_segit, simd_exec> >;

INSTANTIATE_TYPED_TEST_CASE_P(Cilk, ForallTest, CilkTypes);
#endif
