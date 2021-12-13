//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA_test-base.hpp"

template < typename layout, typename IdxLin, typename IdxI, typename IdxJ >
void test_layout_2d(layout const& l_a,
                    IdxLin size,
                    IdxI beginI, IdxI endI, bool projI,
                    IdxJ beginJ, IdxJ endJ, bool projJ)
{
  // Construct using copy ctor
  const layout l_b(l_a);

  // Test default ctor and assignment operator
  layout l;
  l = l_b;

  ASSERT_EQ(size, l.size());
  ASSERT_EQ(size, l_a.size());
  ASSERT_EQ(size, l_b.size());

  IdxLin strideI = (projI ? IdxLin{0} : IdxLin{1}) *
                   (projJ ? IdxLin{1} : IdxLin{RAJA::stripIndexType(endJ-beginJ)});
  IdxLin strideJ = (projJ ? IdxLin{0} : IdxLin{1});

  for (IdxI i = beginI; i < endI; ++i) {
    for (IdxJ j = beginJ; j < endJ; ++j) {

      IdxLin count = strideI * IdxLin{i-beginI} +
                     strideJ * IdxLin{j-beginJ} ;
      // forward map
      ASSERT_EQ(count, l(i, j));

      // check with a and b
      ASSERT_EQ(count, l_a(i, j));
      ASSERT_EQ(count, l_b(i, j));

      // inverse map
      IdxI i_from_count;
      IdxJ j_from_count;
      l.toIndices(count, i_from_count, j_from_count);

      ASSERT_EQ(i_from_count, projI ? IdxI{0} : i);
      ASSERT_EQ(j_from_count, projJ ? IdxJ{0} : j);

      // check with a and b
      IdxI i_a_from_count;
      IdxJ j_a_from_count;
      l_a.toIndices(count, i_a_from_count, j_a_from_count);

      ASSERT_EQ(i_a_from_count, projI ? IdxI{0} : i);
      ASSERT_EQ(j_a_from_count, projJ ? IdxJ{0} : j);

      IdxI i_b_from_count;
      IdxJ j_b_from_count;
      l_b.toIndices(count, i_b_from_count, j_b_from_count);

      ASSERT_EQ(i_b_from_count, projI ? IdxI{0} : i);
      ASSERT_EQ(j_b_from_count, projJ ? IdxJ{0} : j);
    }
  }
}

template < typename layout, typename IdxLin, typename IdxI, typename IdxJ, typename IdxK >
void test_layout_3d(layout const& l_a,
                    IdxLin size,
                    IdxI beginI, IdxI endI, bool projI,
                    IdxJ beginJ, IdxJ endJ, bool projJ,
                    IdxK beginK, IdxK endK, bool projK)
{
  // Construct using copy ctor
  const layout l_b(l_a);

  // Test default ctor and assignment operator
  layout l;
  l = l_b;

  ASSERT_EQ(size, l.size());
  ASSERT_EQ(size, l_a.size());
  ASSERT_EQ(size, l_b.size());

  IdxLin strideI = (projI ? IdxLin{0} : IdxLin{1}) *
                   (projJ ? IdxLin{1} : IdxLin{RAJA::stripIndexType(endJ-beginJ)}) *
                   (projK ? IdxLin{1} : IdxLin{RAJA::stripIndexType(endK-beginK)});
  IdxLin strideJ = (projJ ? IdxLin{0} : IdxLin{1}) *
                   (projK ? IdxLin{1} : IdxLin{RAJA::stripIndexType(endK-beginK)});
  IdxLin strideK = (projK ? IdxLin{0} : IdxLin{1});

  for (IdxI i = beginI; i < endI; ++i) {
    for (IdxJ j = beginJ; j < endJ; ++j) {
      for (IdxK k = beginK; k < endK; ++k) {

        IdxLin count = strideI * IdxLin{i-beginI} +
                       strideJ * IdxLin{j-beginJ} +
                       strideK * IdxLin{k-beginK} ;

        // forward map
        ASSERT_EQ(count, l(i, j, k));

        // check with a and b
        ASSERT_EQ(count, l_a(i, j, k));
        ASSERT_EQ(count, l_b(i, j, k));

        // inverse map
        IdxI i_from_count;
        IdxJ j_from_count;
        IdxJ k_from_count;
        l.toIndices(count, i_from_count, j_from_count, k_from_count);

        ASSERT_EQ(i_from_count, projI ? IdxI{0} : i);
        ASSERT_EQ(j_from_count, projJ ? IdxJ{0} : j);
        ASSERT_EQ(k_from_count, projK ? IdxK{0} : k);

        // check with a and b
        IdxI i_a_from_count;
        IdxJ j_a_from_count;
        IdxJ k_a_from_count;
        l_a.toIndices(count, i_a_from_count, j_a_from_count, k_a_from_count);

        ASSERT_EQ(i_a_from_count, projI ? IdxI{0} : i);
        ASSERT_EQ(j_a_from_count, projJ ? IdxJ{0} : j);
        ASSERT_EQ(k_a_from_count, projK ? IdxK{0} : k);

        IdxI i_b_from_count;
        IdxJ j_b_from_count;
        IdxJ k_b_from_count;
        l_b.toIndices(count, i_b_from_count, j_b_from_count, k_b_from_count);

        ASSERT_EQ(i_b_from_count, projI ? IdxI{0} : i);
        ASSERT_EQ(j_b_from_count, projJ ? IdxJ{0} : j);
        ASSERT_EQ(k_b_from_count, projK ? IdxK{0} : k);
      }
    }
  }
}
