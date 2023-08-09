//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-23, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <gtest/gtest.h>
#include "RAJA/util/types.hpp"
#include "RAJA_test-base.hpp"

using namespace RAJA;

TEST(IndexLayout, IndexList1D) {

  Index_type arr[3] = {1,2,3};

  auto index_tuple = make_index_tuple(IndexList<>(&arr[0]));
  auto index_layout = make_index_layout(index_tuple, 3);

  EXPECT_EQ(index_layout(0), 1);
  EXPECT_EQ(index_layout(1), 2);
  EXPECT_EQ(index_layout(2), 3);

}

TEST(IndexLayout, IndexList1DSubsetOfLayout) {

  Index_type arr[3] = {2,3,4};

  auto index_tuple = make_index_tuple(IndexList<>(&arr[0]));
  auto index_layout = make_index_layout(index_tuple, 5);

  EXPECT_EQ(index_layout(0), 2);
  EXPECT_EQ(index_layout(1), 3);
  EXPECT_EQ(index_layout(2), 4);

}


TEST(IndexLayout, ExtractTwoIndices2DLayoutAxis0) {

  Index_type arr[2] = {1,2};

  auto index_tuple = make_index_tuple(IndexList<>(&arr[0]), DirectIndex<>());
  auto index_layout = make_index_layout(index_tuple, 3, 10);

  for (int i = 0; i < 10; i++ ) {
    EXPECT_EQ(index_layout(0,i), i+10);
    EXPECT_EQ(index_layout(1,i), i+20);
  }

}

TEST(IndexLayout, ExtractTwoIndices2DLayoutAxis1) {

  Index_type arr[2] = {9,5};

  auto index_tuple = make_index_tuple(DirectIndex<>(), IndexList<>(&arr[0]));
  auto index_layout = make_index_layout(index_tuple, 3, 10);

  EXPECT_EQ(index_layout(0,0), 9);
  EXPECT_EQ(index_layout(0,1), 5);
  EXPECT_EQ(index_layout(1,0), 19);
  EXPECT_EQ(index_layout(1,1), 15);
  EXPECT_EQ(index_layout(2,0), 29);
  EXPECT_EQ(index_layout(2,1), 25);

}

TEST(IndexLayout, ExtractOneIndex2DLayoutAxis0) {

  Index_type arr[1] = {2};

  auto index_tuple = make_index_tuple(IndexList<>(&arr[0]), DirectIndex<>());
  auto index_layout = make_index_layout(index_tuple, 3, 3);

  EXPECT_EQ(index_layout(0,0), 6);
  EXPECT_EQ(index_layout(0,1), 7);
  EXPECT_EQ(index_layout(0,2), 8);  

}

TEST(IndexLayout, IndexList2DLayoutExtractOneIndex) {

  Index_type arr[1] = {2};

  auto index_tuple = make_index_tuple(DirectIndex<>(), IndexList<>(&arr[0]));
  auto index_layout = make_index_layout(index_tuple, 3, 3);

  EXPECT_EQ(index_layout(0,0), 2);
  EXPECT_EQ(index_layout(1,0), 5);
  EXPECT_EQ(index_layout(2,0), 8);

}

TEST(IndexLayout, ConditionalIndexListNullPtr) {

  Index_type* arr_ptr = nullptr;

  auto index_tuple = make_index_tuple(ConditionalIndexList<>(arr_ptr));
  auto index_layout = make_index_layout(index_tuple, 3);

  EXPECT_EQ(index_layout(0), 0);
  EXPECT_EQ(index_layout(1), 1);
  EXPECT_EQ(index_layout(2), 2);
}

TEST(IndexLayout, View1DLayout)
{
  Index_type data[5] = {5,10,15,20,25};
  Index_type index_list[3] = {4,2,3};

  auto index_tuple = make_index_tuple(IndexList<>(&index_list[0]));
  auto index_layout = make_index_layout(index_tuple, 5);

  auto view = make_index_view(&data[0], index_layout);

  EXPECT_EQ(view(0), 25);
  EXPECT_EQ(view(1), 15);
  EXPECT_EQ(view(2), 20);

}

TEST(IndexLayout, View2DLayout)
{
  Index_type data[2][3];

  for (int i = 0; i < 2; i ++ )
    for (int j = 0; j < 3; j ++ )
      data[i][j] = i*j;

  Index_type index_list[2] = {1,2};

  auto index_tuple = make_index_tuple(DirectIndex<>(), IndexList<>(&index_list[0]));
  auto index_layout = make_index_layout(index_tuple, 2, 3);

  auto view = make_index_view(&data[0][0], index_layout);

  for (int i = 0; i < 2; i ++ )
    for (int j = 0; j < 2; j ++ )
      EXPECT_EQ(view(i,j), i*(j+1));

}

TEST(IndexLayout, View3DLayout)
{
  Index_type data[2][3][4];

  for (int i = 0; i < 2; i ++ )
    for (int j = 0; j < 3; j ++ )
      for (int k = 0; k < 4; k ++ )
      data[i][j][k] = i*j*k;

  Index_type index_list_j[2] = {1,2};
  Index_type index_list_k[2] = {2,3};

  auto index_tuple = make_index_tuple(DirectIndex<>(), 
                                      IndexList<>(&index_list_j[0]),
                                      IndexList<>(&index_list_k[0]));

  auto index_layout = make_index_layout(index_tuple, 2, 3, 4);

  auto view = make_index_view(&data[0][0][0], index_layout);

  for (int i = 0; i < 2; i ++ )
    for (int j = 0; j < 2; j ++ )
      for (int k = 0; k < 2; k ++ )
        EXPECT_EQ(view(i,j,k), i*(j+1)*(k+2));

}

TEST(IndexLayout, MultiView1DLayout)
{
  Index_type data_squared[4];
  Index_type data_cubed[4];

  for (int i = 0; i < 4; i ++ ) data_squared[i] = i*i;
  for (int i = 0; i < 4; i ++ ) data_cubed[i] = i*i*i;

  Index_type* data_array[2];
  data_array[0] = data_squared;
  data_array[1] = data_cubed;

  Index_type index_list[2] = {1,2};

  auto index_tuple = make_index_tuple(IndexList<>(&index_list[0]));
  auto index_layout = make_index_layout(index_tuple, 4);

  auto view = MultiView<Index_type, IndexLayout<1, Index_type, IndexList<> > >(data_array, index_layout);

  for (int i = 0; i < 2; i ++ ) {
    EXPECT_EQ(view(0,i), data_squared[i+1]);
    EXPECT_EQ(view(1,i), data_cubed[i+1]);
  }

}

