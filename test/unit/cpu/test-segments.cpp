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
/// Source file containing tests for RAJA index set mechanics.
///

#include "gtest/gtest.h"
#include "RAJA/RAJA.hpp"


TEST(RangeStrideSegmentTest, sizes_no_roundoff)
{
  RAJA::RangeStrideSegment segment1(0, 20, 1);  
  ASSERT_EQ(segment1.size(), 20);
  
  RAJA::RangeStrideSegment segment2(0, 20, 2);  
  ASSERT_EQ(segment2.size(), 10);
  
  RAJA::RangeStrideSegment segment3(0, 20, 4);  
  ASSERT_EQ(segment3.size(), 5);
  
  RAJA::RangeStrideSegment segment4(0, 20, 5);  
  ASSERT_EQ(segment4.size(), 4);
  
  RAJA::RangeStrideSegment segment5(0, 20, 10);  
  ASSERT_EQ(segment5.size(), 2);
  
  RAJA::RangeStrideSegment segment6(0, 20, 20);  
  ASSERT_EQ(segment6.size(), 1);
}


TEST(RangeStrideSegmentTest, sizes_roundoff1)
{
  RAJA::RangeStrideSegment segment2(0, 21, 2);  
  ASSERT_EQ(segment2.size(), 11);
  
  RAJA::RangeStrideSegment segment3(0, 21, 4);  
  ASSERT_EQ(segment3.size(), 6);
  
  RAJA::RangeStrideSegment segment4(0, 21, 5);
  ASSERT_EQ(segment4.size(), 5);
  
  RAJA::RangeStrideSegment segment5(0, 21, 10);  
  ASSERT_EQ(segment5.size(), 3);
  
  RAJA::RangeStrideSegment segment6(0, 21, 20);  
  ASSERT_EQ(segment6.size(), 2);
}


TEST(RangeStrideSegmentTest, sizes_primes)
{
  RAJA::RangeStrideSegment segment1(0, 7, 3);  // should produce 0,3,6
  ASSERT_EQ(segment1.size(), 3);
  
  RAJA::RangeStrideSegment segment2(0, 13, 3); // shoudl produce 0,3,6,9,12
  ASSERT_EQ(segment2.size(), 5);
 
  RAJA::RangeStrideSegment segment3(0, 17, 5); // shoudl produce 0,5,10,15
  ASSERT_EQ(segment3.size(), 4);
}


TEST(RangeStrideSegmentTest, sizes_reverse_no_roundoff)
{
  RAJA::RangeStrideSegment segment1(19, -1, -1);  
  ASSERT_EQ(segment1.size(), 20);
  
  RAJA::RangeStrideSegment segment2(19, -1, -2);  
  ASSERT_EQ(segment2.size(), 10);
  
  RAJA::RangeStrideSegment segment3(19, -1, -4);  
  ASSERT_EQ(segment3.size(), 5);
  
  RAJA::RangeStrideSegment segment4(19, -1, -5);  
  ASSERT_EQ(segment4.size(), 4);
  
  RAJA::RangeStrideSegment segment5(19, -1, -10);  
  ASSERT_EQ(segment5.size(), 2);
  
  RAJA::RangeStrideSegment segment6(19, -1, -20);  
  ASSERT_EQ(segment6.size(), 1);
}


TEST(RangeStrideSegmentTest, sizes_reverse_roundoff1)
{  
  RAJA::RangeStrideSegment segment2(20, -1, -2);  
  ASSERT_EQ(segment2.size(), 11);
  
  RAJA::RangeStrideSegment segment3(20, -1, -4);  
  ASSERT_EQ(segment3.size(), 6);
  
  RAJA::RangeStrideSegment segment4(20, -1, -5);  
  ASSERT_EQ(segment4.size(), 5);
  
  RAJA::RangeStrideSegment segment5(20, -1, -10);  
  ASSERT_EQ(segment5.size(), 3);
  
  RAJA::RangeStrideSegment segment6(20, -1, -20);  
  ASSERT_EQ(segment6.size(), 2);
}


TEST(RangeStrideSegmentTest, values_forward_stride1)
{    
  RAJA::Index_type expected[] = {0,1,2,3,4,5};
  RAJA::RangeStrideSegment segment(0,6,1);
  
  ASSERT_EQ(segment.size(), 6);
 
  for(RAJA::Index_type i = 0;i < segment.size();++ i){
    ASSERT_EQ(segment.begin()[i], expected[i]);
  } 
  
  size_t j = 0;
  for(auto i : segment){
    ASSERT_EQ(i, expected[j]);
    ++ j;
  } 
}

TEST(RangeStrideSegmentTest, values_forward_stride3)
{    
  RAJA::Index_type expected[] = {0,3,6,9,12};
  RAJA::RangeStrideSegment segment(0,14,3);
  
  ASSERT_EQ(segment.size(), 5);
 
  for(RAJA::Index_type i = 0;i < segment.size();++ i){
    ASSERT_EQ(segment.begin()[i], expected[i]);
  } 
  
  size_t j = 0;
  for(auto i : segment){
    ASSERT_EQ(i, expected[j]);
    ++ j;
  } 
}

TEST(RangeStrideSegmentTest, values_reverse_stride1)
{    
  RAJA::Index_type expected[] = {5,4,3,2,1,0};
  RAJA::RangeStrideSegment segment(5,-1,-1);
  
  ASSERT_EQ(segment.size(), 6);
 
  for(RAJA::Index_type i = 0;i < segment.size();++ i){
    ASSERT_EQ(segment.begin()[i], expected[i]);
  } 
  
  size_t j = 0;
  for(auto i : segment){
    ASSERT_EQ(i, expected[j]);
    ++ j;
  } 
}


TEST(RangeStrideSegmentTest, values_reverse_stride1_negative)
{    
  RAJA::Index_type expected[] = {-10,-11,-12,-13};
  RAJA::RangeStrideSegment segment(-10,-14,-1);
  
  ASSERT_EQ(segment.size(), 4);
 
  for(RAJA::Index_type i = 0;i < segment.size();++ i){
    ASSERT_EQ(segment.begin()[i], expected[i]);
  } 
  
  size_t j = 0;
  for(auto i : segment){
    ASSERT_EQ(i, expected[j]);
    ++ j;
  } 
}


TEST(RangeStrideSegmentTest, zero_size)
{    
  RAJA::RangeStrideSegment segment(3,2,1);
  
  ASSERT_EQ(segment.size(), 0);
 
}

TEST(RangeStrideSegmentTest, zero_size_reverse)
{    
  RAJA::RangeStrideSegment segment(-3, 3,-1);
  
  ASSERT_EQ(segment.size(), 0);
 
}



TEST(RangeStrideSegmentTest, forall_values_forward_stride3)
{    
  RAJA::Index_type expected[] = {0,3,6,9,12};
  RAJA::RangeStrideSegment segment(0,14,3);
  
  ASSERT_EQ(segment.size(), 5);
 
  for(RAJA::Index_type i = 0;i < segment.size();++ i)
  {
    ASSERT_EQ(segment.begin()[i], expected[i]);
  } 
  
  size_t j = 0;
  
  
  for(auto i = segment.begin();i < segment.end();++i)
  {
    ASSERT_EQ(*i, expected[j++]);
  } 
  
  ASSERT_EQ((RAJA::Index_type)j, segment.size());
  
  
  j = 0;
  
  RAJA::forall<RAJA::seq_exec>(segment, [&](RAJA::Index_type i)
  {
    ASSERT_EQ(i, expected[j++]);
  }); 
  
  
  ASSERT_EQ((RAJA::Index_type)j, segment.size());
}


TEST(RangeStrideSegmentTest, forall_values_reverse_stride5)
{    
  RAJA::Index_type expected[] = {7,2,-3,-8};
  RAJA::RangeStrideSegment segment(7,-11,-5);
  
  ASSERT_EQ(segment.size(), 4);
 
  for(RAJA::Index_type i = 0;i < segment.size();++ i){
    ASSERT_EQ(segment.begin()[i], expected[i]);
  } 
  
  size_t j = 0; 
  
  for(auto i = segment.begin();i < segment.end();++i)
  {
    ASSERT_EQ(*i, expected[j++]);
  } 
  
  ASSERT_EQ((RAJA::Index_type)j, segment.size());
  
  j = 0;
  
  RAJA::forall<RAJA::seq_exec>(segment, [&](RAJA::Index_type i)
  {
    ASSERT_EQ(i, expected[j++]);
  }); 
  
  ASSERT_EQ((RAJA::Index_type)j, segment.size());
}


TEST(RangeStrideSegmentTest, iterator_begin_end)
{    
  RAJA::RangeStrideSegment segment(7,-11,-5);

  auto begin1 = segment.begin();  
  auto begin2 = std::begin(segment);  
  ASSERT_EQ(begin1, begin2);
  
  auto end1 = segment.end();  
  auto end2 = std::end(segment);  
  ASSERT_EQ(end1, end2);
  
}


TEST(RangeStrideSegmentTest, iterator_distance)
{ 
  {   
    RAJA::RangeStrideSegment segment1(0,10,1);  
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 10);
  } 
  
  { 
    RAJA::RangeStrideSegment segment1(10,20,1);  
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 10);
  }
  
  { 
    RAJA::RangeStrideSegment segment1(0,5,2);  
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 3); 
  }
  
  {
    RAJA::RangeStrideSegment segment1(10,20,2);  
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 5); 
  }
  
  {
    RAJA::RangeStrideSegment segment1(20,10,-2);  
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 5); 
  }
  
  {
    RAJA::RangeStrideSegment segment1(-10,10,3);  
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 7); 
  }
  
  
  {
    RAJA::RangeStrideSegment segment1(10,-10,-7);  
    ASSERT_EQ(std::distance(std::begin(segment1), std::end(segment1)), 3); 
  }
}


