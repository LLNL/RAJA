//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "camp/camp.hpp"
#include "gtest/gtest.h"

TEST(CampTuple, AssignCompat)
{
  // Compatible, though different, tuples are assignable
  const camp::tuple<long long, char> t(5, 'a');
  ASSERT_EQ(camp::get<0>(t), 5);
  ASSERT_EQ(camp::get<1>(t), 'a');

  camp::tagged_tuple<camp::list<int, char>, int, char> t2;
  t2 = t;
  ASSERT_EQ(camp::get<0>(t2), 5);
  ASSERT_EQ(camp::get<1>(t2), 'a');
}

TEST(CampTuple, Assign)
{
  camp::tuple<int, char> t(5, 'a');
  ASSERT_EQ(camp::get<0>(t), 5);
  ASSERT_EQ(camp::get<1>(t), 'a');

  camp::tuple<int, char> t2 = t;
  ASSERT_EQ(camp::get<0>(t2), 5);
  ASSERT_EQ(camp::get<1>(t2), 'a');
}

TEST(CampTuple, ForwardAsTuple)
{
  int a, b;
  [](camp::tuple<int &, int &, int &&> t) {
    ASSERT_EQ(camp::get<2>(t), 5);
    camp::get<1>(t) = 3;
    camp::get<2>(t) = 3;
    ASSERT_EQ(camp::get<1>(t), 3);
    ASSERT_EQ(camp::get<2>(t), 3);
  }(camp::forward_as_tuple(a, b, int{5}));
}

TEST(CampTuple, GetByIndex)
{
  camp::tuple<int, char> t(5, 'a');
  ASSERT_EQ(camp::get<0>(t), 5);
  ASSERT_EQ(camp::get<1>(t), 'a');
}

TEST(CampTuple, GetByType)
{
  camp::tuple<int, char> t(5, 'a');
  ASSERT_EQ(camp::get<int>(t), 5);
  ASSERT_EQ(camp::get<char>(t), 'a');
}

TEST(CampTuple, CatPair)
{
  auto t1 = camp::make_tuple(5, 'a');
  auto t2 = camp::make_tuple(5.1f, "meh");
  auto t3 = tuple_cat_pair(t1,
                           camp::make_idx_seq_t<2>{},
                           t2,
                           camp::make_idx_seq_t<2>{});
  ASSERT_EQ(camp::get<1>(t3), 'a');
  ASSERT_EQ(camp::get<2>(t3), 5.1f);

  auto t4 = tuple_cat_pair(t1, t2);

  ASSERT_EQ(camp::get<1>(t4), 'a');
  ASSERT_EQ(camp::get<2>(t4), 5.1f);

  auto t5 =
      tuple_cat_pair(t1, camp::idx_seq<1, 0>{}, t2, camp::idx_seq<1, 0>{});
  ASSERT_EQ(camp::get<0>(t5), 'a');
  ASSERT_EQ(camp::get<3>(t5), 5.1f);
}

struct NoDefCon {
  NoDefCon() = delete;
  NoDefCon(int i) : num{i} { (void)num; }
  NoDefCon(NoDefCon const &) = default;

private:
  int num;
};

TEST(CampTuple, NoDefault) { camp::tuple<NoDefCon> t(NoDefCon(1)); }

struct s1;
struct s2;
struct s3;

TEST(CampTaggedTuple, GetByType)
{
  camp::tagged_tuple<camp::list<s1, s2>, int, char> t(5, 'a');
  ASSERT_EQ(camp::get<s1>(t), 5);
  ASSERT_EQ(camp::get<s2>(t), 'a');
  camp::get<s1>(t) = 15;
  ASSERT_EQ(camp::get<s1>(t), 15);
}

TEST(CampTaggedTuple, MakeTagged)
{
  auto t = camp::make_tagged_tuple<camp::list<s1, s2>>(5, 'a');
  ASSERT_EQ(camp::get<s1>(t), 5);
  ASSERT_EQ(camp::get<s2>(t), 'a');
  camp::get<s1>(t) = 15;
  ASSERT_EQ(camp::get<s1>(t), 15);
}
