#include "gtest/gtest.h"

#include "RAJA/pattern/nested.hpp"

#include <iostream>

TEST(Nested, Basic)
{
  using namespace RAJA::nested;
  using Index_type = RAJA::Index_type;
  using pol = Policy<RAJA::seq_exec, RAJA::seq_exec, For<RAJA::seq_exec, 0>, RAJA::seq_exec, For<RAJA::seq_exec, 1>>;
  RAJA::nested::forall(pol{},
                       RAJA::util::make_tuple(RAJA::RangeSegment(0, 5),
                                              RAJA::RangeSegment(0, 5)),
                       [=](Index_type i, Index_type j) {
                         std::cout << i << " " << j << std::endl;
                       });
}
