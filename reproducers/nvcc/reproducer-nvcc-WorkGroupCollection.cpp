//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

///
/// Source file containing nvcc WorkGroupCollection compilation issue reproducer.
///
/// Immediate Issue: Can not compile WorkGroupCollection code.
///
/// Larger Issue: Can't build more general graph API.
///
/// Seen when compiling for sm_70 on Sierra systems (IBM p9, Nvidia V100)
///
/// Notes: The issue appears to be in the stub code nvcc generates for a global
/// function used to get a device function pointer.
///

#include <cassert>
#include <cstdio>

#include "RAJA/RAJA.hpp"


template < typename Resource >
struct ResourceAllocator
{
  template < typename T >
  struct std_allocator
  {
    using value_type = T;

    std_allocator() = default;

    std_allocator(Resource& res)
      : m_res(res)
    { }

    std_allocator(std_allocator const&) = default;
    std_allocator(std_allocator &&) = default;

    std_allocator& operator=(std_allocator const&) = default;
    std_allocator& operator=(std_allocator &&) = default;

    template < typename U >
    std_allocator(std_allocator<U> const& other) noexcept
      : m_res(other.get_resource())
    { }

    /*[[nodiscard]]*/
    value_type* allocate(size_t num)
    {
      if (num > std::numeric_limits<size_t>::max() / sizeof(value_type)) {
        throw std::bad_alloc();
      }

      value_type* ptr = m_res.template allocate<value_type>(num);

      if (!ptr) {
        throw std::bad_alloc();
      }

      return ptr;
    }

    void deallocate(value_type* ptr, size_t) noexcept
    {
      m_res.deallocate(ptr);
    }

    Resource const& get_resource() const
    {
      return m_res;
    }

    template <typename U>
    friend inline bool operator==(std_allocator const& /*lhs*/, std_allocator<U> const& /*rhs*/)
    {
      return true; // lhs.get_resource() == rhs.get_resource(); // TODO not equality comparable yet
    }

    template <typename U>
    friend inline bool operator!=(std_allocator const& lhs, std_allocator<U> const& rhs)
    {
      return !(lhs == rhs);
    }

  private:
    Resource m_res = Resource::get_default();
  };
};

struct test_functional
{
  int* working_array;
  int test_val;

  RAJA_HOST_DEVICE
  void operator()(int i) const
  {
    working_array[i] += i + test_val;
  }
};

int main(int, char**)
{
  using GraphPolicy = RAJA::loop_graph;
  using WorkGroupExecPolicy = RAJA::cuda_work<256>;
  using OrderPolicy = RAJA::unordered_cuda_loop_y_block_iter_x_threadblock_average;
  using Allocator = typename ResourceAllocator<camp::resources::Cuda>::template std_allocator<char>;
  using WORKING_RES = camp::resources::Cuda;
  using ForallExecPolicy = RAJA::cuda_exec<256>;

  using WorkGroupCollection_type = RAJA::expt::graph::WorkGroupCollection<
                                     WorkGroupExecPolicy,
                                     OrderPolicy,
                                     int,
                                     RAJA::xargs<>,
                                     Allocator
                                   >;

  int begin = 6;
  int end = 26;

  assert(begin >= 0);
  assert(end >= begin);
  int N = end + begin;

  WORKING_RES res = WORKING_RES::get_default();
  camp::resources::Resource work_res{res};
  camp::resources::Resource host_res{camp::resources::Host()};

  int* working_array = work_res.allocate<int>(N);
  int* check_array   = host_res.allocate<int>(N);
  int* test_array    = host_res.allocate<int>(N);

  {
    for (int i = 0; i < N; i++) {
      test_array[i] = 0;
    }

    res.memcpy(working_array, test_array, sizeof(int) * N);

    for (int i = begin; i < end; ++i) {
      test_array[ i ] = i;
    }
  }

  RAJA::expt::graph::DAG g;
  RAJA::expt::graph::DAG::CollectionView<WorkGroupCollection_type> collection =
      g.add_collection(RAJA::expt::graph::WorkGroup< WorkGroupExecPolicy,
                                                     OrderPolicy,
                                                     int,
                                                     RAJA::xargs<>,
                                                     Allocator
                                                   >(Allocator{}));

  int test_val(5);

  {
    g.add_collection_node(collection, RAJA::expt::graph::FusibleForall<ForallExecPolicy>(
      RAJA::TypedRangeSegment<int>{ begin, end },
      test_functional{working_array, test_val}
    //   [=] RAJA_HOST_DEVICE (int i) {
    //   working_array[i] += i + test_val;
    // }
    ));
  }

  RAJA::expt::graph::DAGExec<GraphPolicy, WORKING_RES> ge =
      g.template instantiate<GraphPolicy, WORKING_RES>();
  camp::resources::Event e = ge.exec();
  e.wait();

  bool success = true;
  {
    res.memcpy(check_array, working_array, sizeof(int) * N);

    for (int i = 0;      i < begin; i++) {
      success = success && (test_array[i] == check_array[i]);
    }
    for (int i = begin;  i < end;   i++) {
      success = success && (test_array[i] + test_val == check_array[i]);
    }
    for (int i = end;    i < N;     i++) {
      success = success && (test_array[i] == check_array[i]);
    }
  }


  work_res.deallocate(working_array);

  host_res.deallocate(check_array);
  host_res.deallocate(test_array);

  if (!success) {
    printf("Got wrong answer.\n");
  } else {
    printf("Got correct answer.\n");
  }

  return !success;
}
