//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/LICENSE file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include "RAJA/RAJA.hpp"
#include "RAJA_gtest.hpp"

#include "camp/resource.hpp"

#include <cstdio>

#if defined(RAJA_ENABLE_CUDA)
#include <cuda_runtime.h>
#endif


using namespace RAJA;
using namespace RAJA::statement;



using layout_2d = Layout<2, RAJA::Index_type>;
using view_2d = View<Index_type, layout_2d>;
static constexpr Index_type x_len = 5;
static constexpr Index_type y_len = 5;


RAJA_INDEX_VALUE(TypedIndex, "TypedIndex");
RAJA_INDEX_VALUE(ZoneI, "ZoneI");
RAJA_INDEX_VALUE(ZoneJ, "ZoneJ");
RAJA_INDEX_VALUE(ZoneK, "ZoneK");


template <typename NestedPolicy>
class Kernel : public ::testing::Test
{
protected:
  Index_type *data;
  view_2d view{nullptr, x_len, y_len};

  virtual void SetUp()
  {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaMallocManaged(&data,
                      sizeof(Index_type) * x_len * y_len,
                      cudaMemAttachGlobal));
#else
    data = new Index_type[x_len * y_len];
#endif
    view.set_data(data);
  }

  virtual void TearDown()
  {
#if defined(RAJA_ENABLE_CUDA)
    cudaErrchk(cudaFree(data));
#else
    delete[] data;
#endif
  }
};
TYPED_TEST_SUITE_P(Kernel);

#if defined(RAJA_ENABLE_HIP)
template <typename NestedPolicy>
class Kernel_gpu : public ::testing::Test
{
protected:
  Index_type* data;
  Index_type* d_data;
  view_2d view{nullptr, x_len, y_len};
  view_2d d_view{nullptr, x_len, y_len};

  virtual void SetUp()
  {
    hipErrchk(hipMalloc(&d_data,sizeof(Index_type) * x_len * y_len));
    d_view.set_data(d_data);

    data = new Index_type[x_len * y_len];
    view.set_data(data);
  }

  virtual void TearDown()
  {
    hipErrchk(hipFree(d_data));
    delete[] data;
  }
};
TYPED_TEST_SUITE_P(Kernel_gpu);
#endif

RAJA_HOST_DEVICE constexpr Index_type get_val(Index_type v) noexcept
{
  return v;
}
template <typename T>
RAJA_HOST_DEVICE constexpr Index_type get_val(T v) noexcept
{
  return *v;
}
GPU_TYPED_TEST_P(Kernel, Basic)
{
  using Pol = at_v<TypeParam, 0>;
  using Res = at_v<TypeParam, 1>;
  using IndexTypes = at_v<TypeParam, 2>;
  using Idx0 = at_v<IndexTypes, 0>;
  using Idx1 = at_v<IndexTypes, 1>;
  RAJA::ReduceSum<at_v<TypeParam, 3>, RAJA::Real_type> tsum(0.0);
  RAJA::ReduceMin<at_v<TypeParam, 3>, RAJA::Real_type> tMin(0.0);
  RAJA::ReduceMax<at_v<TypeParam, 3>, RAJA::Real_type> tMax(0.0);
  RAJA::Real_type total{0.0};
  auto ranges = RAJA::make_tuple(RAJA::TypedRangeSegment<Idx0>(0, x_len),
                                 RAJA::TypedRangeSegment<Idx1>(0, y_len));
  auto v = this->view;
  RAJA::kernel<Pol>(ranges, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
    // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
    v(get_val(i), j) = get_val(i) * x_len + j;
    tsum += get_val(i) * 1.1 + j;
  });
  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(this->view(i, j), i * x_len + j);
      total += i * 1.1 + j;
    }
  }
  ASSERT_FLOAT_EQ((double)total, (double)tsum.get());

  camp::resources::Resource work_res{Res()};

  // Check reduction
  int stride1 = 5;
  int arr_len = stride1 * stride1;

  double *arr;
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMallocManaged(&arr, arr_len * sizeof(double)));
#else
  arr = new double[arr_len];
#endif

  for (int i = 0; i < arr_len; ++i) {
    arr[i] = i;
  }

  // set the min and max of the array
  arr[4] = -1;
  arr[8] = 50;

  tsum.reset(0.0);
  auto ranges2 = RAJA::make_tuple(RAJA::TypedRangeSegment<Idx0>(0, stride1),
                                  RAJA::TypedRangeSegment<Idx1>(0, stride1));

  RAJA::kernel<Pol>(ranges, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
    // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
    tsum += get_val(i) * 1.1 + get_val(j);
  });


  RAJA::kernel<Pol>(ranges2, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
    // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
    RAJA::Index_type id = get_val(j) + get_val(i) * stride1;
    tMin.min(arr[id]);
    tMax.max(arr[id]);
  });

  tMin.reset(0.0);
  tMax.reset(0.0);

  RAJA::kernel<Pol>(ranges2, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
    // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
    RAJA::Index_type id = get_val(j) + get_val(i) * stride1;
    tMin.min(arr[id]);

    tMax.max(arr[id]);
  });

  ASSERT_FLOAT_EQ((double)total, (double)tsum.get());
  ASSERT_FLOAT_EQ((double)-1, (double)tMin.get());
  ASSERT_FLOAT_EQ((double)50, (double)tMax.get());

  std::vector<Idx0> idx_x;
  std::vector<Idx1> idx_y;

  for (int i = 0; i < x_len; ++i)
    idx_x.push_back(static_cast<Idx0>(i));
  for (int i = 0; i < y_len; ++i)
    idx_y.push_back(static_cast<Idx1>(i));

  tsum.reset(0.0);
  total = 0.0;
  RAJA::TypedListSegment<Idx0> idx_list(&idx_x[0], idx_x.size(), work_res);
  RAJA::TypedListSegment<Idx1> idy_list(&idx_y[0], idx_y.size(), work_res);
  auto rangeList = RAJA::make_tuple(idx_list, idy_list);

  RAJA::kernel<Pol>(rangeList, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
    // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
    v(get_val(i), j) = get_val(i) * x_len + j;
    tsum += get_val(i) * 1.1 + j;
  });


  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(this->view(i, j), i * x_len + j);
      total += i * 1.1 + j;
    }
  }
  ASSERT_FLOAT_EQ((double)total, (double)tsum.get());

  total = 0.0;
  tsum.reset(0.0);
  double *idx_test;
#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaMallocManaged(&idx_test,
                    sizeof(double) * x_len * y_len,
                    cudaMemAttachGlobal));
#else
  idx_test = new double[x_len * y_len];
#endif

  auto iterSpace2 =
      RAJA::make_tuple(RAJA::TypedRangeSegment<Idx0>(0, x_len), idy_list);
  RAJA::kernel<Pol>(iterSpace2, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
    Index_type id = get_val(i) * x_len + get_val(j);
    idx_test[id] = get_val(i) * x_len + get_val(j);
    tsum += get_val(i) * 1.1 + get_val(j);
  });


  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(idx_test[i * x_len + j], i * x_len + j);
      total += i * 1.1 + j;
    }
  }
  ASSERT_FLOAT_EQ((double)total, (double)tsum.get());



#if defined(RAJA_ENABLE_CUDA)
  cudaErrchk(cudaFree(arr));
  cudaErrchk(cudaFree(idx_test));
#else
  delete[] arr;
  delete[] idx_test;
#endif
}

REGISTER_TYPED_TEST_SUITE_P(Kernel, Basic);

#ifdef RAJA_ENABLE_HIP
GPU_TYPED_TEST_P(Kernel_gpu, Basic)
{
  using Pol2d = at_v<TypeParam, 0>;
  using Pol3d = at_v<TypeParam, 1>;
  using Res = at_v<TypeParam, 2>;
  using IndexTypes = at_v<TypeParam, 3>;
  using Idx0 = at_v<IndexTypes, 0>;
  using Idx1 = at_v<IndexTypes, 1>;
  RAJA::ReduceSum<at_v<TypeParam, 4>, RAJA::Real_type> tsum(0.0);
  RAJA::ReduceMin<at_v<TypeParam, 4>, RAJA::Real_type> tMin(0.0);
  RAJA::ReduceMax<at_v<TypeParam, 4>, RAJA::Real_type> tMax(0.0);
  RAJA::Real_type total{0.0};
  auto ranges = RAJA::make_tuple(RAJA::TypedRangeSegment<Idx0>(0, x_len),
                                 RAJA::TypedRangeSegment<Idx1>(0, y_len));
  auto v = this->d_view;
  RAJA::kernel<Pol2d>(ranges, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
    // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
    v(get_val(i), j) = get_val(i) * x_len + j;
    tsum += get_val(i) * 1.1 + j;
  });

  hipDeviceSynchronize();

  hipErrchk(hipMemcpy(this->data, this->d_data, x_len*y_len*sizeof(double), hipMemcpyDeviceToHost));
  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      // std::cout << "i: " << get_val(i) << " j: " << j << "  " << this->view(i, j) << "  " << i * x_len + j <<  std::endl;
      ASSERT_EQ(this->view(i, j), i * x_len + j);
      total += i * 1.1 + j;
    }
  }
  ASSERT_FLOAT_EQ(total, tsum.get());

  camp::resources::Resource work_res{Res()};

  //Check reduction
  int stride1 = 5;
  int arr_len = stride1*stride1;

  double *arr = new double[arr_len];
  double *d_arr;
  hipErrchk(hipMalloc(&d_arr, arr_len*sizeof(double)));

  for(int i=0; i<arr_len; ++i){
    arr[i] = i;
  }

  //set the min and max of the array
  arr[4] = -1;
  arr[8] = 50;

  hipErrchk(hipMemcpy(d_arr, arr, arr_len*sizeof(double), hipMemcpyHostToDevice));

  tsum.reset(0.0);
  auto ranges2 = RAJA::make_tuple(RAJA::TypedRangeSegment<Idx0>(0, stride1),
                                  RAJA::TypedRangeSegment<Idx1>(0, stride1));

  RAJA::kernel<Pol2d>(ranges, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
      // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
      tsum += get_val(i) * 1.1 + get_val(j);
  });

  RAJA::kernel<Pol2d>(ranges2, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
      // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
      RAJA::Index_type id = get_val(j) + get_val(i) * stride1;
      tMin.min(d_arr[id]);
      tMax.max(d_arr[id]);
  });

  tMin.reset(0.0);
  tMax.reset(0.0);

  RAJA::kernel<Pol2d>(ranges2, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
      // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
      RAJA::Index_type id = get_val(j) + get_val(i) * stride1;
      tMin.min(d_arr[id]);

      tMax.max(d_arr[id]);
  });

  ASSERT_FLOAT_EQ(total, tsum.get());
  ASSERT_FLOAT_EQ(-1,  tMin.get());
  ASSERT_FLOAT_EQ(50, tMax.get());

  std::vector<Idx0> idx_x;
  std::vector<Idx1> idx_y;

  for(int i=0; i<x_len; ++i) idx_x.push_back(static_cast<Idx0>(i));
  for(int i=0; i<y_len; ++i) idx_y.push_back(static_cast<Idx1>(i));

  tsum.reset(0.0);
  total = 0.0;
  RAJA::TypedListSegment<Idx0> idx_list(&idx_x[0], idx_x.size(), work_res);
  RAJA::TypedListSegment<Idx1> idy_list(&idx_y[0], idx_y.size(), work_res);

  auto rangeList = RAJA::make_tuple(idx_list, idy_list);

  RAJA::kernel<Pol2d>(rangeList, [=] RAJA_HOST_DEVICE(Idx0 i, Idx1 j) {
    // std::cerr << "i: " << get_val(i) << " j: " << j << std::endl;
      v(get_val(i), j) = get_val(i) * x_len + j;
      tsum += get_val(i) * 1.1 + j;
  });
  hipErrchk(hipMemcpy(this->data, this->d_data, x_len*y_len*sizeof(double), hipMemcpyDeviceToHost));

  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(this->view(i, j), i * x_len + j);
      total += i * 1.1 + j;
    }
  }
  ASSERT_FLOAT_EQ(total, tsum.get());

  total=0.0;
  tsum.reset(0.0);
  double *idx_test = new double[x_len*y_len];
  double *d_idx_test;
  hipErrchk(hipMalloc(&d_idx_test, sizeof(double) * x_len * y_len));

  auto iterSpace2 = RAJA::make_tuple(RAJA::TypedRangeSegment<Idx0>(0,x_len), idy_list);
  RAJA::kernel<Pol2d>(iterSpace2, [=] RAJA_HOST_DEVICE (Idx0 i, Idx1 j) {
      Index_type id = get_val(i)*x_len + get_val(j);
      d_idx_test[id] = get_val(i) * x_len + get_val(j);
      tsum += get_val(i) * 1.1 + get_val(j);
  });
  hipErrchk(hipMemcpy(idx_test, d_idx_test, x_len*y_len*sizeof(double), hipMemcpyDeviceToHost));

  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(idx_test[i*x_len+j], i * x_len + j);
      total += i * 1.1 + j;
    }
  }
  ASSERT_FLOAT_EQ(total, tsum.get());


  total=0.0;
  tsum.reset(0.0);
  auto iterSpace3 = RAJA::make_tuple(RAJA::TypedRangeSegment<Idx0>(0,x_len), idy_list,RAJA::TypedRangeSegment<Idx1>(0,10));
  RAJA::kernel<Pol3d>(iterSpace3, [=] RAJA_HOST_DEVICE (Idx0 i, Idx1 j, Idx1 k) {
      Index_type id = get_val(i)*x_len + get_val(j);
      d_idx_test[id] = get_val(i) * x_len + get_val(j) + get_val(k) - get_val(k);
    tsum += get_val(i) * 1.1 + get_val(j);
  });
  hipErrchk(hipMemcpy(idx_test, d_idx_test, x_len*y_len*sizeof(double), hipMemcpyDeviceToHost));

  for (Index_type i = 0; i < x_len; ++i) {
    for (Index_type j = 0; j < y_len; ++j) {
      ASSERT_EQ(idx_test[i*x_len+j], i * x_len + j);
      total += i * 1.1 + j;
    }
  }

  ASSERT_FLOAT_EQ(total, tsum.get());

  hipErrchk(hipFree(d_arr));
  hipErrchk(hipFree(d_idx_test));
  delete[] arr;
  delete[] idx_test;
}
REGISTER_TYPED_TEST_SUITE_P(Kernel_gpu, Basic);
#endif

using RAJA::list;
using s = RAJA::seq_exec;
using TestTypes = ::testing::Types<
    list<KernelPolicy<For<1, s, statement::For<0, s, Lambda<0>>>>,
         camp::resources::Host,
         list<TypedIndex, Index_type>,
         RAJA::seq_reduce>,
    list<KernelPolicy<
             statement::Tile<1,
                             tile_fixed<2>,
                             RAJA::loop_exec,
                             statement::Tile<0,
                                             tile_fixed<2>,
                                             RAJA::loop_exec,
                                             For<0, s, For<1, s, Lambda<0>>>>>>,
         camp::resources::Host,
         list<Index_type, Index_type>,
         RAJA::seq_reduce>,
    list<KernelPolicy<statement::Collapse<s, ArgList<0, 1>, Lambda<0>>>,
         camp::resources::Host,
         list<Index_type, Index_type>,
         RAJA::seq_reduce>>;


INSTANTIATE_TYPED_TEST_SUITE_P(Sequential, Kernel, TestTypes);

#if defined(RAJA_ENABLE_OPENMP)
using OMPTypes = ::testing::Types<
    list<
        KernelPolicy<For<1, RAJA::omp_parallel_for_exec, For<0, s, Lambda<0>>>>,
        camp::resources::Host,
        list<TypedIndex, Index_type>,
        RAJA::omp_reduce>,
    list<KernelPolicy<
             statement::Tile<1,
                             tile_fixed<2>,
                             RAJA::omp_parallel_for_exec,
                             For<1, RAJA::loop_exec, For<0, s, Lambda<0>>>>>,
         camp::resources::Host,
         list<TypedIndex, Index_type>,
         RAJA::omp_reduce>>;
INSTANTIATE_TYPED_TEST_SUITE_P(OpenMP, Kernel, OMPTypes);
#endif
#if defined(RAJA_ENABLE_TBB)
using TBBTypes = ::testing::Types<
    list<KernelPolicy<For<1, RAJA::tbb_for_exec, For<0, s, Lambda<0>>>>,
         camp::resources::Host,
         list<TypedIndex, Index_type>,
         RAJA::tbb_reduce>>;
INSTANTIATE_TYPED_TEST_SUITE_P(TBB, Kernel, TBBTypes);
#endif
#if defined(RAJA_ENABLE_CUDA)
using CUDATypes = ::testing::Types<
    list<KernelPolicy<For<
             1,
             s,
             CudaKernel<For<0, RAJA::cuda_thread_x_loop, Lambda<0>>>>>,
         camp::resources::Cuda,
         list<TypedIndex, Index_type>,
         RAJA::cuda_reduce>>;
INSTANTIATE_TYPED_TEST_SUITE_P(CUDA, Kernel, CUDATypes);
#endif
#if defined(RAJA_ENABLE_HIP)
using HIPTypes = ::testing::Types<
    list<KernelPolicy<
           For<1, s,
             HipKernel<
              For<0, RAJA::hip_thread_x_loop, 
                Lambda<0>
              >
            >
          >
        >,
        KernelPolicy<
           For<1, s,
             HipKernel<               
                 For<0, RAJA::hip_thread_x_loop, 
                   Lambda<0, Segs<0, 1>, ValuesT<int, 0>>
                 >               
             >
           >
         >,
         camp::resources::Hip,
         list<TypedIndex, Index_type>,
         RAJA::hip_reduce>>;
INSTANTIATE_TYPED_TEST_SUITE_P(HIP, Kernel_gpu, HIPTypes);
#endif

#if defined(RAJA_ENABLE_CUDA)

GPU_TEST(Kernel, CudaZeroIter)
{
  using Pol =
      KernelPolicy<
        CudaKernel<
          For<0, cuda_thread_z_loop,
            For<1, cuda_thread_y_loop,
              For<2, cuda_thread_x_loop,
                Lambda<0>
              >
            >
          >
        >
      >;


  int *x = nullptr;
  cudaErrchk(cudaMallocManaged(&x, 3 * 2 * 5 * sizeof(int)));

  for (int i = 0; i < 3 * 2 * 5; ++i) {
    x[i] = 123;
  }

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
                       RAJA::RangeSegment(2, 2),  // should do 0 iterations
                       RAJA::RangeSegment(0, 5)),
      [=] __device__(Index_type i, Index_type j, Index_type k) {
        x[i + j * 3 + k * 3 * 2] = 321;
      });

  cudaErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < 3 * 2 * 5; ++i) {
    ASSERT_EQ(x[i], 123);
  }

  cudaErrchk(cudaFree(x));
}




GPU_TEST(Kernel, CudaReduceA)
{

  using Pol = RAJA::KernelPolicy<
      CudaKernel<
       For<0, cuda_block_x_loop,
         For<1, cuda_thread_z_loop,
           For<2, RAJA::seq_exec, Lambda<0>>>>>>;

  RAJA::ReduceSum<RAJA::cuda_reduce, int> reducer(0);

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, 3),
                                     RAJA::RangeSegment(0, 2),
                                     RAJA::RangeSegment(0, 5)),
                    [=] RAJA_DEVICE(Index_type i, Index_type j, Index_type k) {
                      reducer += 1;
                    });


  ASSERT_EQ((int)reducer, 3 * 2 * 5);
}


GPU_TEST(Kernel, CudaReduceB)
{

  using Pol = RAJA::KernelPolicy<
      For<2, RAJA::seq_exec,
        CudaKernel<
          For<0, cuda_block_z_loop,
            For<1, cuda_thread_y_loop,
              Lambda<0>
            > > > > >;

  RAJA::ReduceSum<RAJA::cuda_reduce, int> reducer(0);

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, 3),
                                     RAJA::RangeSegment(0, 2),
                                     RAJA::RangeSegment(0, 5)),
                    [=] RAJA_DEVICE(Index_type i, Index_type j, Index_type k) {
                      reducer += 1;
                    });


  ASSERT_EQ((int)reducer, 3 * 2 * 5);
}



GPU_TEST(Kernel, SubRange_ThreadBlock)
{
  using Pol = RAJA::KernelPolicy<
      CudaKernel<For<0, RAJA::cuda_thread_x_loop, Lambda<0>>>>;

  size_t num_elem = 2048;
  size_t first = 10;
  size_t last = num_elem - 10;

  double *ptr = nullptr;
  cudaErrchk(cudaMallocManaged(&ptr, sizeof(double) * num_elem));

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, num_elem)),
                    [=] RAJA_HOST_DEVICE(Index_type i) { ptr[i] = 0.0; });

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(first, last)),
                    [=] RAJA_HOST_DEVICE(Index_type i) { ptr[i] = 1.0; });
  cudaErrchk(cudaDeviceSynchronize());

  size_t count = 0;
  for (size_t i = 0; i < num_elem; ++i) {
    count += ptr[i];
  }
  ASSERT_EQ(count, num_elem - 20);
  for (size_t i = 0; i < 10; ++i) {
    ASSERT_EQ(ptr[i], 0.0);
    ASSERT_EQ(ptr[num_elem - 1 - i], 0.0);
  }
  cudaErrchk(cudaFree(ptr));
}


#endif

#if defined(RAJA_ENABLE_HIP)

GPU_TEST(Kernel_gpu, HipZeroIter)
{
  using Pol =
      KernelPolicy<
        HipKernel<
          For<0, hip_thread_z_loop,
            For<1, hip_thread_y_loop,
              For<2, hip_thread_x_loop,
                Lambda<0>
              >
            >
          >
        >
      >;


  int *x = new int[3 * 2 * 5];
  int *d_x = nullptr;
  hipErrchk(hipMalloc(&d_x, 3 * 2 * 5 * sizeof(int)));

  for (int i = 0; i < 3 * 2 * 5; ++i) {
    x[i] = 123;
  }

  hipErrchk(hipMemcpy(d_x, x, 3 * 2 * 5 * sizeof(int), hipMemcpyHostToDevice));

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, 3),
                       RAJA::RangeSegment(2, 2),  // should do 0 iterations
                       RAJA::RangeSegment(0, 5)),
      [=] __device__(Index_type i, Index_type j, Index_type k) {
        d_x[i + j * 3 + k * 3 * 2] = 321;
      });

  hipErrchk(hipMemcpy(x, d_x, 3 * 2 * 5 *sizeof(int), hipMemcpyDeviceToHost));

  for (int i = 0; i < 3 * 2 * 5; ++i) {
    ASSERT_EQ(x[i], 123);
  }

  hipErrchk(hipFree(d_x));
  delete[] x;
}




GPU_TEST(Kernel_gpu, HipReduceA)
{

  using Pol = RAJA::KernelPolicy<
      HipKernel<
       For<0, hip_block_x_loop,
         For<1, hip_thread_z_loop,
           For<2, RAJA::seq_exec, Lambda<0>>>>>>;

  RAJA::ReduceSum<RAJA::hip_reduce, int> reducer(0);

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, 3),
                                     RAJA::RangeSegment(0, 2),
                                     RAJA::RangeSegment(0, 5)),
                    [=] RAJA_DEVICE(Index_type RAJA_UNUSED_ARG(i), 
                                    Index_type RAJA_UNUSED_ARG(j), 
                                    Index_type RAJA_UNUSED_ARG(k)) {
                      reducer += 1;
                    });

  ASSERT_EQ((int)reducer, 3 * 2 * 5);
}


GPU_TEST(Kernel_gpu, HipReduceB)
{

  using Pol = RAJA::KernelPolicy<
      For<2, RAJA::seq_exec,
        HipKernel<
          For<0, hip_block_z_loop,
            For<1, hip_thread_y_loop,
              Lambda<0>
            > > > > >;

  RAJA::ReduceSum<RAJA::hip_reduce, int> reducer(0);

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, 3),
                                     RAJA::RangeSegment(0, 2),
                                     RAJA::RangeSegment(0, 5)),
                    [=] RAJA_DEVICE(Index_type RAJA_UNUSED_ARG(i), 
                                    Index_type RAJA_UNUSED_ARG(j), 
                                    Index_type RAJA_UNUSED_ARG(k)) {
                      reducer += 1;
                    });

  ASSERT_EQ((int)reducer, 3 * 2 * 5);
}



GPU_TEST(Kernel_gpu, SubRange_ThreadBlock)
{
  using Pol = RAJA::KernelPolicy<
      HipKernel<For<0, RAJA::hip_thread_x_loop, Lambda<0>>>>;

  size_t num_elem = 2048;
  size_t first = 10;
  size_t last = num_elem - 10;

  double *ptr = (double*) malloc(sizeof(double) * num_elem);
  double *d_ptr = nullptr;
  hipErrchk(hipMalloc(&d_ptr, sizeof(double) * num_elem) );

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, num_elem)),
                    [=] RAJA_HOST_DEVICE(Index_type i) { d_ptr[i] = 0.0; });

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(first, last)),
                    [=] RAJA_HOST_DEVICE(Index_type i) { d_ptr[i] = 1.0; });

  hipErrchk(hipMemcpy(ptr, d_ptr, sizeof(double) * num_elem, hipMemcpyDeviceToHost));

  size_t count = 0;
  for (size_t i = 0; i < num_elem; ++i) {
    count += ptr[i];
  }
  ASSERT_EQ(count, num_elem - 20);
  for (size_t i = 0; i < 10; ++i) {
    ASSERT_EQ(ptr[i], 0.0);
    ASSERT_EQ(ptr[num_elem - 1 - i], 0.0);
  }

  free(ptr);
  hipErrchk(hipFree(d_ptr));
}


#endif


TEST(Kernel, FissionFusion)
{
  using namespace RAJA;

  // Loop Fusion
  using Pol_Fusion = KernelPolicy<For<0, seq_exec, Lambda<0, Segs<0>>, Lambda<1, Segs<0>>>>;

  // Loop Fission
  using Pol_Fission =
      KernelPolicy<For<0, seq_exec, Lambda<0, Segs<0>>>, For<0, seq_exec, Lambda<1, Segs<0>>>>;


  constexpr int N = 16;
  int *x = new int[N];
  int *y = new int[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0;
    y[i] = 0;
  }

  kernel<Pol_Fission>(

      RAJA::make_tuple(RangeSegment(0, N), RangeSegment(0, N)),

      [=](int i) { x[i] += 1; },

      [=](int i) { x[i] += 2; });


  kernel<Pol_Fusion>(

      RAJA::make_tuple(RangeSegment(0, N), RangeSegment(0, N)),

      [=](int i) { y[i] += 1; },

      [=](int i) { y[i] += 2; });

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], y[i]);
  }

  delete[] x;
  delete[] y;
}

TEST(Kernel, FissionFusion_Conditional)
{
  using namespace RAJA;


  // Loop Fusion if param == 0
  // Loop Fission if param == 1

  using Pol = KernelPolicy<
      If<Equals<Param<0>, Value<0>>, For<0, seq_exec, Lambda<0, Segs<0>>, Lambda<1, Segs<0>>>>,
      If<Equals<Param<0>, Value<1>>,
         For<0, seq_exec, Lambda<0, Segs<0>>>,
         For<0, seq_exec, Lambda<1, Segs<0>>>>>;


  constexpr int N = 16;
  int *x = new int[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0;
  }


  for (int param = 0; param < 2; ++param) {


    kernel_param<Pol>(

        RAJA::make_tuple(RangeSegment(0, N), RangeSegment(0, N)),

        RAJA::make_tuple(param),

        [=](int i) { x[i] += 1; },

        [=](int i) { x[i] += 2; });

    for (int i = 0; i < N; ++i) {
      ASSERT_EQ(x[i], 3 + 3 * param);
    }
  }


  delete[] x;
}

TEST(Kernel, ForICount)
{
  using namespace RAJA;

  constexpr int N = 17;

  // Loop Fusion
  using Pol = KernelPolicy<
      statement::ForICount<0, Param<0>, seq_exec,
                           Lambda<0>>>;


  int *x = new int[N];
  int *xi = new int[N];

  for (int i = 0; i < N; ++i) {
    x[i] = 0;
    xi[i] = 0;
  }

  kernel_param<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),
      RAJA::make_tuple((RAJA::Index_type)0),

      [=](RAJA::Index_type i, RAJA::Index_type ii) {
        x[i] += 1;
        xi[ii] += 1;
      });

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], 1);
    ASSERT_EQ(xi[i], 1);
  }

  delete[] xi;
  delete[] x;
}

TEST(Kernel, ForICountTyped_seq)
{
  using namespace RAJA;

  constexpr int N = 17;

  // Loop Fusion
  using Pol = KernelPolicy<
      statement::ForICount<0, Param<0>, seq_exec,
                           Lambda<0>>>;


  int *x = new int[N];
  int *xi = new int[N];

  for (int i = 0; i < N; ++i) {
    x[i] = 0;
    xi[i] = 0;
  }

  kernel_param<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),
      RAJA::make_tuple(ZoneI(0)),

      [=](RAJA::Index_type i, ZoneI ii) {
        x[i] += 1;
        xi[*ii] += 1;
      });

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], 1);
    ASSERT_EQ(xi[i], 1);
  }

  delete[] xi;
  delete[] x;
}

// simd bug when using intel version pre 19.0
#if defined(__INTEL_COMPILER) and (__INTEL_COMPILER_BUILD_DATE < 20180804) 
#else
TEST(Kernel, ForICountTyped_simd)
{
  using namespace RAJA;

  constexpr int N = 17;

  // Loop Fusion
  using Pol = KernelPolicy<
      statement::ForICount<0, Param<0>, simd_exec,
                           Lambda<0>>>;


  int *x = new int[N];
  int *xi = new int[N];

  for (int i = 0; i < N; ++i) {
    x[i] = 0;
    xi[i] = 0;
  }

  kernel_param<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),
      RAJA::make_tuple(ZoneI(0)),

      [=](RAJA::Index_type i, ZoneI ii) {
        x[i] += 1;
        xi[*ii] += 1;
      });

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], 1);
    ASSERT_EQ(xi[i], 1);
  }

  delete[] xi;
  delete[] x;
}
#endif



TEST(Kernel, Tile)
{
  using namespace RAJA;


  // Loop Fusion
  using Pol = KernelPolicy<
      statement::Tile<1,
                      tile_fixed<4>,
                      seq_exec,
                      For<0, seq_exec, For<1, seq_exec, Lambda<0, Segs<0>>>>,
                      For<0, seq_exec, For<1, seq_exec, Lambda<0, Segs<0>>>>>,
      For<1, seq_exec, Lambda<1, Segs<1>>>>;


  constexpr int N = 16;
  int *x = new int[N];
  for (int i = 0; i < N; ++i) {
    x[i] = 0;
  }

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N), RangeSegment(0, N)),

      [=](RAJA::Index_type i) { x[i] += 1; },
      [=](RAJA::Index_type j) { x[j] *= 10; });

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], 320);
  }

  delete[] x;
}

TEST(Kernel, TileTCount)
{
  using namespace RAJA;

  constexpr int N = 17;
  constexpr int T = 4;
  constexpr int NT = (N+T-1)/T;

  // Loop Fusion
  using Pol = KernelPolicy<
      statement::TileTCount<0, Param<0>,
                      tile_fixed<T>, seq_exec,
                      For<0, seq_exec, Lambda<0>>>>;


  int *x = new int[N];
  int *xt = new int[NT];

  for (int i = 0; i < N; ++i) {
    x[i] = 0;
  }
  for (int t = 0; t < NT; ++t) {
    xt[t] = 0;
  }

  kernel_param<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),
      RAJA::make_tuple((RAJA::Index_type)0),

      [=](RAJA::Index_type i, RAJA::Index_type it) {
        x[i] += 1;
        xt[it] += 1;
      });

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], 1);
  }
  for (int t = 0; t < NT; ++t) {
    int expect = T;
    if ((t+1)*T > N) {
      expect = N - t*T;
    }
    ASSERT_EQ(xt[t], expect);
  }

  delete[] xt;
  delete[] x;
}


TEST(Kernel, TileTCountTyped)
{
  using namespace RAJA;

  constexpr int N = 17;
  constexpr int T = 4;
  constexpr int NT = (N+T-1)/T;

  // Loop Fusion
  using Pol = KernelPolicy<
      statement::TileTCount<0, Param<0>,
                      tile_fixed<T>, seq_exec,
                      For<0, seq_exec, Lambda<0>>>>;


  int *x = new int[N];
  int *xt = new int[NT];

  for (int i = 0; i < N; ++i) {
    x[i] = 0;
  }
  for (int t = 0; t < NT; ++t) {
    xt[t] = 0;
  }

  kernel_param<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),
      RAJA::make_tuple(ZoneI(0)),

      [=](RAJA::Index_type i, ZoneI it) {
        x[i] += 1;
        xt[*it] += 1;
      });

  for (int i = 0; i < N; ++i) {
    ASSERT_EQ(x[i], 1);
  }
  for (int t = 0; t < NT; ++t) {
    int expect = T;
    if ((t+1)*T > N) {
      expect = N - t*T;
    }
    ASSERT_EQ(xt[t], expect);
  }

  delete[] xt;
  delete[] x;
}


TEST(Kernel, CollapseSeq)
{
  using namespace RAJA;


  // Loop Fusion
  using Pol = KernelPolicy<Collapse<seq_exec, ArgList<0, 1>, Lambda<0>>>;


  constexpr int N = 16;
  int *x = new int[N * N];
  for (int i = 0; i < N * N; ++i) {
    x[i] = 0;
  }

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N), RangeSegment(0, N)),

      [=](RAJA::Index_type i, RAJA::Index_type j) { x[i * N + j] += 1; });

  for (int i = 0; i < N * N; ++i) {
    ASSERT_EQ(x[i], 1);
  }

  delete[] x;
}

#if defined(RAJA_ENABLE_OPENMP)
TEST(Kernel, Collapse2)
{
  int N = 16;
  int M = 7;


  int *data = new int[N * M];
  for (int i = 0; i < M * N; ++i) {
    data[i] = -1;
  }

  using Pol = RAJA::KernelPolicy<
      RAJA::statement::
          Collapse<RAJA::omp_parallel_collapse_exec, ArgList<0, 1>, Lambda<0>>>;

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, N),
                                     RAJA::RangeSegment(0, M)),

                    [=](Index_type i, Index_type j) { data[i + j * N] = i; });

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      ASSERT_EQ(data[i + j * N], i);
    }
  }


  delete[] data;
}


TEST(Kernel, Collapse3)
{
  int N = 1;
  int M = 2;
  int K = 3;

  int *data = new int[N * M * K];
  for (int i = 0; i < M * N * K; ++i) {
    data[i] = -1;
  }

  using Pol = RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                ArgList<0, 1, 2>,
                                Lambda<0>>>;

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, K),
                                     RAJA::RangeSegment(0, M),
                                     RAJA::RangeSegment(0, N)),
                    [=](Index_type k, Index_type j, Index_type i) {
                      data[i + N * (j + M * k)] = i + N * (j + M * k);
                    });


  for (int k = 0; k < K; k++) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {

        int id = i + N * (j + M * k);
        ASSERT_EQ(data[id], id);
      }
    }
  }

  delete[] data;
}

TEST(Kernel, Collapse4)
{
  int N = 1;
  int M = 2;
  int K = 3;

  int *data = new int[N * M * K];
  for (int i = 0; i < M * N * K; ++i) {
    data[i] = -1;
  }

  using Pol = RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                ArgList<0, 1, 2>,
                                Lambda<0>>>;

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, K),
                                     RAJA::RangeSegment(0, M),
                                     RAJA::RangeSegment(0, N)),
                    [=](Index_type k, Index_type j, Index_type i) {
                      Index_type id = i + N * (j + M * k);
                      data[id] = id;
                    });

  for (int k = 0; k < K; k++) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {

        int id = i + N * (j + M * k);
        ASSERT_EQ(data[id], id);
      }
    }
  }

  delete[] data;
}


TEST(Kernel, Collapse5)
{

  int N = 4;
  int M = 4;
  int K = 4;

  int *data = new int[N * M * K];
  for (int i = 0; i < M * N * K; ++i) {
    data[i] = -1;
  }


  using Pol = RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                ArgList<0, 1>,
                                For<2, RAJA::seq_exec, Lambda<0>>>>;

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, K),
                                     RAJA::RangeSegment(0, M),
                                     RAJA::RangeSegment(0, N)),
                    [=](Index_type k, Index_type j, Index_type i) {
                      data[i + N * (j + M * k)] = i + N * (j + M * k);
                    });

  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {

        int id = i + N * (j + M * k);
        ASSERT_EQ(data[id], id);
      }
    }
  }

  delete[] data;
}


TEST(Kernel, Collapse6)
{

  int N = 3;
  int M = 3;
  int K = 4;

  int *data = new int[N * M];
  for (int i = 0; i < M * N; ++i) {
    data[i] = 0;
  }


  using Pol = RAJA::KernelPolicy<
      For<0,
          RAJA::seq_exec,
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    ArgList<1, 2>,
                                    Lambda<0>>>>;


  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, K),
                                     RAJA::RangeSegment(0, M),
                                     RAJA::RangeSegment(0, N)),
                    [=](Index_type k, Index_type j, Index_type i) {
                      data[i + N * j] += k;
                    });

  for (int j = 0; j < M; ++j) {
    for (int i = 0; i < N; ++i) {
      ASSERT_EQ(data[i + N * j], 6);
    }
  }


  delete[] data;
}

TEST(Kernel, Collapse7)
{

  int N = 3;
  int M = 3;
  int K = 4;
  int P = 8;

  int *data = new int[N * M * K * P];
  for (int i = 0; i < N * M * K * P; ++i) {
    data[i] = 0;
  }

  using Pol = RAJA::KernelPolicy<
      For<0,
          RAJA::seq_exec,
          RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                    ArgList<1, 2, 3>,
                                    Lambda<0>>>>;

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, K),
                       RAJA::RangeSegment(0, M),
                       RAJA::RangeSegment(0, N),
                       RAJA::RangeSegment(0, P)),
      [=](Index_type k, Index_type j, Index_type i, Index_type r) {
        Index_type id = r + P * (i + N * (j + M * k));
        data[id] += id;
      });

  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {
        for (int r = 0; r < P; ++r) {
          Index_type id = r + P * (i + N * (j + M * k));
          ASSERT_EQ(data[id], id);
        }
      }
    }
  }

  delete[] data;
}


TEST(Kernel, Collapse8)
{

  int N = 3;
  int M = 3;
  int K = 4;
  int P = 8;

  int *data = new int[N * M * K * P];
  for (int i = 0; i < N * M * K * P; ++i) {
    data[i] = 0;
  }

  using Pol = RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                ArgList<0, 1, 2>,
                                For<3, RAJA::seq_exec, Lambda<0>>>>;

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, K),
                       RAJA::RangeSegment(0, M),
                       RAJA::RangeSegment(0, N),
                       RAJA::RangeSegment(0, P)),
      [=](Index_type k, Index_type j, Index_type i, Index_type r) {
        Index_type id = r + P * (i + N * (j + M * k));
        data[id] += id;
      });

  for (int k = 0; k < K; ++k) {
    for (int j = 0; j < M; ++j) {
      for (int i = 0; i < N; ++i) {
        for (int r = 0; r < P; ++r) {
          Index_type id = r + P * (i + N * (j + M * k));
          ASSERT_EQ(data[id], id);
        }
      }
    }
  }

  delete[] data;
}

#endif  // RAJA_ENABLE_OPENMP




TEST(Kernel, ReduceSeqSum)
{

  int N = 1023;

  int *data = new int[N];
  for (int i = 0; i < N; ++i) {
    data[i] = i;
  }

  using Pol = RAJA::KernelPolicy<
      RAJA::statement::For<0, seq_exec,
        Lambda<0>,
        RAJA::statement::Reduce<seq_reduce, RAJA::operators::plus, Param<0>,
          Lambda<1, Params<0>>
        >
      >
     >;

  int sum = 0;
  int *sumPtr = &sum;

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      RAJA::make_tuple((int)0),

      [=](Index_type i, int &value) {
        value = data[i];
      },
      [=](int &value) {
        (*sumPtr) += value;
      });

  ASSERT_EQ(sum, N*(N-1)/2);

  delete[] data;
}



#if defined(RAJA_ENABLE_CUDA)


GPU_TEST(Kernel, ReduceCudaSum1)
{

  long N = 2345;

  using Pol =
      KernelPolicy<CudaKernel<
        For<0, cuda_thread_x_loop, Lambda<0>>,
        RAJA::statement::Reduce<cuda_block_reduce, RAJA::operators::plus, Param<0>,
          Lambda<1, Params<0>>
        >
      >>;

  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      RAJA::make_tuple((long)0),

      [=] __device__ (Index_type i, long &value) {
        value += i;
      },
      [=] __device__ (long &value) {
        // This only gets executed on the "root" thread which reecieved the
        // reduced value
        trip_count += value;
      });


  ASSERT_EQ(trip_count.get(), N*(N-1)/2);

}


GPU_TEST(Kernel, CudaWarpLoop1)
{

  long N = 2345;

  using Pol =
      KernelPolicy<CudaKernel<
        For<0, cuda_warp_loop, Lambda<0>>
      >>;

  RAJA::ReduceSum<cuda_reduce, long> value(0);

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      [=] __device__ (Index_type i) {
        value += i;
      });


  ASSERT_EQ(value.get(), N*(N-1)/2);

}

GPU_TEST(Kernel, CudaWarpLoop2)
{

  long N = 2345;

  using Pol =
      KernelPolicy<CudaKernel<
        Tile<0, tile_fixed<32>, seq_exec,
          For<0, cuda_warp_direct, Lambda<0>>
        >
      >>;

  RAJA::ReduceSum<cuda_reduce, long> value(0);

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      [=] __device__ (Index_type i) {
        value += i;
      });


  ASSERT_EQ(value.get(), N*(N-1)/2);

}

GPU_TEST(Kernel, CudaWarpLoop3)
{

  long A = 32;
  long B = 43;
  long N = A*B;

  using Pol =
      KernelPolicy<CudaKernel<
        Tile<0, tile_fixed<32>, seq_exec,
          ForICount<0, Param<0>, cuda_warp_direct, Lambda<0>>
        >
      >>;

  RAJA::ReduceSum<cuda_reduce, long> value(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      RAJA::make_tuple((Index_type)0),

      [=] __device__ (Index_type i, Index_type j) {
        value += j;  // j should only be 0..31
      });


  ASSERT_EQ(value.get(), B*A*(A-1)/2);

}

GPU_TEST(Kernel, CudaWarpLoop4)
{

  long A = 16;
  long B = 43;

  using maskA = BitMask<4,0>;
  using maskB = BitMask<1,4>;

  using Pol =
      KernelPolicy<CudaKernel<
        For<0, cuda_warp_masked_direct<maskA>,
          For<1, cuda_warp_masked_loop<maskB>,
            Lambda<0>
          >
        >
      >>;

  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);
  RAJA::ReduceSum<cuda_reduce, long> value(0);

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, A), RAJA::RangeSegment(0, B)),

      [=] __device__ (Index_type i, Index_type j) {
        trip_count += 1;
        value += i;  // i should only be 0..A-1
      });

  ASSERT_EQ(trip_count.get(), A*B);
  ASSERT_EQ(value.get(), B*A*(A-1)/2);

}

GPU_TEST(Kernel, CudaThreadMasked1)
{

  long A = 64;
  long B = 943;

  using maskA = BitMask<6,0>;
  using maskB = BitMask<2,6>;

  using Pol =
      KernelPolicy<CudaKernel<
        For<0, cuda_thread_masked_direct<maskA>,
          For<1, cuda_thread_masked_loop<maskB>,
            Lambda<0>
          >
        >
      >>;

  RAJA::ReduceMax<cuda_reduce, int> max_thread(0);
  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);
  RAJA::ReduceSum<cuda_reduce, long> value(0);

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, A), RAJA::RangeSegment(0, B)),

      [=] __device__ (Index_type i, Index_type j) {
        trip_count += 1;
        value += i;  // i should only be 0..A-1
        max_thread.max(threadIdx.x);
      });

  ASSERT_EQ(max_thread.get(), 255);
  ASSERT_EQ(trip_count.get(), A*B);
  ASSERT_EQ(value.get(), B*A*(A-1)/2);

}

GPU_TEST(Kernel, CudaThreadMasked2)
{

  long A = 64;
  long B = 4*123;

  using maskA = BitMask<6,0>;
  using maskB = BitMask<2,6>;

  using Pol =
      KernelPolicy<CudaKernel<
        ForICount<0, Param<0>, cuda_thread_masked_direct<maskA>,
        ForICount<1, Param<1>, cuda_thread_masked_loop<maskB>,
            Lambda<0>
          >
        >
      >>;

  RAJA::ReduceMax<cuda_reduce, int> max_thread(0);
  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);
  RAJA::ReduceSum<cuda_reduce, long> value(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, A), RAJA::RangeSegment(0, B)),

      RAJA::make_tuple((Index_type)0, (Index_type)0),

      [=] __device__ (Index_type i, Index_type j, Index_type x, Index_type y) {
        trip_count += 1;
        value += y;  // i should only be 0..3
        max_thread.max(threadIdx.x);
      });

  ASSERT_EQ(max_thread.get(), 255);
  ASSERT_EQ(trip_count.get(), A*B);
  ASSERT_EQ(value.get(), A*B*(B-1)/2);

}


GPU_TEST(Kernel, ReduceCudaWarpLoop1)
{

  long N = 2345;

  using Pol =
      KernelPolicy<CudaKernel<
        For<0, cuda_warp_loop, Lambda<0>>,
        RAJA::statement::Reduce<cuda_warp_reduce, RAJA::operators::plus, Param<0>,
          Lambda<1, Params<0>>
        >
      >>;

  RAJA::ReduceSum<cuda_reduce, long> total_count(0);
  RAJA::ReduceSum<cuda_reduce, long> reduce_count(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      RAJA::make_tuple((long)0),

      [=] __device__ (Index_type i, long &value) {
        value += i;
      },
      [=] __device__ (long &value) {
        // This only gets executed on the "root" thread which recieved the
        // reduced value
        total_count += value;
        reduce_count += 1;
      });


  ASSERT_EQ(total_count.get(), N*(N-1)/2);
  ASSERT_EQ(reduce_count.get(), 1);

}


GPU_TEST(Kernel, ReduceCudaWarpLoop2)
{

  long N = 239;
  long M = 17;

  using Pol =
      KernelPolicy<CudaKernel<
        For<1, cuda_thread_y_loop,
          For<0, cuda_warp_loop, Lambda<0>>
        >,
        RAJA::statement::Reduce<cuda_warp_reduce, RAJA::operators::plus, Param<0>,
          Lambda<1, Params<0>>
        >
      >>;

  RAJA::ReduceSum<cuda_reduce, long> total_count(0);
  RAJA::ReduceSum<cuda_reduce, long> reduce_count(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N),
                       RAJA::RangeSegment(0, M)),

      RAJA::make_tuple((long)0),

      [=] __device__ (Index_type i, Index_type j, long &value) {
        value += i + j*N;
      },
      [=] __device__ (long &value) {
        // This only gets executed on the "root" thread which recieved the
        // reduced value
        total_count += value;
        reduce_count += 1;
      });


  long NM = N*M;
  ASSERT_EQ(total_count.get(), NM*(NM-1)/2);
  ASSERT_EQ(reduce_count.get(), M);

}


GPU_TEST(Kernel, ReduceCudaWarpLoop3)
{

  long N = 25;
  long M = 16;
  long O = 10;

  using Pol =
      KernelPolicy<CudaKernel<
        For<2, cuda_block_x_loop,
          For<1, cuda_thread_y_direct,
            For<0, cuda_warp_direct, Lambda<0>>
          >,
          RAJA::statement::CudaSyncWarp,
          RAJA::statement::Reduce<cuda_warp_reduce, RAJA::operators::plus, Param<0>,
            Lambda<1, Params<0>>
          >
        >
      >>;

  RAJA::ReduceSum<cuda_reduce, long> total_count(0);
  RAJA::ReduceSum<cuda_reduce, long> reduce_count(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N),
                       RAJA::RangeSegment(0, M),
                       RAJA::RangeSegment(0, O)),

      RAJA::make_tuple((long)0),
      [=] __device__ (Index_type i, Index_type j, Index_type k, long &value) {
        value = i + j*N + k*N*M;
      },
      [=] __device__ (long &value) {
        // This only gets executed on the "root" thread which reecieved the
        // reduced value
        total_count += value;
        reduce_count += 1;
      });


  long NMO = N*M*O;
  ASSERT_EQ(total_count.get(), NMO*(NMO-1)/2);
  ASSERT_EQ(reduce_count.get(), M*O);

}


GPU_TEST(Kernel, CudaForICount)
{
  using namespace RAJA;


  constexpr long N = 1035;
  constexpr long T = 32;

  // Loop Fusion
  using Pol =
      KernelPolicy<CudaKernel<
       statement::Tile<0, tile_fixed<T>, cuda_block_x_loop,
         ForICount<0, Param<0>, cuda_thread_x_direct, Lambda<0>>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  for (long t = 0; t < T; ++t) {
    RAJA::ReduceSum<cuda_reduce, long> tile_count(0);

    kernel_param<Pol>(

        RAJA::make_tuple(RangeSegment(0, N)),
        RAJA::make_tuple((RAJA::Index_type)0),

        [=] __device__(RAJA::Index_type i, RAJA::Index_type ii) {
          trip_count += 1;
          if (i%T == t && ii == t) {
            tile_count += 1;
          }
        });
    cudaErrchk(cudaDeviceSynchronize());

    long trip_result = (long)trip_count;
    long tile_result = (long)tile_count;

    ASSERT_EQ(trip_result, (t+1)*N);

    long tile_expect = N/T;
    if (t < N%T) {
      tile_expect += 1;
    }
    ASSERT_EQ(tile_result, tile_expect);
  }
}


GPU_TEST(Kernel, CudaTileTCount)
{
  using namespace RAJA;


  constexpr long N = 1035;
  constexpr long T = 32;
  constexpr long NT = (N+T-1)/T;

  // Loop Fusion
  using Pol =
      KernelPolicy<CudaKernel<
       statement::TileTCount<0, Param<0>, tile_fixed<T>, cuda_block_x_loop,
         For<0, cuda_thread_x_direct, Lambda<0>>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  for (long t = 0; t < NT; ++t) {
    RAJA::ReduceSum<cuda_reduce, long> tile_count(0);

    kernel_param<Pol>(

        RAJA::make_tuple(RangeSegment(0, N)),
        RAJA::make_tuple((RAJA::Index_type)0),

        [=] __device__(RAJA::Index_type i, RAJA::Index_type ti) {
          trip_count += 1;
          if (i/T == t && ti == t) {
            tile_count += 1;
          }
        });
    cudaErrchk(cudaDeviceSynchronize());

    long trip_result = (long)trip_count;
    long tile_result = (long)tile_count;

    ASSERT_EQ(trip_result, (t+1)*N);

    long tile_expect = T;
    if ((t+1)*T > N) {
      tile_expect = N - t*T;
    }
    ASSERT_EQ(tile_result, tile_expect);
  }
}


GPU_TEST(Kernel, CudaConditional)
{
  using namespace RAJA;


  constexpr long N = (long)1024;

  // Loop Fusion
  using Pol = KernelPolicy<CudaKernel<
      statement::Tile<0, tile_fixed<32>, cuda_block_x_loop,
        For<0, cuda_thread_x_loop, If<Param<0>, Lambda<0>>, Lambda<1>>>>>;

  for (int param = 0; param < 2; ++param) {

    RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

    kernel_param<Pol>(

        RAJA::make_tuple(TypedRangeSegment<int>(0, N)),

        RAJA::make_tuple((bool)param),

        // This only gets executed if param==1
        [=] RAJA_DEVICE(int i, bool) { trip_count += 2; },

        // This always gets executed
        [=] RAJA_DEVICE(int i, bool) { trip_count += 1; });
    cudaErrchk(cudaDeviceSynchronize());

    long result = (long)trip_count;

    ASSERT_EQ(result, N * (2 * param + 1));
  }
}


GPU_TEST(Kernel, CudaExec1)
{
  using namespace RAJA;


  constexpr long N = (long)3 * 1024 * 1024;

  // Loop Fusion
  using Pol =
      KernelPolicy<CudaKernel<For<0, cuda_thread_x_loop, Lambda<0>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),

      [=] __device__(ptrdiff_t i) { trip_count += 1; });
  cudaErrchk(cudaDeviceSynchronize());

  long result = (long)trip_count;

  ASSERT_EQ(result, N);
}


GPU_TEST(Kernel, CudaExec1ab)
{
  using namespace RAJA;


  constexpr long N = (long)128;

  // Loop Fusion
  using Pol = KernelPolicy<CudaKernel<
      For<0, cuda_block_x_loop,
        For<1, cuda_block_y_loop,
          For<2, cuda_block_z_loop,
            Lambda<0>
   >>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N),
                       RangeSegment(0, N),
                       RangeStrideSegment(0, N, 2)),

      [=] __device__(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) {
        trip_count += 1;
      });
  cudaErrchk(cudaDeviceSynchronize());

  long result = (long)trip_count;

  ASSERT_EQ(result, N * N * N / 2);
}


GPU_TEST(Kernel, CudaExec1b)
{
  using namespace RAJA;


  constexpr long N = (long)3 * 1024 * 1024;

  // Loop Fusion
  using Pol =
      KernelPolicy<CudaKernel<
        statement::Tile<0, tile_fixed<128>, cuda_block_z_loop,
          For<0, cuda_thread_y_loop, Lambda<0>>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),

      [=] __device__(ptrdiff_t i) { trip_count += 1; });
  cudaErrchk(cudaDeviceSynchronize());

  long result = (long)trip_count;

  ASSERT_EQ(result, N);
}



GPU_TEST(Kernel, CudaExec1c)
{
  using namespace RAJA;


  constexpr long N = (long)3;

  // Loop Fusion
  using Pol = KernelPolicy<
      CudaKernelExt<cuda_explicit_launch<false, 5, 3, 1>,
           statement::Tile<2, tile_fixed<2>, cuda_block_z_loop,
                    For<0, cuda_block_x_loop,
                        For<1, cuda_block_y_loop,
                            For<2, cuda_thread_z_loop, Lambda<0>>>>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N),
                       RangeSegment(0, N),
                       RangeSegment(0, N)),

      [=] __device__(RAJA::Index_type i,
                     RAJA::Index_type j,
                     RAJA::Index_type k) { trip_count += 1; });
  cudaErrchk(cudaDeviceSynchronize());

  long result = (long)trip_count;

  ASSERT_EQ(result, N * N * N);
}


GPU_TEST(Kernel, CudaComplexNested)
{
  using namespace RAJA;


  constexpr long N = (long)739;

  using Pol = KernelPolicy<CudaKernel<
      For<0, cuda_block_x_loop,
          For<1, cuda_thread_x_loop, For<2, cuda_thread_y_loop, Lambda<0>>>,
          For<2, cuda_thread_x_loop, Lambda<0, Segs<0>, ValuesT<Index_type, 0>, Segs<2>>>>>>;

  int *ptr = nullptr;
  cudaErrchk(cudaMallocManaged(&ptr, sizeof(int) * N));

  for (long i = 0; i < N; ++i) {
    ptr[i] = 0;
  }


  auto segments = RAJA::make_tuple(RangeSegment(0, N),
                                   RangeSegment(0, N),
                                   RangeSegment(0, N));


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      segments,

      [=] __device__(RAJA::Index_type i,
                     RAJA::Index_type j,
                     RAJA::Index_type k) {
        trip_count += 1;
        RAJA::atomicAdd<RAJA::auto_atomic>(ptr + i, (int)1);
      });
  cudaErrchk(cudaDeviceSynchronize());

  for (long i = 0; i < N; ++i) {
    ASSERT_EQ(ptr[i], (int)(N * N + N));
  }

  // check trip count
  long result = (long)trip_count;
  ASSERT_EQ(result, N * N * N + N * N);


  cudaErrchk(cudaFree(ptr));
}







GPU_TEST(Kernel, CudaExec_1blockexec)
{
  using namespace RAJA;


  constexpr long N = (long)64;  //*1024;

  // Loop Fusion
  using Pol = KernelPolicy<CudaKernel<For<0, cuda_block_x_loop, Lambda<0>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),

      [=] __device__(int i) { trip_count += 1; });
  cudaErrchk(cudaDeviceSynchronize());


  long result = (long)trip_count;

  ASSERT_EQ(result, N);
}


GPU_TEST(Kernel, CudaExec_2threadloop)
{
  using namespace RAJA;


  constexpr long N = (long)1024;

  // Loop Fusion
  using Pol = KernelPolicy<CudaKernel<
      For<0, cuda_thread_x_loop, For<1, cuda_thread_y_loop, Lambda<0>>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N), RangeSegment(0, N)),

      [=] __device__(ptrdiff_t i, ptrdiff_t j) { trip_count += 1; });
  cudaErrchk(cudaDeviceSynchronize());

  long result = (long)trip_count;

  ASSERT_EQ(result, N * N);
}

GPU_TEST(Kernel, CudaExec_1thread1block)
{
  using namespace RAJA;


  constexpr long N = (long)1024;

  // Loop Fusion
  using Pol = KernelPolicy<
      CudaKernel<For<0, cuda_block_x_loop, For<1, cuda_thread_x_loop, Lambda<0>>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N), RangeSegment(0, N)),

      [=] __device__(ptrdiff_t i, ptrdiff_t j) {
        if (i == j) {
          trip_count += 1;
        }
      });
  cudaErrchk(cudaDeviceSynchronize());

  long result = (long)trip_count;

  ASSERT_EQ(result, N);
}


GPU_TEST(Kernel, CudaExec_3threadloop)
{
  using namespace RAJA;


  constexpr long N = (long)256;

  // Loop Fusion
  using Pol = KernelPolicy<CudaKernel<
      For<0,
          cuda_thread_z_loop,
          For<1, cuda_thread_x_loop, For<2, cuda_thread_y_loop, Lambda<0>>>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N),
                       RangeSegment(0, N),
                       RangeSegment(0, N)),

      [=] __device__(ptrdiff_t i, ptrdiff_t j, ptrdiff_t k) {
        trip_count += 1;
      });
  cudaErrchk(cudaDeviceSynchronize());

  long result = (long)trip_count;

  ASSERT_EQ(result, N * N * N);
}


GPU_TEST(Kernel, CudaExec_tile1threaddirect)
{
  using namespace RAJA;


  constexpr long N = (long)1024 * 1024;

  // Loop Fusion
  using Pol = KernelPolicy<
      CudaKernel<statement::Tile<0,
                                 tile_fixed<128>,
                                 seq_exec,
                                 For<0, cuda_thread_x_direct, Lambda<0>>>>>;


  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),

      [=] __device__(ptrdiff_t i) { trip_count += 1; });
  cudaErrchk(cudaDeviceSynchronize());


  long result = (long)trip_count;

  ASSERT_EQ(result, N);
}


#endif  // CUDA


TEST(Kernel, Hyperplane_seq)
{

  using namespace RAJA;


  constexpr long N = (long)4;

  using Pol =
      KernelPolicy<Hyperplane<0, seq_exec, ArgList<1>, seq_exec, Lambda<0>>>;


  RAJA::ReduceSum<seq_reduce, long> trip_count(0);


  kernel<Pol>(

      RAJA::make_tuple(TypedRangeSegment<int>(0, N),
                       TypedRangeSegment<int>(0, N)),

      [=](int, int) { trip_count += 1; });

  long result = (long)trip_count;

  ASSERT_EQ(result, N * N);
}


#if defined(RAJA_ENABLE_CUDA)



GPU_TEST(Kernel, Hyperplane_cuda_2d)
{
  using namespace RAJA;

  using Pol =
      RAJA::KernelPolicy<CudaKernel<
        For<1, cuda_thread_x_direct,
          Hyperplane<0, seq_exec, ArgList<1>,
                                     Lambda<0>, CudaSyncThreads>>>>;

  constexpr long N = (long)24;
  constexpr long M = (long)11;

  int *x = nullptr;
  cudaErrchk(cudaMallocManaged(&x, N * M * sizeof(int)));


  using myview = View<int, Layout<2, RAJA::Index_type>>;
  myview xv{x, N, M};

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, N),
                                     RAJA::RangeSegment(0, M)),
                    [=] __device__(Index_type i, Index_type j) {
                      int left = 1;
                      if (i > 0) {
                        left = xv(i - 1, j);
                      }

                      int up = 1;
                      if (j > 0) {
                        up = xv(i, j - 1);
                      }

                      xv(i, j) = left + up;
                    });

  cudaErrchk(cudaDeviceSynchronize());

  for (int i = 1; i < N; ++i) {
    for (int j = 1; j < M; ++j) {
      ASSERT_EQ(xv(i, j), xv(i - 1, j) + xv(i, j - 1));
    }
  }

  cudaErrchk(cudaFree(x));
}

GPU_TEST(Kernel, Hyperplane_cuda_2d_negstride)
{
  using namespace RAJA;

  using Pol =
      RAJA::KernelPolicy<CudaKernel<
        For<1, cuda_thread_y_direct,
          Hyperplane<0, seq_exec, ArgList<1>,
                                           Lambda<0>, CudaSyncThreads>>>>;

  constexpr long N = (long)24;
  constexpr long M = (long)11;

  int *x = nullptr;
  cudaErrchk(cudaMallocManaged(&x, N * M * sizeof(int)));


  using myview = View<int, Layout<2, RAJA::Index_type>>;
  myview xv{x, N, M};

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeStrideSegment(N - 1, -1, -1),
                                     RAJA::RangeStrideSegment(M - 1, -1, -1)),
                    [=] __device__(Index_type i, Index_type j) {
                      int right = 1;
                      if (i < N - 1) {
                        right = xv(i + 1, j);
                      }

                      int down = 1;
                      if (j < M - 1) {
                        down = xv(i, j + 1);
                      }

                      xv(i, j) = right + down;
                    });

  cudaErrchk(cudaDeviceSynchronize());

  for (int i = 0; i < N - 1; ++i) {
    for (int j = 0; j < M - 1; ++j) {
      ASSERT_EQ(xv(i, j), xv(i + 1, j) + xv(i, j + 1));
    }
  }

  cudaErrchk(cudaFree(x));
}



GPU_TEST(Kernel, Hyperplane_cuda_3d_tiled)
{
  using namespace RAJA;

  using Pol = RAJA::KernelPolicy<CudaKernel<
      For<0, cuda_block_x_loop,
        RAJA::statement::Tile<2, RAJA::tile_fixed<13>, seq_exec,
          RAJA::statement::Tile<3, RAJA::tile_fixed<7>, seq_exec,
            For<2, cuda_thread_x_direct,
              For<3, cuda_thread_y_direct,
                Hyperplane<1, seq_exec, ArgList<2, 3>,
                                           Lambda<0>, CudaSyncThreads>>>>>>>>;

  constexpr long L = (long)1;
  constexpr long N = (long)11;
  constexpr long M = (long)27;
  constexpr long O = (long)13;

  long *x = nullptr;
  cudaErrchk(cudaMallocManaged(&x, N * M * O * sizeof(long)));


  using myview =
      TypedView<long, Layout<3, RAJA::Index_type>, ZoneI, ZoneJ, ZoneK>;
  myview xv{x, N, M, O};

  for (long i = 0; i < N * M * O; ++i) {
    x[i] = i;
  }

  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);
  RAJA::ReduceSum<cuda_reduce, long> oob_count(0);

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, L),
                       RAJA::TypedRangeStrideSegment<ZoneI>(0, N, 1),
                       RAJA::TypedRangeStrideSegment<ZoneJ>(M - 1, -1, -1),
                       RAJA::TypedRangeStrideSegment<ZoneK>(0, O, 1)),
      [=] __device__(int g, ZoneI i, ZoneJ j, ZoneK k) {
        if (i < 0 || i >= N || j < 0 || j >= M || k < 0 || k >= O) {
          oob_count += 1;
        }

        long left = 1;
        if (i > 0) {
          left = xv(i - 1, j, k);
        }

        long up = 1;
        if (j > 0) {
          up = xv(i, j - 1, k);
        }

        long back = 1;
        if (k > 0) {
          back = xv(i, j, k - 1);
        }

        xv(i, j, k) = left + up + back;

        trip_count += 1;
      });

  cudaErrchk(cudaDeviceSynchronize());


  ASSERT_EQ((long)trip_count, (long)L * N * M * O);
  ASSERT_EQ((long)oob_count, (long)0);


  long y[N][M][O];
  long *y_ptr = &y[0][0][0];

  for (long i = 0; i < N * M * O; ++i) {
    y_ptr[i] = i;
  }

  for (int l = 0; l < L; ++l) {
    for (int i = 0; i < N; ++i) {
      for (int j = M - 1; j >= 0; --j) {
        for (int k = 0; k < O; ++k) {
          long left = 1;
          if (i > 0) {
            left = y[i - 1][j][k];
          }

          long up = 1;
          if (j > 0) {
            up = y[i][j - 1][k];
          }

          long back = 1;
          if (k > 0) {
            back = y[i][j][k - 1];
          }

          y[i][j][k] = left + up + back;
        }
      }
    }
  }


  for (ZoneI i(0); i < N; ++i) {
    for (ZoneJ j(0); j < M; ++j) {
      for (ZoneK k(0); k < O; ++k) {
        ASSERT_EQ(xv(i, j, k), y[*i][*j][*k]);
      }
    }
  }

  cudaErrchk(cudaFree(x));
}



GPU_TEST(Kernel, CudaExec_1threadexec)
{
  using namespace RAJA;


  constexpr long N = (long)200;

  // Loop Fusion
  using Pol = KernelPolicy<CudaKernel<
      statement::Tile<2, tile_fixed<64>, cuda_block_z_loop,
        statement::Tile<3, tile_fixed<16>, seq_exec,
          For<0, cuda_block_x_loop,
            For<1, cuda_block_y_loop,
              For<2, cuda_thread_x_direct,
                For<3, cuda_thread_y_direct,
                  Lambda<0>
                >
              >
            >
          >
        >
      >
    >
  >;



  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N),
                       RangeSegment(0, N),
                       RangeSegment(0, N),
                       RangeSegment(0, N)),

      [=] __device__(Index_type i, Index_type j, Index_type k, Index_type l) {
        trip_count += 1;
      });

  cudaErrchk(cudaDeviceSynchronize());


  long result = (long)trip_count;

  ASSERT_EQ(result, N * N * N * N);
}

GPU_TEST(Kernel, CudaExec_fixedspillexec)
{
  using namespace RAJA;


  constexpr long N = (long)2048;
  constexpr long M = (long)32;

  // Loop Fusion
  using Pol = KernelPolicy<CudaKernelFixed<1024,
      statement::Tile<0, tile_fixed<1024>, cuda_block_x_loop,
        For<0, cuda_thread_x_direct,
          Lambda<0>
        >
      >
    >
  >;


  long *x = nullptr;
  cudaErrchk(cudaMallocManaged(&x, (N+M) * sizeof(long)));
  long *y = x;

  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),

      [=] __device__(Index_type i) {
        constexpr long M = (long)32; // M must be constexpr on the device
        long a[M];
        for (int j = 0; j < M; ++j) {
          a[j] = x[i+j];
          y[i+j] = a[j];
        }
        trip_count += 1;
        for (int j = 0; j < M; ++j) {
          x[i+j] = a[j];
        }
      });

  cudaErrchk(cudaDeviceSynchronize());


  long result = (long)trip_count;

  ASSERT_EQ(result, N);

  cudaErrchk(cudaFree(x));
}

GPU_TEST(Kernel, CudaExec_fixedsmspillexec)
{
  using namespace RAJA;


  constexpr long N = (long)2048;
  constexpr long M = (long)32;

  // Loop Fusion
  using Pol = KernelPolicy<CudaKernelFixedSM<512, 3,
      statement::Tile<0, tile_fixed<512>, cuda_block_x_loop,
        For<0, cuda_thread_x_direct,
          Lambda<0>
        >
      >
    >
  >;


  long *x = nullptr;
  cudaErrchk(cudaMallocManaged(&x, (N+M) * sizeof(long)));
  long *y = x;

  RAJA::ReduceSum<cuda_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),

      [=] __device__(Index_type i) {
        constexpr long M = (long)32; // M must be constexpr on the device
        long a[M];
        for (int j = 0; j < M; ++j) {
          a[j] = x[i+j];
          y[i+j] = a[j];
        }
        trip_count += 1;
        for (int j = 0; j < M; ++j) {
          x[i+j] = a[j];
        }
      });

  cudaErrchk(cudaDeviceSynchronize());


  long result = (long)trip_count;

  ASSERT_EQ(result, N);

  cudaErrchk(cudaFree(x));
}

#endif

#if defined(RAJA_ENABLE_HIP)


GPU_TEST(Kernel_gpu, ReduceHipSum1)
{

  long N = 2345;

  using Pol =
      KernelPolicy<HipKernel<
        For<0, hip_thread_x_loop, Lambda<0>>,
        RAJA::statement::Reduce<hip_block_reduce, RAJA::operators::plus, Param<0>,
          Lambda<1, Params<0>>
        >
      >>;

  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      RAJA::make_tuple((long)0),

      [=] __device__ (Index_type i, long &value) {
        value += i;
      },
      [=] __device__ (long &value) {
        // This only gets executed on the "root" thread which receieved the
        // reduced value
        trip_count += value;
      });

  ASSERT_EQ(trip_count.get(), N*(N-1)/2);

}

GPU_TEST(Kernel, HipWarpLoop1)
{

  long N = 2345;

  using Pol =
      KernelPolicy<HipKernel<
        For<0, hip_warp_loop, Lambda<0>>
      >>;

  RAJA::ReduceSum<hip_reduce, long> value(0);

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      [=] __device__ (Index_type i) {
        value += i;
      });

  ASSERT_EQ(value.get(), N*(N-1)/2);

}

GPU_TEST(Kernel, HipWarpLoop2)
{

  long N = 2345;

  using Pol =
      KernelPolicy<HipKernel<
        Tile<0, tile_fixed<32>, seq_exec,
          For<0, hip_warp_direct, Lambda<0>>
        >
      >>;

  RAJA::ReduceSum<hip_reduce, long> value(0);

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      [=] __device__ (Index_type i) {
        value += i;
      });

  ASSERT_EQ(value.get(), N*(N-1)/2);

}

GPU_TEST(Kernel, HipWarpLoop3)
{

  long A = 32;
  long B = 43;
  long N = A*B;

  using Pol =
      KernelPolicy<HipKernel<
        Tile<0, tile_fixed<32>, seq_exec,
          ForICount<0, Param<0>, hip_warp_direct, Lambda<0>>
        >
      >>;

  RAJA::ReduceSum<hip_reduce, long> value(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      RAJA::make_tuple((Index_type)0),

      [=] __device__ (Index_type RAJA_UNUSED_ARG(i), 
                      Index_type j) {
        value += j;  // j should only be 0..31
      });

  ASSERT_EQ(value.get(), B*A*(A-1)/2);

}

GPU_TEST(Kernel, HipWarpLoop4)
{

  long A = 16;
  long B = 43;

  using maskA = BitMask<4,0>;
  using maskB = BitMask<1,4>;

  using Pol =
      KernelPolicy<HipKernel<
        For<0, hip_warp_masked_direct<maskA>,
          For<1, hip_warp_masked_loop<maskB>,
            Lambda<0>
          >
        >
      >>;

  RAJA::ReduceSum<hip_reduce, long> trip_count(0);
  RAJA::ReduceSum<hip_reduce, long> value(0);

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, A), RAJA::RangeSegment(0, B)),

      [=] __device__ (Index_type i, 
                      Index_type RAJA_UNUSED_ARG(j)) {
        trip_count += 1;
        value += i;  // i should only be 0..A-1
      });

  //factor of 2 since wavefronts are 64-wide on AMD
  ASSERT_EQ(trip_count.get(), 2*A*B);
  ASSERT_EQ(value.get(), B*A*(A-1));

}

GPU_TEST(Kernel, HipThreadMasked1)
{

  long A = 64;
  long B = 943;

  using maskA = BitMask<6,0>;
  using maskB = BitMask<2,6>;

  using Pol =
      KernelPolicy<HipKernel<
        For<0, hip_thread_masked_direct<maskA>,
          For<1, hip_thread_masked_loop<maskB>,
            Lambda<0>
          >
        >
      >>;

  RAJA::ReduceMax<hip_reduce, int> max_thread(0);
  RAJA::ReduceSum<hip_reduce, long> trip_count(0);
  RAJA::ReduceSum<hip_reduce, long> value(0);

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, A), RAJA::RangeSegment(0, B)),

      [=] __device__ (Index_type i, Index_type RAJA_UNUSED_ARG(j)) {
        trip_count += 1;
        value += i;  // i should only be 0..A-1
        max_thread.max(threadIdx.x);
      });

  ASSERT_EQ(max_thread.get(), 255);
  ASSERT_EQ(trip_count.get(), A*B);
  ASSERT_EQ(value.get(), B*A*(A-1)/2);

}

GPU_TEST(Kernel, HipThreadMasked2)
{

  long A = 64;
  long B = 4*123;

  using maskA = BitMask<6,0>;
  using maskB = BitMask<2,6>;

  using Pol =
      KernelPolicy<HipKernel<
        ForICount<0, Param<0>, hip_thread_masked_direct<maskA>,
        ForICount<1, Param<1>, hip_thread_masked_loop<maskB>,
            Lambda<0>
          >
        >
      >>;

  RAJA::ReduceMax<hip_reduce, int> max_thread(0);
  RAJA::ReduceSum<hip_reduce, long> trip_count(0);
  RAJA::ReduceSum<hip_reduce, long> value(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, A), RAJA::RangeSegment(0, B)),

      RAJA::make_tuple((Index_type)0, (Index_type)0),

      [=] __device__ (Index_type RAJA_UNUSED_ARG(i), 
                      Index_type RAJA_UNUSED_ARG(j), 
                      Index_type RAJA_UNUSED_ARG(x), 
                      Index_type y) {
        trip_count += 1;
        value += y;  // i should only be 0..3
        max_thread.max(threadIdx.x);
      });

  ASSERT_EQ(max_thread.get(), 255);
  ASSERT_EQ(trip_count.get(), A*B);
  ASSERT_EQ(value.get(), A*B*(B-1)/2);

}


GPU_TEST(Kernel_gpu, ReduceHipWarpLoop1)
{

  long N = 2345;

  using Pol =
      KernelPolicy<HipKernel<
        For<0, hip_warp_loop, Lambda<0>>,
        RAJA::statement::Reduce<hip_warp_reduce, RAJA::operators::plus, Param<0>,
          Lambda<1, Params<0>>
        >
      >>;

  RAJA::ReduceSum<hip_reduce, long> total_count(0);
  RAJA::ReduceSum<hip_reduce, long> reduce_count(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N)),

      RAJA::make_tuple((long)0),

      [=] __device__ (Index_type i, long &value) {
        value += i;
      },
      [=] __device__ (long &value) {
        // This only gets executed on the "root" thread which recieved the
        // reduced value
        total_count += value;
        reduce_count += 1;
      });

  ASSERT_EQ(total_count.get(), N*(N-1)/2);
  ASSERT_EQ(reduce_count.get(), 1);

}


GPU_TEST(Kernel_gpu, ReduceHipWarpLoop2)
{

  long N = 239;
  long M = 16;

  using Pol =
      KernelPolicy<HipKernel<
        For<1, hip_thread_y_loop,
          For<0, hip_warp_loop, Lambda<0>>
        >,
        RAJA::statement::Reduce<hip_warp_reduce, RAJA::operators::plus, Param<0>,
          Lambda<1, Params<0> >
        >
      >>;

  RAJA::ReduceSum<hip_reduce, long> total_count(0);
  RAJA::ReduceSum<hip_reduce, long> reduce_count(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N),
                       RAJA::RangeSegment(0, M)),

      RAJA::make_tuple((long)0),

      [=] __device__ (Index_type i, Index_type j, long &value) {
        value += i + j*N;
      },
      [=] __device__ (long &value) {
        // This only gets executed on the "root" thread which recieved the
        // reduced value
        total_count += value;
        reduce_count += 1;
      });

  long NM = N*M;
  ASSERT_EQ(total_count.get(), NM*(NM-1)/2);
  ASSERT_EQ(reduce_count.get(), M);

}


GPU_TEST(Kernel_gpu, ReduceHipWarpLoop3)
{

  long N = 25;
  long M = 16;
  long O = 10;

  using Pol =
      KernelPolicy<HipKernel<
        For<2, hip_block_x_loop,
          For<1, hip_thread_y_direct,
            For<0, hip_warp_direct, Lambda<0>>
          >,
          RAJA::statement::HipSyncThreads,
          RAJA::statement::Reduce<hip_warp_reduce, RAJA::operators::plus, Param<0>,
            Lambda<1, Params<0>>
          >
        >
      >>;

  RAJA::ReduceSum<hip_reduce, long> total_count(0);
  RAJA::ReduceSum<hip_reduce, long> reduce_count(0);

  RAJA::kernel_param<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, N),
                       RAJA::RangeSegment(0, M),
                       RAJA::RangeSegment(0, O)),

      RAJA::make_tuple((long)0),
      [=] __device__ (Index_type i, Index_type j, Index_type k, long &value) {
        value = i + j*N + k*N*M;
      },
      [=] __device__ (long &value) {
        // This only gets executed on the "root" thread which reecieved the
        // reduced value
        total_count += value;
        reduce_count += 1;
      });

  long NMO = N*M*O;
  ASSERT_EQ(total_count.get(), NMO*(NMO-1)/2);
  ASSERT_EQ(reduce_count.get(), M*O);

}


GPU_TEST(Kernel_gpu, HipTileTCount)
{
  using namespace RAJA;


  constexpr long N = 1035;
  constexpr long T = 32;
  constexpr long NT = (N+T-1)/T;

  // Loop Fusion
  using Pol =
      KernelPolicy<HipKernel<
       statement::TileTCount<0, Param<0>, tile_fixed<T>, hip_block_x_loop,
         For<0, hip_thread_x_direct, Lambda<0>>>>>;


  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  for (long t = 0; t < NT; ++t) {
    RAJA::ReduceSum<hip_reduce, long> tile_count(0);

    kernel_param<Pol>(

        RAJA::make_tuple(RangeSegment(0, N)),
        RAJA::make_tuple((RAJA::Index_type)0),

        [=] __device__(RAJA::Index_type i, RAJA::Index_type ti) {
          trip_count += 1;
          if (i/T == t && ti == t) {
            tile_count += 1;
          }
        });

    long trip_result = (long)trip_count;
    long tile_result = (long)tile_count;

    ASSERT_EQ(trip_result, (t+1)*N);

    long tile_expect = T;
    if ((t+1)*T > N) {
      tile_expect = N - t*T;
    }
    ASSERT_EQ(tile_result, tile_expect);
  }
}


GPU_TEST(Kernel_gpu, HipConditional)
{
  using namespace RAJA;


  constexpr long N = (long)1024;

  // Loop Fusion
  using Pol = KernelPolicy<HipKernel<
      statement::Tile<0, tile_fixed<32>, hip_block_x_loop,
        For<0, hip_thread_x_loop, If<Param<0>, Lambda<0>>, Lambda<1>>>>>;

  for (int param = 0; param < 2; ++param) {

    RAJA::ReduceSum<hip_reduce, long> trip_count(0);

    kernel_param<Pol>(

        RAJA::make_tuple(TypedRangeSegment<int>(0, N)),

        RAJA::make_tuple((bool)param),

        // This only gets executed if param==1
        [=] RAJA_DEVICE(int RAJA_UNUSED_ARG(i), bool) { trip_count += 2; },

        // This always gets executed
        [=] RAJA_DEVICE(int RAJA_UNUSED_ARG(i), bool) { trip_count += 1; });

    long result = (long)trip_count;

    ASSERT_EQ(result, N * (2 * param + 1));
  }
}


GPU_TEST(Kernel_gpu, HipExec1)
{
  using namespace RAJA;


  constexpr long N = (long)3 * 1024 * 1024;

  // Loop Fusion
  using Pol =
      KernelPolicy<HipKernel<For<0, hip_thread_x_loop, Lambda<0>>>>;


  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),

      [=] __device__(ptrdiff_t RAJA_UNUSED_ARG(i)) { trip_count += 1; });

  long result = (long)trip_count;

  ASSERT_EQ(result, N);
}


GPU_TEST(Kernel_gpu, HipExec1ab)
{
  using namespace RAJA;


  constexpr long N = (long)128;

  // Loop Fusion
  using Pol = KernelPolicy<HipKernel<
      For<0, hip_block_x_loop,
        For<1, hip_block_y_loop,
          For<2, hip_block_z_loop,
            Lambda<0>
   >>>>>;


  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N),
                       RangeSegment(0, N),
                       RangeStrideSegment(0, N, 2)),

      [=] __device__(ptrdiff_t RAJA_UNUSED_ARG(i), 
                     ptrdiff_t RAJA_UNUSED_ARG(j), 
                     ptrdiff_t RAJA_UNUSED_ARG(k)) {
        trip_count += 1;
      });

  long result = (long)trip_count;

  ASSERT_EQ(result, N * N * N / 2);
}


GPU_TEST(Kernel_gpu, HipExec1c)
{
  using namespace RAJA;


  constexpr long N = (long)3;

  // Loop Fusion
  using Pol = KernelPolicy<
      HipKernelExt<hip_explicit_launch<false, 5, 3>,
           statement::Tile<2, tile_fixed<2>, hip_block_z_loop,
                    For<0, hip_block_x_loop,
                        For<1, hip_block_y_loop,
                            For<2, hip_thread_z_loop, Lambda<0>>>>>>>;


  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N),
                       RangeSegment(0, N),
                       RangeSegment(0, N)),

      [=] __device__(RAJA::Index_type RAJA_UNUSED_ARG(i),
                     RAJA::Index_type RAJA_UNUSED_ARG(j),
                     RAJA::Index_type RAJA_UNUSED_ARG(k)) { 
            trip_count += 1; });

  long result = (long)trip_count;

  ASSERT_EQ(result, N * N * N);
}


GPU_TEST(Kernel_gpu, HipComplexNested)
{
  using namespace RAJA;


  constexpr long N = (long)739;

  using Pol = KernelPolicy<HipKernel<
      For<0, hip_block_x_loop,
          For<1, hip_thread_x_loop, For<2, hip_thread_y_loop, Lambda<0>>>,
          For<2, hip_thread_x_loop, Lambda<0, Segs<0>, ValuesT<RAJA::Index_type, 0>, Segs<2>>>>>>;

  int *ptr = (int*) malloc(sizeof(int) * N);
  int *d_ptr = nullptr;
  hipErrchk(hipMalloc(&d_ptr, sizeof(int) * N) );

  for (long i = 0; i < N; ++i) {
    ptr[i] = 0;
  }
  hipErrchk(hipMemcpy(d_ptr, ptr, sizeof(int) * N, hipMemcpyHostToDevice));

  auto segments = RAJA::make_tuple(RangeSegment(0, N),
                                   RangeSegment(0, N),
                                   RangeSegment(0, N));


  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  kernel<Pol>(

      segments,

      [=] __device__(RAJA::Index_type i,
                     RAJA::Index_type RAJA_UNUSED_ARG(j),
                     RAJA::Index_type RAJA_UNUSED_ARG(k)) {
        trip_count += 1;
        RAJA::atomicAdd<RAJA::auto_atomic>(d_ptr + i, (int)1);
      });
  hipErrchk(hipMemcpy(ptr, d_ptr, sizeof(int) * N, hipMemcpyDeviceToHost));

  for (long i = 0; i < N; ++i) {
    ASSERT_EQ(ptr[i], (int)(N * N + N));
  }

  // check trip count
  long result = (long)trip_count;
  ASSERT_EQ(result, N * N * N + N * N);

  free(ptr);
  hipErrchk(hipFree(d_ptr));
}







GPU_TEST(Kernel_gpu, HipExec_1blockexec)
{
  using namespace RAJA;


  constexpr long N = (long)64;  //*1024;

  // Loop Fusion
  using Pol = KernelPolicy<HipKernel<For<0, hip_block_x_loop, Lambda<0>>>>;


  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),

      [=] __device__(int RAJA_UNUSED_ARG(i)) { trip_count += 1; });

  long result = (long)trip_count;

  ASSERT_EQ(result, N);
}


GPU_TEST(Kernel_gpu, HipExec_2threadloop)
{
  using namespace RAJA;


  constexpr long N = (long)1024;

  // Loop Fusion
  using Pol = KernelPolicy<HipKernel<
      For<0, hip_thread_x_loop, For<1, hip_thread_y_loop, Lambda<0>>>>>;


  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N), RangeSegment(0, N)),

      [=] __device__(ptrdiff_t RAJA_UNUSED_ARG(i), 
                     ptrdiff_t RAJA_UNUSED_ARG(j)) 
            { trip_count += 1; });

  long result = (long)trip_count;

  ASSERT_EQ(result, N * N);
}

GPU_TEST(Kernel_gpu, HipExec_1thread1block)
{
  using namespace RAJA;


  constexpr long N = (long)1024;

  // Loop Fusion
  using Pol = KernelPolicy<
      HipKernel<For<0, hip_block_x_loop, For<1, hip_thread_x_loop, Lambda<0>>>>>;


  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N), RangeSegment(0, N)),

      [=] __device__(ptrdiff_t i, ptrdiff_t j) {
        if (i == j) {
          trip_count += 1;
        }
      });

  long result = (long)trip_count;

  ASSERT_EQ(result, N);
}


GPU_TEST(Kernel_gpu, HipExec_3threadloop)
{
  using namespace RAJA;


  constexpr long N = (long)256;

  // Loop Fusion
  using Pol = KernelPolicy<HipKernel<
      For<0,
          hip_thread_z_loop,
          For<1, hip_thread_x_loop, For<2, hip_thread_y_loop, Lambda<0>>>>>>;


  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N),
                       RangeSegment(0, N),
                       RangeSegment(0, N)),

      [=] __device__(ptrdiff_t RAJA_UNUSED_ARG(i), 
                     ptrdiff_t RAJA_UNUSED_ARG(j), 
                     ptrdiff_t RAJA_UNUSED_ARG(k)) {
        trip_count += 1;
      });

  long result = (long)trip_count;

  ASSERT_EQ(result, N * N * N);
}


GPU_TEST(Kernel_gpu, Hyperplane_hip_2d)
{
  using namespace RAJA;

  using Pol =
      RAJA::KernelPolicy<HipKernel<
        For<1, hip_thread_x_direct,
          Hyperplane<0, seq_exec, ArgList<1>,
                                     Lambda<0>, HipSyncThreads>>>>;

  constexpr long N = (long)24;
  constexpr long M = (long)11;

  int *x = (int*) malloc(N*M*sizeof(int));
  int *d_x = nullptr;
  hipErrchk(hipMalloc(&d_x, N*M*sizeof(int)));

  using myview = View<int, Layout<2, RAJA::Index_type>>;
  myview xv{x, N, M};
  myview d_xv{d_x, N, M};

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeSegment(0, N),
                                     RAJA::RangeSegment(0, M)),
                    [=] __device__(Index_type i, Index_type j) {
                      int left = 1;
                      if (i > 0) {
                        left = d_xv(i - 1, j);
                      }

                      int up = 1;
                      if (j > 0) {
                        up = d_xv(i, j - 1);
                      }

                      d_xv(i, j) = left + up;
                    });

  hipErrchk(hipMemcpy(x, d_x, N*M*sizeof(int), hipMemcpyDeviceToHost));

  for (int i = 1; i < N; ++i) {
    for (int j = 1; j < M; ++j) {
      ASSERT_EQ(xv(i, j), xv(i - 1, j) + xv(i, j - 1));
    }
  }

  free(x);
  hipErrchk(hipFree(d_x));
}

GPU_TEST(Kernel_gpu, Hyperplane_hip_2d_negstride)
{
  using namespace RAJA;

  using Pol =
      RAJA::KernelPolicy<HipKernel<
        For<1, hip_thread_y_direct,
          Hyperplane<0, seq_exec, ArgList<1>,
                                           Lambda<0>, HipSyncThreads>>>>;

  constexpr long N = (long)24;
  constexpr long M = (long)11;


  int *x = (int*) malloc(N*M*sizeof(int));
  int *d_x = nullptr;
  hipErrchk(hipMalloc(&d_x, N*M*sizeof(int)));


  using myview = View<int, Layout<2, RAJA::Index_type>>;
  myview xv{x, N, M};
  myview d_xv{d_x, N, M};

  RAJA::kernel<Pol>(RAJA::make_tuple(RAJA::RangeStrideSegment(N - 1, -1, -1),
                                     RAJA::RangeStrideSegment(M - 1, -1, -1)),
                    [=] __device__(Index_type i, Index_type j) {
                      int right = 1;
                      if (i < N - 1) {
                        right = d_xv(i + 1, j);
                      }

                      int down = 1;
                      if (j < M - 1) {
                        down = d_xv(i, j + 1);
                      }

                      d_xv(i, j) = right + down;
                    });

  hipErrchk(hipMemcpy(x, d_x, N*M*sizeof(int), hipMemcpyDeviceToHost));

  for (int i = 0; i < N - 1; ++i) {
    for (int j = 0; j < M - 1; ++j) {
      ASSERT_EQ(xv(i, j), xv(i + 1, j) + xv(i, j + 1));
    }
  }

  free(x);
  hipErrchk(hipFree(d_x));
}



GPU_TEST(Kernel_gpu, Hyperplane_hip_3d_tiled)
{
  using namespace RAJA;

  using Pol = RAJA::KernelPolicy<HipKernel<
      For<0, hip_block_x_loop,
        RAJA::statement::Tile<2, RAJA::tile_fixed<13>, seq_exec,
          RAJA::statement::Tile<3, RAJA::tile_fixed<7>, seq_exec,
            For<2, hip_thread_x_direct,
              For<3, hip_thread_y_direct,
                Hyperplane<1, seq_exec, ArgList<2, 3>,
                                           Lambda<0>, HipSyncThreads>>>>>>>>;

  constexpr long L = (long)1;
  constexpr long N = (long)11;
  constexpr long M = (long)27;
  constexpr long O = (long)13;

  long *x = (long*) malloc(N*M*O*sizeof(long));
  long *d_x = nullptr;
  hipErrchk(hipMalloc(&d_x, N*M*O*sizeof(long)));


  using myview =
      TypedView<long, Layout<3, RAJA::Index_type>, ZoneI, ZoneJ, ZoneK>;
  myview xv{x, N, M, O};
  myview d_xv{d_x, N, M, O};

  for (long i = 0; i < N * M * O; ++i) {
    x[i] = i;
  }
  hipErrchk(hipMemcpy(d_x, x, N*M*O*sizeof(long), hipMemcpyHostToDevice));

  RAJA::ReduceSum<hip_reduce, long> trip_count(0);
  RAJA::ReduceSum<hip_reduce, long> oob_count(0);

  RAJA::kernel<Pol>(
      RAJA::make_tuple(RAJA::RangeSegment(0, L),
                       RAJA::TypedRangeStrideSegment<ZoneI>(0, N, 1),
                       RAJA::TypedRangeStrideSegment<ZoneJ>(M - 1, -1, -1),
                       RAJA::TypedRangeStrideSegment<ZoneK>(0, O, 1)),
      [=] __device__(int RAJA_UNUSED_ARG(g), ZoneI i, ZoneJ j, ZoneK k) {
        if (i < 0 || i >= N || j < 0 || j >= M || k < 0 || k >= O) {
          oob_count += 1;
        }

        long left = 1;
        if (i > 0) {
          left = d_xv(i - 1, j, k);
        }

        long up = 1;
        if (j > 0) {
          up = d_xv(i, j - 1, k);
        }

        long back = 1;
        if (k > 0) {
          back = d_xv(i, j, k - 1);
        }

        d_xv(i, j, k) = left + up + back;

        trip_count += 1;
      });

  hipErrchk(hipMemcpy(x, d_x, N*M*O*sizeof(long), hipMemcpyDeviceToHost));

  ASSERT_EQ((long)trip_count, (long)L * N * M * O);
  ASSERT_EQ((long)oob_count, (long)0);


  long y[N][M][O];
  long *y_ptr = &y[0][0][0];

  for (long i = 0; i < N * M * O; ++i) {
    y_ptr[i] = i;
  }

  for (int l = 0; l < L; ++l) {
    for (int i = 0; i < N; ++i) {
      for (int j = M - 1; j >= 0; --j) {
        for (int k = 0; k < O; ++k) {
          long left = 1;
          if (i > 0) {
            left = y[i - 1][j][k];
          }

          long up = 1;
          if (j > 0) {
            up = y[i][j - 1][k];
          }

          long back = 1;
          if (k > 0) {
            back = y[i][j][k - 1];
          }

          y[i][j][k] = left + up + back;
        }
      }
    }
  }


  for (ZoneI i(0); i < N; ++i) {
    for (ZoneJ j(0); j < M; ++j) {
      for (ZoneK k(0); k < O; ++k) {
        ASSERT_EQ(xv(i, j, k), y[*i][*j][*k]);
      }
    }
  }

  free(x);
  hipErrchk(hipFree(d_x));
}



GPU_TEST(Kernel_gpu, HipExec_1threadexec)
{
  using namespace RAJA;


  constexpr long N = (long)200;

  // Loop Fusion
  using Pol = KernelPolicy<HipKernel<
      statement::Tile<2, tile_fixed<64>, hip_block_z_loop,
        statement::Tile<3, tile_fixed<16>, seq_exec,
          For<0, hip_block_x_loop,
            For<1, hip_block_y_loop,
              For<2, hip_thread_x_direct,
                For<3, hip_thread_y_direct,
                  Lambda<0>
                >
              >
            >
          >
        >
      >
    >
  >;



  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N),
                       RangeSegment(0, N),
                       RangeSegment(0, N),
                       RangeSegment(0, N)),

      [=] __device__(Index_type RAJA_UNUSED_ARG(i), 
                     Index_type RAJA_UNUSED_ARG(j), 
                     Index_type RAJA_UNUSED_ARG(k), 
                     Index_type RAJA_UNUSED_ARG(l)) {
        trip_count += 1;
      });

  long result = (long)trip_count;

  ASSERT_EQ(result, N * N * N * N);
}

GPU_TEST(Kernel, HipExec_fixedspillexec)
{
  using namespace RAJA;


  constexpr long N = (long)2048;
  constexpr long M = (long)20;

  // Loop Fusion
  using Pol = KernelPolicy<HipKernelFixed<1024,
      statement::Tile<0, tile_fixed<1024>, hip_block_x_loop,
        For<0, hip_thread_x_direct,
          Lambda<0>
        >
      >
    >
  >;


  long *x = nullptr;
  hipErrchk(hipMalloc(&x, (N+M) * sizeof(long)));
  long *y = x;

  RAJA::ReduceSum<hip_reduce, long> trip_count(0);

  kernel<Pol>(

      RAJA::make_tuple(RangeSegment(0, N)),

      [=] __device__(Index_type i) {
        constexpr long M = (long)20; // M must be constexpr on the device
        long a[M];
        for (int j = 0; j < M; ++j) {
          a[j] = x[i+j];
          y[i+j] = a[j];
        }
        trip_count += 1;
        for (int j = 0; j < M; ++j) {
          x[i+j] = a[j];
        }
      });

  long result = (long)trip_count;

  ASSERT_EQ(result, N);

  hipErrchk(hipFree(x));
}
#endif

