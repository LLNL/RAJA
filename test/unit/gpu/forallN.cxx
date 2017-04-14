#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <tuple>
#include <type_traits>

#include <cstdlib>

#include <gtest/gtest.h>
#include <RAJA/RAJA.hxx>

const int x = 500, y = 500, z = 50;

using namespace RAJA;
void stride_test(int stride) {
    double *arr = NULL;
    cudaErrchk(cudaMallocManaged(&arr, sizeof(*arr) * x * y * z));
    cudaMemset (arr, 0, sizeof(*arr) * x * y * z);

    forallN<NestedPolicy<ExecList<seq_exec, cuda_block_x_exec, cuda_thread_y_exec>, Permute<PERM_IJK>>>(
            RangeStrideSegment(0, z, stride),
            RangeStrideSegment(0, y, stride),
            RangeStrideSegment(0, x, stride),
            [=] RAJA_DEVICE (int i, int j, int k) {
            int val = z*y*i + y * j + k;
            arr[val] = val;
            });
    cudaDeviceSynchronize();

    int prev_val = 0;
    for (int i=0; i < z; i+=stride) {
        for (int j=0; j < y; j+=stride) {
            for (int k=0; k < x; k+=stride) {
                int val = z*y*i + y * j + k;
                ASSERT_EQ(arr[val], val);
                for (int inner=prev_val+1; inner < val; ++inner) {
                    ASSERT_EQ(arr[inner], 0);
                }
                prev_val = val;
            }
        }
    }
    cudaFree(arr);
}

TEST(forallN, rangeStrides) {

    stride_test(1);
    stride_test(2);
    stride_test(3);
    stride_test(4);

}
