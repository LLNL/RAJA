#include "RAJA/RAJA.hpp"

using EXEC_POL8 =
    RAJA::KernelPolicy<RAJA::statement::CudaKernel<RAJA::statement::For<
        1,
        RAJA::cuda_block_x_loop,  // row
        RAJA::statement::For<
            0,
            RAJA::cuda_thread_x_loop,                     // col
            RAJA::statement::Lambda<0, RAJA::Params<0>>,  // dot = 0.0
            RAJA::statement::For<2,
                                 RAJA::seq_exec,
                                 RAJA::statement::Lambda<1>  // dot += ...
                                 >,
            RAJA::statement::
                Lambda<2, RAJA::Segs<0, 1>, RAJA::Params<0>>  // set C = ...
            >>>>;
// _matmult_3lambdakernel_cuda_end

RAJA::kernel_param<EXEC_POL8>(
    RAJA::make_tuple(col_range, row_range, dot_range),

    RAJA::tuple<double> {0.0},  // thread local variable for 'dot'

    // lambda 0
    [=] RAJA_DEVICE(double& dot) {
      dot = 0.0;
    },

    // lambda 1
    [=] RAJA_DEVICE(int col, int row, int k, double& dot) {
      dot += Aview(row, k) * Bview(k, col);
    },

    // lambda 2
    [=] RAJA_DEVICE(int col, int row, double& dot) {
      Cview(row, col) = dot;
    }

);

checkResult<double>(Cview, N);