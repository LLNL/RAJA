// RAJA/RAJA.hpp cannot be included here because the include logic will
// default all atomic implementations to a desul backend.
#include "RAJA/policy/loop/policy.hpp"
#include "RAJA/policy/openmp/atomic.hpp"
#include "RAJA/RAJA.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "RAJA/policy/cuda.hpp"
#include "RAJA/policy/cuda/atomic.hpp"
#include "desul/atomics.hpp"
#include "benchmark/benchmark.h"
#include <type_traits>
#include <iostream>

#define N 100000000
using raja_default_desul_order = desul::MemoryOrderRelaxed;
using raja_default_desul_scope = desul::MemoryScopeDevice;

template<typename AtomicType>
struct DesulAtomicSignature {
    using signature = int(*)(AtomicType*, const AtomicType, raja_default_desul_order, raja_default_desul_scope);
};

template<typename AtomicType>
struct RajaAtomicSignature {
    using signature = int(*)(AtomicType*, const AtomicType);
};

template<typename T, typename Policy>
RAJA_HOST_DEVICE T atomicAddWrapperDesul(T * acc, T value) {
    return desul::atomic_fetch_add(acc, value, raja_default_desul_order{},
                    raja_default_desul_scope{});
}

template<typename T, typename Policy>
RAJA_HOST_DEVICE T atomicMaxWrappeDesul(T * acc, T value) {
    return desul::atomic_fetch_max(acc, value, raja_default_desul_order{},
                    raja_default_desul_scope{});
}

template<typename T, typename Policy>
RAJA_HOST_DEVICE T atomicAddWrapper(T * acc, T value) {
    return RAJA::atomicAdd(Policy{}, acc, value);
}

template<typename T, typename Policy>
RAJA_HOST_DEVICE T atomicMaxWrapper(T * acc, T value) {
    return RAJA::atomicMax(Policy{}, acc, value);
}

template<typename>
struct IsCuda : public std::false_type {};

template<int M>
struct IsCuda<RAJA::cuda_exec<M>> : public std::true_type {};

template <class ExecPolicy, typename DesulAtomicSignature<int>::signature AtomicImpl>
void DesulAtomicOpLoopInt(benchmark::State& state) {
    for (auto _ : state) {
        int* value;
        int zero = 0;
        if (IsCuda<ExecPolicy>::value) {
            #if defined(RAJA_ENABLE_CUDA)
            cudaErrchk(
            cudaMallocManaged((void **)&value, sizeof(int), cudaMemAttachGlobal));
            cudaMemset(value, 0, sizeof(int));
            #endif
        } else {
            value = &zero;
        }
        RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, N),
        [=]RAJA_HOST_DEVICE(int)  {
            AtomicImpl(value, 1, raja_default_desul_order{},
                    raja_default_desul_scope{});
        });
        assert(*value == N);
    }
}

template <class ExecPolicy, typename RajaAtomicSignature<int>::signature AtomicImpl>
void AtomicOpLoopInt(benchmark::State& state) {
    for (auto _ : state) {
        int* value;
        int zero = 0;
#if defined(RAJA_ENABLE_CUDA)
    if (IsCuda<ExecPolicy>::value) {
        cudaErrchk(
        cudaMallocManaged((void **)&value, sizeof(int), cudaMemAttachGlobal));
        cudaMemset(value, 0, sizeof(int));
    }
#else
    value = &zero;
#endif
        RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, N),
        [=]RAJA_HOST_DEVICE(int)  {
            AtomicImpl(value, 1);
        });
        assert(*value == N);
    }
}

BENCHMARK(DesulAtomicOpLoopInt<RAJA::omp_for_exec, desul::atomic_fetch_add>);
BENCHMARK(AtomicOpLoopInt<RAJA::omp_for_exec, atomicAddWrapper<int, RAJA::policy::omp::omp_atomic>>);
// CUDA addition
BENCHMARK(DesulAtomicOpLoopInt<RAJA::cuda_exec<1024>, desul::atomic_fetch_add>);
BENCHMARK(AtomicOpLoopInt<RAJA::cuda_exec<1024>, atomicAddWrapper<int, RAJA::policy::cuda::cuda_atomic>>);
BENCHMARK(DesulAtomicOpLoopInt<RAJA::cuda_exec<512>, desul::atomic_fetch_add>);
BENCHMARK(AtomicOpLoopInt<RAJA::cuda_exec<512>, atomicAddWrapper<int, RAJA::policy::cuda::cuda_atomic>>);
BENCHMARK(DesulAtomicOpLoopInt<RAJA::cuda_exec<256>, desul::atomic_fetch_add>);
BENCHMARK(AtomicOpLoopInt<RAJA::cuda_exec<256>, atomicAddWrapper<int, RAJA::policy::cuda::cuda_atomic>>);
BENCHMARK(DesulAtomicOpLoopInt<RAJA::cuda_exec<128>, desul::atomic_fetch_add>);
BENCHMARK(AtomicOpLoopInt<RAJA::cuda_exec<128>, atomicAddWrapper<int, RAJA::policy::cuda::cuda_atomic>>);
BENCHMARK(DesulAtomicOpLoopInt<RAJA::cuda_exec<64>, desul::atomic_fetch_add>);
BENCHMARK(AtomicOpLoopInt<RAJA::cuda_exec<64>, atomicAddWrapper<int, RAJA::policy::cuda::cuda_atomic>>);
// CUDA max
//BENCHMARK(DesulAtomicOpLoopInt<RAJA::cuda_exec<1024>, desul::atomic_fetch_max>);
//BENCHMARK(AtomicOpLoopInt<RAJA::cuda_exec<1024>, atomicMaxWrapper<int, RAJA::policy::cuda::cuda_atomic>>);
//BENCHMARK(DesulAtomicOpLoopInt<RAJA::cuda_exec<512>, desul::atomic_fetch_max>);
//BENCHMARK(AtomicOpLoopInt<RAJA::cuda_exec<512>, atomicMaxWrapper<int, RAJA::policy::cuda::cuda_atomic>>);
//BENCHMARK(DesulAtomicOpLoopInt<RAJA::cuda_exec<256>, desul::atomic_fetch_max>);
//BENCHMARK(AtomicOpLoopInt<RAJA::cuda_exec<256>, atomicMaxWrapper<int, RAJA::policy::cuda::cuda_atomic>>);
//BENCHMARK(DesulAtomicOpLoopInt<RAJA::cuda_exec<128>, desul::atomic_fetch_max>);
//BENCHMARK(AtomicOpLoopInt<RAJA::cuda_exec<128>, atomicMaxWrapper<int, RAJA::policy::cuda::cuda_atomic>>);
//BENCHMARK(DesulAtomicOpLoopInt<RAJA::cuda_exec<64>, desul::atomic_fetch_max>);
//BENCHMARK(AtomicOpLoopInt<RAJA::cuda_exec<64>, atomicMaxWrapper<int, RAJA::policy::cuda::cuda_atomic>>);

BENCHMARK_MAIN();
