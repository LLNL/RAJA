#include "RAJA/policy/desul/atomic.hpp"
#include "RAJA/policy/loop/policy.hpp"
#include "RAJA/policy/openmp/atomic.hpp"
#include "RAJA/RAJA.hpp"
#include "RAJA/policy/openmp/policy.hpp"
#include "desul/atomics.hpp"
#include "benchmark/benchmark.h"
#include <iostream>

#define N 10000000
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

template <class ExecPolicy, typename DesulAtomicSignature<int>::signature AtomicImpl>
void DesulAtomicOpLoopInt(benchmark::State& state) {
    int value = 0;
    for (auto _ : state) {
        RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, N),
        [=](int) RAJA_HOST_DEVICE {
            AtomicImpl(const_cast<int*>(&value), 1, raja_default_desul_order{},
                    raja_default_desul_scope{});
        });
        assert(value == N);
    }
}

template <class ExecPolicy, typename RajaAtomicSignature<int>::signature AtomicImpl>
void AtomicOpLoopInt(benchmark::State& state) {
    int value = 0;
    for (auto _ : state) {
        RAJA::forall<ExecPolicy>(RAJA::RangeSegment(0, N),
        [=](int) RAJA_HOST_DEVICE {
            AtomicImpl(const_cast<int*>(&value), 1);
        });
        assert(value == N);
    }
}

template<typename T>
T atomicAddWrapper(T * acc, T value) {
    return RAJA::atomicAdd(RAJA::policy::omp::omp_atomic{}, acc, value);
}

BENCHMARK(DesulAtomicOpLoopInt<RAJA::omp_for_exec, desul::atomic_fetch_add>);
BENCHMARK(DesulAtomicOpLoopInt<RAJA::omp_for_exec, desul::atomic_fetch_add>);
BENCHMARK(AtomicOpLoopInt<RAJA::omp_for_exec, atomicAddWrapper<int>>);

BENCHMARK_MAIN();
//int main() {
//    //AtomicOpLoopInt<RAJA::omp_for_exec, atomicAddWrapper<int>>;
//}
//