//
// File:        simpletest.cxx
// Copyright:   (c) 2017 Lawrence Livermore National Security, LLC
// Revision:    @(#) $Revision$
// Date:        $Date$
// Description: 
// 

#include <omp.h>
#include <cstdint>
#include <cstdlib>
#include "RAJA/RAJA.hxx"
#include <iostream>

RAJA_INDEX_VALUE(TimeInd, "Time Index")

RAJA_INDEX_VALUE(YInd, "Y Spatial Index")

RAJA_INDEX_VALUE(XInd, "X Spatial Index")

template<typename T>
struct TypeName {
  static const char *Get() { return typeid(T).name(); }
};
template<>
struct TypeName<RAJA::omp_reduce_ordered> {
  static const char *Get() { return "RAJA::omp_reduce_ordered"; }
};
template<>
struct TypeName<RAJA::omp_reduce> {
  static const char *Get() { return "RAJA::omp_reduce"; }
};
template<>
struct TypeName<RAJA::seq_reduce> {
  static const char *Get() { return "RAJA::seq_reduce"; }
};

void
twoDistinctRandom(RAJA::Index_type &minIdx,
		  RAJA::Index_type &maxIdx,
		  const std::size_t maxIndex)
{
  const RAJA::Index_type maxInd = static_cast<RAJA::Index_type>(maxIndex);
  if (maxInd > 0) {
    minIdx = rand() % maxInd;
    do {
      maxIdx = rand() % maxInd;
    } while (minIdx == maxIdx);
  }
  else {
    minIdx = 0;
    maxIdx = 1;
  }
}





template<typename LoopPolicy, typename LayoutType, typename ReducePolicy>
int
testReductions(const RAJA::Index_type size[3])
{
  int numFailures = 0;
  const int numTests = 16;
  std::size_t memsize = size[0]*size[1]*size[2];
  if (memsize > 2) {
    const std::int64_t expectedSum = static_cast<std::int64_t>(memsize)*2;
    const std::int64_t expectedMin = 1;
    const std::int64_t expectedMax = 3;
    std::int64_t * const memory = new std::int64_t[memsize];
    RAJA::View<std::int64_t, LayoutType> val_view(memory, size[0], size[1], size[2]);
    for(int i = 0; i < numTests; ++i) {
      RAJA::Index_type minIndex, maxIndex;
      RAJA::forallN<LoopPolicy, TimeInd, XInd, YInd >
	(RAJA::RangeSegment(0, size[0]),
	 RAJA::RangeSegment(0, size[1]),
	 RAJA::RangeSegment(0, size[2]),
	 [=](TimeInd t, XInd x, YInd y) {
	  val_view(t, x, y) = 2;
	});
      twoDistinctRandom(minIndex, maxIndex, memsize);
      memory[minIndex] = expectedMin;
      memory[maxIndex] = expectedMax;
      RAJA::ReduceSum<ReducePolicy, std::int64_t> sumr(0);
      RAJA::ReduceMin<ReducePolicy, std::int64_t> minr(INT64_MAX);
      RAJA::ReduceMax<ReducePolicy, std::int64_t> maxr(INT64_MIN);
      RAJA::forallN<LoopPolicy, TimeInd, XInd, YInd >
	(RAJA::RangeSegment(0, size[0]),
	 RAJA::RangeSegment(0, size[1]),
	 RAJA::RangeSegment(0, size[2]),
	 [=](TimeInd t, XInd x, YInd y) {
	  sumr += val_view(t, x, y);
	  minr.min(val_view(t, x, y));
	  maxr.max(val_view(t, x, y));
	});
      const std::int64_t calculatedSum(sumr);
      const std::int64_t calculatedMin(minr);
      const std::int64_t calculatedMax(maxr);
      if (expectedSum != calculatedSum) {
	std::cout << "Error sum reduction does not match expectations: "
		  << calculatedSum << " != " << expectedSum
		  << " (" << TypeName<ReducePolicy>::Get() << ")\n";
	++numFailures;
      }
      if (expectedMin != calculatedMin) {
	std::cout << "Error min reduction does not match expectations: "
		  << calculatedMin << " != "
		  << expectedMin 
		  << " (" << TypeName<ReducePolicy>::Get() << ")\n";
	++numFailures;
      }
      if (expectedMax != calculatedMax) {
	std::cout << "Error max reduction does not match expectations: "
		  << calculatedMax << " != "
		  << expectedMax
		  << " (" << TypeName<ReducePolicy>::Get() << ")\n";
	++numFailures;
      }
    }
  }
  return numFailures;
}

int
main(int argc, char **argv) {
  int numFailures = 0;
  using OMPcollapse = RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_collapse_nowait_exec,
							RAJA::omp_collapse_nowait_exec,
							RAJA::omp_collapse_nowait_exec>,
					 RAJA::OMP_Parallel<> >;
  using OMPtwocollapse = RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_collapse_nowait_exec,
							   RAJA::omp_collapse_nowait_exec,
							   RAJA::seq_exec>,
					    RAJA::OMP_Parallel<> >;
  using OMPouter =  RAJA::NestedPolicy<RAJA::ExecList<RAJA::omp_parallel_for_exec,
						      RAJA::seq_exec,
						      RAJA::seq_exec> >;
  using Sequential = RAJA::NestedPolicy<RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec,RAJA::seq_exec> >;
  using NormalLayout = RAJA::Layout<int, RAJA::PERM_IJK, TimeInd, XInd, YInd>;
  const RAJA::Index_type test_cases[][3] = {
    { 3, 1, 1 },
    { 2, 2, 2 },
    { 4, 4, 4 },
    { 16, 16, 16},
    { 13, 17, 19 }		// all prime
  };
  const std::size_t numCases = sizeof(test_cases)/(3*sizeof(RAJA::Index_type));
  std::cout << "Starting\nMaximum number of threads = "
	    << omp_get_max_threads()
	    << std::endl;
  for(std::size_t i = 0 ; i < numCases; ++i) {
    numFailures += testReductions<OMPcollapse, NormalLayout, RAJA::omp_reduce_ordered>
      (test_cases[i]);
    numFailures += testReductions<OMPcollapse, NormalLayout, RAJA::omp_reduce>
      (test_cases[i]);
    numFailures += testReductions<OMPtwocollapse, NormalLayout, RAJA::omp_reduce_ordered>
      (test_cases[i]);
    numFailures += testReductions<OMPtwocollapse, NormalLayout, RAJA::omp_reduce>
      (test_cases[i]);
    numFailures += testReductions<OMPouter, NormalLayout, RAJA::omp_reduce_ordered>
      (test_cases[i]);
    numFailures += testReductions<OMPouter, NormalLayout, RAJA::omp_reduce>
      (test_cases[i]);
    numFailures += testReductions<Sequential, NormalLayout, RAJA::seq_reduce>
      (test_cases[i]);
  }
  std::cout << "Done\nTotal failures =  "
	    << numFailures << "\n";

  std::cout << RAJA_MAX(INT64_MIN, 1) << std::endl;
  return numFailures;
}
