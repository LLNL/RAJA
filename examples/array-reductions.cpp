//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <math.h> 

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"
#include "array-reductions.hpp"

int main(int argc, char* argv[])
{
  // Test Parameters
  constexpr int NUM_NODES = 50000;
  int NUM_PAIRS = 50000;
  if (argc > 1)
    NUM_PAIRS = std::atoi(argv[1]);

  RAJA::RangeSegment np_range(0, NUM_PAIRS);

  // Initialize NodePair list
  auto pairlist = generatePairList(NUM_NODES, NUM_PAIRS);

  // Calculate Solution
  auto nodeDataSolution = generateSolution(NUM_NODES, pairlist);

  //using EXEC_POL = RAJA::seq_exec;
  //using REDUCE_POL = RAJA::seq_reduce;
  using EXEC_POL = RAJA::omp_parallel_for_exec;
  using REDUCE_POL = RAJA::omp_reduce;


  // --------------------------------------------------------------------------------
  // Run Vector-of-Type Reduction Example
  // --------------------------------------------------------------------------------
  std::cout << "\nRunning Vector-of-Type (VoT_t) Reducer example ...\n";

  using BASE_T = double;
  VoT_t<BASE_T> VoT_nodeData(NUM_NODES);

  RAJA::ChronoTimer type_timer;
  type_timer.start();
  RAJA::forall<EXEC_POL> (np_range, [=](int i) {
    int i_idx = pairlist[ i ].first;
    int j_idx = pairlist[ i ].second;

    BASE_T& i_data = VoT_nodeData[ i_idx ];
    BASE_T& j_data = VoT_nodeData[ j_idx ];
    i_data += j_idx;
    j_data += i_idx;
  });
  type_timer.stop();
  checkResults(nodeDataSolution, VoT_nodeData, type_timer);


#if 1
  // --------------------------------------------------------------------------------
  // Run Vector-of-Reducer Reduction Example
  // --------------------------------------------------------------------------------
  std::cout << "\nRunning Vector-of-Reducer (VoR_t) Reducer example ...\n";

  using REDUCESUM_T = RAJA::ReduceSum<REDUCE_POL, BASE_T>;
  VoR_t<REDUCESUM_T> VoR_nodeData(NUM_NODES);

  RAJA::ChronoTimer reduce_timer;
  reduce_timer.start();
  RAJA::forall<EXEC_POL> (np_range, [=](int i) {
    int i_idx = pairlist[ i ].first;
    int j_idx = pairlist[ i ].second;

    REDUCESUM_T& i_data = VoR_nodeData[ i_idx ];
    REDUCESUM_T& j_data = VoR_nodeData[ j_idx ];
    i_data += j_idx;
    j_data += i_idx;
  });
  reduce_timer.stop();
  checkResults(nodeDataSolution, VoR_nodeData, reduce_timer);
#endif


}




// --------------------------------------------------------------------------------
// Helper Function Definitions
// --------------------------------------------------------------------------------

std::vector<double> generateSolution(const int n_nodes, const pairlist_t pl){
  std::vector<double> solution(n_nodes);
  for (size_t i = 0; i < pl.size(); i++){
    int i_idx = pl[ i ].first;
    int j_idx = pl[ i ].second;
    double& i_data = solution[ i_idx ];
    double& j_data = solution[ j_idx ];
    i_data += j_idx;
    j_data += i_idx;
  }
  return solution;
}

pairlist_t generatePairList(const int n_nodes, const int n_pairs){
  srand(0);
  pairlist_t pl;
  for (auto i = 0; i < n_pairs; i++)
    pl.push_back(std::make_pair(rand() % n_nodes, rand() % n_nodes));
  return pl;
}

template<typename T1, typename T2>
void checkResults(const  T1& solution, const T2& test, const RAJA::ChronoTimer& timer){

  bool correctness = true;
  for (size_t i = 0; i < solution.size(); i++){
    if (solution[i] != test[i]) correctness = false;
  }

  //size_t print_sz = solution.size() < 10 ? solution.size() : 10;
  //for (size_t i = 0; i < print_sz; i++){
  //  std::cout << "(" << solution[i] << ", " << test[i] << ")  ";
  //}
  //std::cout << "\n";

  std::cout << "\tTime : " << timer.elapsed() << "\n";

  if (correctness)
    std::cout<< "\tPASSED !\n";
  else
    std::cout<< "\tFAILED !\n";
}
