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
  constexpr int NUM_NODES = 5000;
  int NUM_NODE_LISTS = 2;
  int NUM_PAIRS = 50000;
  if (argc > 1)
    NUM_PAIRS = std::atoi(argv[1]);

  RAJA::RangeSegment np_range(0, NUM_PAIRS);

  // Initialize NodePair list
  auto pairlist = generatePairList(NUM_NODES, NUM_PAIRS);
  auto pairlist2d = generate2DPairList(NUM_NODES, NUM_NODE_LISTS, NUM_PAIRS);

  // Calculate Solution
  auto nodeDataSolution = generateSolution(NUM_NODES, pairlist);

  //using EXEC_POL = RAJA::seq_exec;
  //using REDUCE_POL = RAJA::seq_reduce;
  using EXEC_POL = RAJA::omp_parallel_for_exec;
  using REDUCE_POL = RAJA::omp_reduce;

  using BASE_T = double;
  using REDUCESUM_T = RAJA::ReduceSum<REDUCE_POL, BASE_T>;

#if 1
  // --------------------------------------------------------------------------------
  // Run Vector-of-Type Reduction Example
  // --------------------------------------------------------------------------------
  std::cout << "\nRunning 1D Vector-of-Type (VoT_t) Reducer example ...\n";

  VoT_t<1, BASE_T> VoT_nodeData(NUM_NODES);

  RAJA::ChronoTimer type_timer;
  type_timer.start();
  RAJA::forall<EXEC_POL> (np_range, [=](int i) {
    int i_idx = pairlist[ i ].first;
    int j_idx = pairlist[ i ].second;

    BASE_T& i_data = VoT_nodeData.at( i_idx );
    BASE_T& j_data = VoT_nodeData.at( j_idx );
    i_data += j_idx;
    j_data += i_idx;
  });
  type_timer.stop();
  checkResults(nodeDataSolution, VoT_nodeData, type_timer);
#endif

#if 0
  // --------------------------------------------------------------------------------
  // Run 2D Col-Major Vector-of-Type Reduction Example
  // --------------------------------------------------------------------------------
  std::cout << "\nRunning 2D Col-Major Vector-of-Type (VoT_t) Reducer example ...\n";

  VoT_t<2, BASE_T> VoT_nodeData2c(1, NUM_NODES);

  RAJA::ChronoTimer type_timer2c;
  type_timer2c.start();
  RAJA::forall<EXEC_POL> (np_range, [=](int i) {
    int i_idx = pairlist[ i ].first;
    int j_idx = pairlist[ i ].second;

    BASE_T& i_data = VoT_nodeData2c.at( 0, i_idx );
    BASE_T& j_data = VoT_nodeData2c.at( 0, j_idx );
    i_data += j_idx;
    j_data += i_idx;
  });
  type_timer2c.stop();
  checkResults(nodeDataSolution, VoT_nodeData2c, type_timer2c);
#endif

#if 0
  {
    // --------------------------------------------------------------------------------
    // Run 2D Row-Major Vector-of-Type Reduction Example
    // --------------------------------------------------------------------------------
    std::cout << "\nRunning 2D Row-Major Vector-of-Type (VoT_t) Reducer example ...\n";

    auto nodeDataSolution2d = generate2DSolution(1, NUM_NODES, pairlist2d);
    VoT_t<2, BASE_T> VoT_nodeData2r(1, NUM_NODES);

    RAJA::ChronoTimer type_timer2;
    type_timer2.start();
    RAJA::forall<EXEC_POL> (np_range, [=](int i) {
      int i_idx = pairlist2d[ i ].first;
      int j_idx = pairlist2d[ i ].second;
      BASE_T& _data = VoT_nodeData2r.at( i_idx , j_idx );
      _data += i_idx + j_idx;
    });
    type_timer2.stop();
    checkResults(nodeDataSolution2d, VoT_nodeData2r, type_timer2);
  }
#endif

#if 1
  // --------------------------------------------------------------------------------
  // Run Vector-of-Reducer Reduction Example
  // --------------------------------------------------------------------------------
  std::cout << "\nRunning Vector-of-Reducer (VoR_t) Reducer example ...\n";

  VoR_t<REDUCESUM_T> VoR_nodeData(NUM_NODES);

  RAJA::ChronoTimer reduce_timer;
  reduce_timer.start();
  RAJA::forall<EXEC_POL> (np_range, [=](int i) {
    int i_idx = pairlist[ i ].first;
    int j_idx = pairlist[ i ].second;

    REDUCESUM_T& i_data = VoR_nodeData.at( i_idx );
    REDUCESUM_T& j_data = VoR_nodeData.at( j_idx );
    i_data += j_idx;
    j_data += i_idx;
  });
  reduce_timer.stop();
  checkResults(nodeDataSolution, VoR_nodeData, reduce_timer);
#endif

#if 0
  {
    VoT_t<2, BASE_T> VoT_nodeData2(NUM_NODE_LISTS, NUM_NODES);

    auto nodeDataSolution2d = generate2DSolution(NUM_NODE_LISTS, NUM_NODES, pairlist2d);

    std::cout << "check\n";
    RAJA::ChronoTimer type_timer2d;
    type_timer2d.start();
    RAJA::forall<EXEC_POL> (np_range, [=](int i) {
      int i_idx = pairlist2d[ i ].first;
      int j_idx = pairlist2d[ i ].second;
      BASE_T& _data = VoT_nodeData2.at( i_idx , j_idx );
      _data += i_idx + j_idx;
    });
    type_timer2d.stop();
    checkResults(nodeDataSolution2d, VoT_nodeData2, type_timer2d);
  }
#endif
}




// --------------------------------------------------------------------------------
// Helper Function Definitions
// --------------------------------------------------------------------------------

pairlist_t generatePairList(const int n_nodes, const int n_pairs){
  srand(0);
  pairlist_t pl;
  for (auto i = 0; i < n_pairs; i++)
    pl.push_back(std::make_pair(rand() % n_nodes, rand() % n_nodes));
  return pl;
}

pairlist_t generate2DPairList(const int n_nodes, const int n_node_lists, const int n_pairs){
  srand(0);
  pairlist_t pl;
  for (auto i = 0; i < n_pairs; i++)
    pl.push_back(std::make_pair(rand() % n_node_lists, rand() % n_nodes));
  return pl;
}

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

std::vector<std::vector<double>> generate2DSolution(const int n_node_lists, const int n_nodes, const pairlist_t pl){
  std::vector<std::vector<double>> solution;
  for (int i = 0; i < n_node_lists; i++) {
    solution.push_back(std::vector<double>(n_nodes));
  }

  for (size_t i = 0; i < pl.size(); i++){
    int i_idx = pl[ i ].first;
    int j_idx = pl[ i ].second;
    double& _data = solution[i_idx][j_idx];
    _data += i_idx + j_idx;
  }
  return solution;
}

template<typename T1, typename T2>
void checkResults(const  T1& solution, const T2& test, const RAJA::ChronoTimer& timer){

  std::cout << "\tTime : " << timer.elapsed() << "\n";

  //if (correctness)
  if (test == solution)
    std::cout<< "\tPASSED !\n";
  else
    std::cout<< "\tFAILED !\n";
}
