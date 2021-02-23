//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <math.h> 
#include <typeinfo>
#include <cxxabi.h>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"
#include "array-reductions2.hpp"

//#ifdef HAVE_CXA_DEMANGLE
#if 1
const char* demangle(const char* name)
{
     char buf[1024];
     size_t size=1024;
     int status;
     char* res = abi::__cxa_demangle (name,
                                      buf,
                                      &size,
                                      &status);
     return res;
}
#else
const char* demangle(const char* name)
{
    return name;
}
#endif

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
  //using REDUCE_POL = RAJA::omp_reduce;

  using BASE_T = double;
  //using REDUCESUM_T = RAJA::ReduceSum<REDUCE_POL, BASE_T>;

#if 0
  // --------------------------------------------------------------------------------
  // Run 1D Reduction Sum Test
  // --------------------------------------------------------------------------------
{
  std::cout << "\n1D Constructor Test...\n";
  ContainerReducer<double, 1> data(1.0, NUM_NODES);
  data.print();
}

{
  std::cout << "\n3D Constructor Test...\n";
  ContainerReducer<double, 3> data(2.0, 4, 3, 2);
  data.print();
}

{
  std::cout << "\n5D Constructor Test...\n";
  ContainerReducer<double, 5> data(5.0, 4, 3, 2, 1, 7);
  data.print();
}
#endif

#if 1
{
  // --------------------------------------------------------------------------------
  // Run 1D Reduction Sum Test
  // --------------------------------------------------------------------------------
  std::cout << "\nRunning 1D Reducer Sum Test...\n";

  ContainerReducer<BASE_T, 1> r_nodeData(0, 7);

  RAJA::ChronoTimer timer;
  timer.start();
  
  RAJA::forall<EXEC_POL> (np_range, [=](int) {
    r_nodeData[2] += 5;
  });

  timer.stop();
  r_nodeData.print();
}
#endif

#if 1
{
  // --------------------------------------------------------------------------------
  // Run 2D Reduction Sum Test
  // --------------------------------------------------------------------------------
  std::cout << "\nRunning 2D Reducer Sum Test...\n";

  ContainerReducer<BASE_T, 2> r_nodeData(0, 4,3);

  RAJA::ChronoTimer timer;
  timer.start();
  
  RAJA::forall<EXEC_POL> (np_range, [=](int) {
    r_nodeData[2][2] += 5;
  });

  timer.stop();
  r_nodeData.print();
}
#endif

#if 1
{
  // --------------------------------------------------------------------------------
  // Run 1D Reduction Performance/Correctness Test
  // --------------------------------------------------------------------------------
  std::cout << "\nRunning 1D Reducer Performance/Correctness test ...\n";

  ContainerReducer<BASE_T, 1> r_nodeData(0, NUM_NODES);

  RAJA::ChronoTimer timer;
  timer.start();
  RAJA::forall<EXEC_POL> (np_range, [=](int i) {
    int i_idx = pairlist[ i ].first;
    int j_idx = pairlist[ i ].second;

    BASE_T& i_data = r_nodeData[ i_idx ];
    BASE_T& j_data = r_nodeData[ j_idx ];
    i_data += j_idx;
    j_data += i_idx;

  });
  timer.stop();
  checkResults(nodeDataSolution, r_nodeData, timer);
}
#endif

#if 1
{
  // --------------------------------------------------------------------------------
  // Run 2D Reduction Performance/Correctness Test
  // --------------------------------------------------------------------------------
  std::cout << "\nRunning 2D Reducer Performance/Correctness test ...\n";

  auto nodeDataSolution2d = generate2DSolution(NUM_NODE_LISTS, NUM_NODES, pairlist2d);
  //VoT_t<2, BASE_T> VoT_nodeData2(NUM_NODE_LISTS, NUM_NODES);
  
  ContainerReducer<BASE_T, 2> r_nodeData(0, NUM_NODE_LISTS, NUM_NODES);

  RAJA::ChronoTimer timer;
  timer.start();
  RAJA::forall<EXEC_POL> (np_range, [=](int i) {
    int i_idx = pairlist2d[ i ].first;
    int j_idx = pairlist2d[ i ].second;
    BASE_T& _data = r_nodeData[ i_idx ][ j_idx ];
    _data += i_idx + j_idx;
  });
  timer.stop();

  checkResults(nodeDataSolution2d, r_nodeData, timer);
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
