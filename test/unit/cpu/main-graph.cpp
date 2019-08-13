/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 *
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * For release details and restrictions, please see raja/README-license.txt
 */

//
// Main program illustrating simple RAJA dependence graph creation
// and execution and methods.
//

#include <time.h>
#include <cstdlib>

#include <iostream>
#include <string>
#include <vector>

#include "RAJA/RAJA.hpp"
//#include "RAJA/util/defines.hpp"
#include "RAJA/internal/RAJAVec.hpp"
#include "RAJA/internal/MemUtils_GPU.hpp"

#include "RAJA/index/Graph.hpp"
#include "RAJA/index/GraphBuilder.hpp"

using namespace RAJA;
using namespace std;

///////////////////////////////////////////////////////////////////////////
//
// Main Program.
//
///////////////////////////////////////////////////////////////////////////

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  int starting_vertex = 0;
  int graph_size = 8;
  RAJA::RangeSegment r(starting_vertex,starting_vertex+graph_size);
  RAJA::Graph<RAJA::RangeSegment> g(r);

  //Execution order: indices
  //4343           : 4567
  //1212           : 0123

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

  { //build the graph
    RAJA::GraphBuilder<RAJA::RangeSegment> gb(g);

    std::vector<Index_type> v0_deps;
    v0_deps.push_back(starting_vertex+1);   //vertex 1 depends on vertex 0
    gb.addVertex(starting_vertex+0,v0_deps);

    std::vector<Index_type> v1_deps;
    v1_deps.push_back(starting_vertex+5);   //vertex 5 depends on vertex 1
    gb.addVertex(starting_vertex+1,v1_deps);

    std::vector<Index_type> v2_deps;
    v2_deps.push_back(starting_vertex+3);   //vertex 2 depends on vertex 3
    gb.addVertex(starting_vertex+2,v2_deps);

    std::vector<Index_type> v3_deps;
    v3_deps.push_back(starting_vertex+7);   //vertex 7 depends on vertex 3
    gb.addVertex(starting_vertex+3,v3_deps);

    std::vector<Index_type> v4_deps;        //no one depends on vertex 4
    gb.addVertex(starting_vertex+4,v4_deps);

    std::vector<Index_type> v5_deps;        //vertex 4 depends on vertex 5
    v5_deps.push_back(starting_vertex+4);
    gb.addVertex(starting_vertex+5,v5_deps);

    std::vector<Index_type> v6_deps;        //no one depends on vertex 6
    gb.addVertex(starting_vertex+6,v6_deps);

    std::vector<Index_type> v7_deps;        //vertex 6 depends on vertex 7
    v7_deps.push_back(starting_vertex+6);
    gb.addVertex(starting_vertex+7,v7_deps);

    gb.createDependenceGraph();
  }

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

  std::cout<<"Built graph of size="<<g.size()<<":"<<std::endl;
  g.printGraph(std::cout);
  std::cout<<std::endl;
  std::cout<<"Starting forall on IndexSet containing the graph as one of the segments"<<std::endl;

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

  RAJA::TypedIndexSet<RAJA::GraphRangeSegment> is0;

#if defined(RAJA_ENABLE_CUDA)
 cudaDeviceSynchronize();
#endif

  is0.push_back(g);

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

  using seq_seq_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;
  RAJA::forall<seq_seq_pol>(is0, [=](int i){printf("body(%d)\n",i);});
  std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;"<<std::endl;


#if defined(RAJA_ENABLE_OPENMP)
  std::cout<<std::endl;
  std::cout<<"Starting forall using OpenMP"<<std::endl;

  using seq_omp_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::policy::omp::omp_for_dependence_graph>;
  RAJA::forall<seq_omp_pol>(is0, [=](int i){printf("body(%d)\n",i);});
  std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_dependence_graph>;"<<std::endl;
#endif



#if defined(RAJA_ENABLE_CUDA)
  //std::cout<<std::endl;
  //std::cout<<"Starting forall using CUDA"<<std::endl;

  //using seq_omp_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_dependence_graph>;
  //RAJA::forall<seq_omp_pol>(is0, [=](int i){printf("body(%d)\n",i);});
  //std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_dependence_graph>;"<<std::endl;
#endif

  cout << "\n DONE!!! " << endl;

  return 0;
}
