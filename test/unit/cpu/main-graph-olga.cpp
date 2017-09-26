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
#include "RAJA/util/defines.hpp"
#include "RAJA/internal/RAJAVec.hpp"

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

  int starting_vertex = 2;
  int graph_size = 8;
  RAJA::Graph<unsigned> g(starting_vertex,starting_vertex + graph_size);
  //Color map: indices
  //3434     : 4567
  //1212     : 0123


  { //build the graph
    RAJA::GraphBuilder<unsigned> gb(g);

    std::vector<unsigned> v0_deps;
    v0_deps.push_back(starting_vertex+1);
    v0_deps.push_back(starting_vertex+4);
    v0_deps.push_back(starting_vertex+5);
    gb.addVertex(starting_vertex+0,v0_deps);

    std::vector<unsigned> v1_deps;
    v1_deps.push_back(starting_vertex+0);
    v1_deps.push_back(starting_vertex+2);
    v1_deps.push_back(starting_vertex+4);
    v1_deps.push_back(starting_vertex+5);
    v1_deps.push_back(starting_vertex+6);
    gb.addVertex(starting_vertex+1,v1_deps);

    std::vector<unsigned> v2_deps;
    v2_deps.push_back(starting_vertex+1);
    v2_deps.push_back(starting_vertex+3);
    v2_deps.push_back(starting_vertex+5);
    v2_deps.push_back(starting_vertex+6);
    v2_deps.push_back(starting_vertex+7);
    gb.addVertex(starting_vertex+2,v2_deps);

    std::vector<unsigned> v3_deps;
    v3_deps.push_back(starting_vertex+2);
    v3_deps.push_back(starting_vertex+6);
    v3_deps.push_back(starting_vertex+7);
    gb.addVertex(starting_vertex+3,v3_deps);

    std::vector<unsigned> v4_deps;
    v4_deps.push_back(starting_vertex+0);
    v4_deps.push_back(starting_vertex+1);
    v4_deps.push_back(starting_vertex+5);
    gb.addVertex(starting_vertex+4,v4_deps);

    std::vector<unsigned> v5_deps;
    v5_deps.push_back(starting_vertex+0);
    v5_deps.push_back(starting_vertex+1);
    v5_deps.push_back(starting_vertex+2);
    v5_deps.push_back(starting_vertex+4);
    v5_deps.push_back(starting_vertex+6);
    gb.addVertex(starting_vertex+5,v5_deps);

    std::vector<unsigned> v6_deps;
    v6_deps.push_back(starting_vertex+1);
    v6_deps.push_back(starting_vertex+2);
    v6_deps.push_back(starting_vertex+3);
    v6_deps.push_back(starting_vertex+5);
    v6_deps.push_back(starting_vertex+7);
    gb.addVertex(starting_vertex+6,v6_deps);

    std::vector<unsigned> v7_deps;
    v7_deps.push_back(starting_vertex+2);
    v7_deps.push_back(starting_vertex+3);
    v7_deps.push_back(starting_vertex+6);
    gb.addVertex(starting_vertex+7,v7_deps);

  }//when GraphBuilder is destroyed, the graph is finalized


  std::cout<<"Built graph:"<<std::endl;
  g.printGraph(std::cout);

  std::cout<<std::endl;
  g.satisfyDependents(starting_vertex+3);

  std::cout<<"Satisfied dependencies of vertex 3:"<<std::endl;
  g.printGraph(std::cout);










/*
  RAJA::IndexSet<RAJA::RangeSegment, RAJA::ListSegment> is0;

  std::vector<int> segvec{3,2,1};
  is0.push_back(RAJA::ListSegment(segvec));

  is0.push_back(RAJA::RangeSegment(0,5));
  is0.push_back(RAJA::RangeSegment(-5,0));

  cout << "\n\nIndexSet( master ) " << endl;
  is0.print(cout);


  RAJA::IndexSet<RAJA::RangeSegment, RAJA::ListSegment> is1;
  int num_segments = is0.getNumSegments();

  for (int i = 0; i < num_segments; ++i) {
    is0.segment_push_into(i, is1, PUSH_BACK, PUSH_COPY);
  }

  cout << "\n\nIndexSet( copy ) " << endl;
  is1.print(cout);


  if (is1 == is0) { } //std::cout<<"first copy correct"<<std::endl; }
  else { std::cout<<"Copy not correct"<<std::endl; }

  if (is1 != is0) { std::cout<<"!= operator is not correct"<<std::endl; }

  int slice_size = num_segments - 1;
  std::cout<<"creating a slice of size "<<slice_size;//<<std::endl;
  RAJA::IndexSet<RAJA::RangeSegment, RAJA::ListSegment>* iset_slice
    = is1.createSlice(0, slice_size);

  cout << "\n\nIndexSet( slice ) " << endl;
  iset_slice->print(cout);

  using seq_seq_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;
  RAJA::forall<seq_seq_pol>(is0, [=](int i){printf("body(%d)\n",i);});
  std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;"<<std::endl;

#ifdef RAJA_ENABLE_OPENMP
  using omp_seq_pol = RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::seq_exec>;
  RAJA::forall<omp_seq_pol>(is0, [=](int i){printf("body(%d)\n",i);});
  std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::omp_parallel_for_segit, RAJA::seq_exec>;"<<std::endl;
#endif

*/

  cout << "\n DONE!!! " << endl;

  return 0;
}
