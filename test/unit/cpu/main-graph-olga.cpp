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
#include "RAJA/internal/MemUtils_GPU.hpp"

#include "RAJA/index/Graph.hpp"
#include "RAJA/index/GraphBuilder.hpp"

using namespace RAJA;
using namespace std;

static RAJA::Real_ptr ref_array, test_array;

///////////////////////////////////////////////////////////////////////////
//
// Main Program.
//
///////////////////////////////////////////////////////////////////////////

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
#if defined(RAJA_ENABLE_CUDA)

  {
    RAJA::RAJAVec<int, managed_allocator<int> > myvec;
    myvec.push_back(3);
    int myvar = myvec[0];
    std::cout<<"myvar="<<myvar<<std::endl;
    myvec[0] = 4;
    int myvar0 = myvec[0];
    std::cout<<"myvar0="<<myvar0<<std::endl;
  }
#endif

  int starting_vertex = 0; //2;
  int graph_size = 8;
  RAJA::RangeSegment r(starting_vertex,starting_vertex+graph_size);
  RAJA::Graph<RangeSegment> g(r);
  //Color map: indices - size 4
  //32     : 23
  //01     : 01

  //Color map: indices - size 8
  //4343     : 4567
  //1212     : 0123

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

  { //build the graph
    RAJA::GraphBuilder<RangeSegment> gb(g);

/*
    std::vector<Index_type> v0_deps;
    v0_deps.push_back(starting_vertex+1);
    gb.addVertex(starting_vertex+0,v0_deps);

    std::vector<Index_type> v1_deps;
    v1_deps.push_back(starting_vertex+3);
    gb.addVertex(starting_vertex+1,v1_deps);

    std::vector<Index_type> v2_deps;
    gb.addVertex(starting_vertex+2,v2_deps);

    std::vector<Index_type> v3_deps;
    v3_deps.push_back(starting_vertex+2);
    gb.addVertex(starting_vertex+3,v3_deps);
*/


    std::vector<Index_type> v0_deps;
    v0_deps.push_back(starting_vertex+1);
    gb.addVertex(starting_vertex+0,v0_deps);

    std::vector<Index_type> v1_deps;
    v1_deps.push_back(starting_vertex+5);
    gb.addVertex(starting_vertex+1,v1_deps);

    std::vector<Index_type> v2_deps;
    v2_deps.push_back(starting_vertex+3);
    gb.addVertex(starting_vertex+2,v2_deps);

    std::vector<Index_type> v3_deps;
    v3_deps.push_back(starting_vertex+7);
    gb.addVertex(starting_vertex+3,v3_deps);

    std::vector<Index_type> v4_deps;
    gb.addVertex(starting_vertex+4,v4_deps);

    std::vector<Index_type> v5_deps;
    v5_deps.push_back(starting_vertex+4);
    gb.addVertex(starting_vertex+5,v5_deps);

    std::vector<Index_type> v6_deps;
    gb.addVertex(starting_vertex+6,v6_deps);

    std::vector<Index_type> v7_deps;
    v7_deps.push_back(starting_vertex+6);
    gb.addVertex(starting_vertex+7,v7_deps);

    gb.createDependenceGraph();
  }

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

  std::cout<<"Built graph of size="<<g.size()<<":"<<std::endl;
  g.printGraph(std::cout);
/*
  std::cout<<std::endl;
  g.satisfyDependents(starting_vertex+3);

  std::cout<<"Satisfied dependencies of vertex 3:"<<std::endl;
  g.printGraph(std::cout);
*/

  std::cout<<std::endl;
  std::cout<<"Starting forall on IndexSet containing the graph as one of the segments"<<std::endl;

  // Insert the Graph/Range segment into an IndexSet

  //RAJA::StaticIndexSet<RAJA::RangeSegment, RAJA::ListSegment> is0;

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

  //RAJA::StaticIndexSet<RAJA::RangeSegment, RAJA::GraphRangeSegment> is0; //OLGA FIX ME - should be able to have RangeSegment also
  RAJA::StaticIndexSet<RAJA::GraphRangeSegment> is0;
  //RAJA::IndexSet<RAJA::GraphRangeSegment> is0;

#if defined(RAJA_ENABLE_CUDA)
 cudaDeviceSynchronize();
#endif

//  is0.push_back(RAJA::RangeSegment(0,5));  //OLGA FIX ME - should be able to add this back in
 is0.push_back(g);

#if defined(RAJA_ENABLE_CUDA)
  cudaDeviceSynchronize();
#endif

/*
  using seq_seq_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;
  RAJA::forall<seq_seq_pol>(is0, [=](int i){printf("body(%d)\n",i);});
  std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;"<<std::endl;


#if defined(RAJA_ENABLE_OPENMP)
  std::cout<<std::endl;
  std::cout<<"Starting forall using OpenMP"<<std::endl;

  using seq_omp_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_dependence_graph>;
  RAJA::forall<seq_omp_pol>(is0, [=](int i){printf("body(%d)\n",i);});
  std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_dependence_graph>;"<<std::endl;
#endif

*/

#if defined(RAJA_ENABLE_CUDA)
  //std::cout<<std::endl;
  //std::cout<<"Starting forall using CUDA"<<std::endl;

  //using seq_omp_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_dependence_graph>;
  //RAJA::forall<seq_omp_pol>(is0, [=](int i){printf("body(%d)\n",i);});
  //std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_dependence_graph>;"<<std::endl;
#endif



  std::cout<<std::endl;
  std::cout<<"Starting forall using CUDA"<<std::endl;

  //make a test for IndexSet on GPU
  RAJA::IndexSet is1;

  const int num_segments = 32;
  for (int i=0; i<num_segments; i++) {
    is1.push_back(RAJA::RangeSegment(i*2,i*2+2));
  }

  std::cout<<"pushed all segments onto the index set"<<std::endl;

  int max_size = num_segments * 2;

#if defined(RAJA_ENABLE_CUDA)
  RAJA::Real_ptr test_array = ::test_array;
  RAJA::Real_ptr ref_array = ::ref_array;
  cudaMallocManaged((void **)&test_array,
                    sizeof(Real_type) * max_size,
                    cudaMemAttachGlobal);

  cudaMallocManaged((void **)&ref_array,
                    sizeof(Real_type) * max_size,
                    cudaMemAttachGlobal);

  cudaMemset(test_array, 0, sizeof(RAJA::Real_type) * num_segments*2);
  cudaMemset(ref_array, 0, sizeof(RAJA::Real_type) * num_segments*2);

  //int myvar2 = test_array[0];   //CRASHES!!!!
  //std::cout<<"myvar2="<<myvar2<<std::endl;

  RAJA::RAJAVec<int, managed_allocator<int> > test_array2;
  for (int i=0; i<num_segments*2; i++) {
    test_array2.push_back(0);
  }

  int myvar3 = test_array2[0];
  std::cout<<"myvar3="<<myvar3<<std::endl;

  using cuda_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::cuda_exec<num_segments>>;
  //using cuda_pol = RAJA::ExecPolicy<RAJA::cuda_exec<num_segments>, RAJA::seq_segit>;

  RAJA::forall<cuda_pol>(
    is1, [=] __device__ (RAJA::Index_type idx) {
      test_array[idx] = idx;
      //int i=0;
      //i++;
    });


  cudaDeviceSynchronize();

  std::cout<<"answer array: ";
  for (int i=0; i<max_size; i++) {
    std::cout<<test_array[i]<<" ";
  } std::cout<<std::endl;


  std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::seq_exec>;"<<std::endl;

  cudaFree(::test_array);
  cudaFree(::ref_array);
  cudaDeviceSynchronize();

#endif


/*
  std::cout<<std::endl;
  std::cout<<"Starting forall using OpenMP with dependencies"<<std::endl;

  using seq_ompdep_pol = RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_dependence_graph>;
  RAJA::forall<seq_ompdep_pol>(is0, [=](int i){printf("body(%d)\n",i);});
  std::cout<<"finished forall with RAJA::ExecPolicy<RAJA::seq_segit, RAJA::omp_for_dependence_graph>;"<<std::endl;
*/


  //TO DO:
  //figure out which forall this is going through and how to use the dependence graph there
  //put in the implementaiton without atomics
  //try to put in the atomics back




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
