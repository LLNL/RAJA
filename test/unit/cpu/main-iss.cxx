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
// Main program illustrating simple RAJA index set creation
// and execution and methods.
//

#include <time.h>
#include <cstdlib>

#include <iostream>
#include <string>

#include "RAJA/RAJA.hxx"
#include "RAJA/internal/RAJAVec.hxx"
#include "RAJA/internal/defines.hxx"

using namespace RAJA;
using namespace std;

struct Dumper {

  template<typename T>
  RAJA_INLINE
  void operator()(T const &foo) const {
    std::cout << "MEOWMEOW: " << foo << std::endl;
  }

};

///////////////////////////////////////////////////////////////////////////
//
// Main Program.
//
///////////////////////////////////////////////////////////////////////////

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv))
{
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

  cout << "\n DONE!!! " << endl;

  return 0;
}
