//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
// and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
//
// SPDX-License-Identifier: (BSD-3-Clause)
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <cstring>
#include <iostream>

#include "RAJA/RAJA.hpp"

/*
 *  Reproducer for...
 *
 */

//
// Functions for checking and printing arrays used for all cases
//
void checkResult(int* c, int* c_ref, int len); 
void printArray(int* v, int len);


#if defined(RAJA_ENABLE_TARGET_OPENMP)
//
// For OpenMP target variants
//
const size_t threads_per_team = 256;

template <typename T>
void initOpenMPDeviceData(T& dptr, const T hptr, int len, 
                          int did, int hid)
{
  omp_target_memcpy( dptr, hptr, 
                     len * sizeof(typename std::remove_pointer<T>::type),
                     0, 0, did, hid ); 
}

template <typename T>
void allocAndInitOpenMPDeviceData(T& dptr, const T hptr, int len,
                                  int did, int hid)
{
  dptr = static_cast<T>( omp_target_alloc(
                         len * sizeof(typename std::remove_pointer<T>::type), 
                         did) );

  initOpenMPDeviceData(dptr, hptr, len, did, hid);
}

template <typename T>
void getOpenMPDeviceData(T& hptr, const T dptr, int len, int hid, int did)
{
  omp_target_memcpy( hptr, dptr, 
                     len * sizeof(typename std::remove_pointer<T>::type),
                     0, 0, hid, did );
}

template <typename T>
void deallocOpenMPDeviceData(T& dptr, int did)
{
  omp_target_free( dptr, did );
  dptr = 0;
}
#endif


int main(int RAJA_UNUSED_ARG(argc), char **RAJA_UNUSED_ARG(argv[]))
{

///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////
  std::cout << "\n\n1D case using RAJA::forall...\n";

//
// Define vector length
//
  constexpr int N = 100;

//
// Allocate and initialize vector data.
//
  int* a = new int[N];
  int* b = new int[N];

  for (int i = 0; i < N; ++i) {
    a[i] = i;  
    b[i] = i+1;  
  }

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  // Pointers for device-allocated data
  int* da;
  int* db;
  int* dc;
#endif

  // Host allocations for variant results
  int* c_ref = new int[N];
  int* c_raja_seq = new int[N];
  int* c_raja_omp = new int[N];
  int* c_base_omptarget = new int[N];
  int* c_raja_omptarget = new int[N];

//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//
  std::memset(c_ref, 0, N * sizeof(int));

  std::cout << "\n Running C-style sequential vector addition...\n";

  for (int i = 0; i < N; ++i) {
    c_ref[i] = a[i] + b[i];
  }

//printArray(c_ref, N);


//----------------------------------------------------------------------------//
// RAJA::seq_exec variant
//----------------------------------------------------------------------------//
  std::memset(c_raja_seq, 0, N * sizeof(int));

  std::cout << "\n Running RAJA sequential vector addition...\n";

  RAJA::forall< RAJA::seq_exec >(RAJA::RangeSegment(0, N), 
  [=] (int i) { 
    c_raja_seq[i] = a[i] + b[i]; 
  });    

  checkResult(c_raja_seq, c_ref, N);
//printArray(c_raja_seq, N);


//----------------------------------------------------------------------------//
// RAJA::omp_parallel_for_exec variant
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c_raja_omp, 0, N * sizeof(int));

  std::cout << "\n Running RAJA OpenMP multithreaded vector addition...\n";

  RAJA::forall< RAJA::omp_parallel_for_exec >(RAJA::RangeSegment(0, N), 
  [=] (int i) { 
    c_raja_omp[i] = a[i] + b[i]; 
  });    

  checkResult(c_raja_omp, c_ref, N);
//printArray(c_raja_omp, N);

#endif


//----------------------------------------------------------------------------//
// OpenMP target variants
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_TARGET_OPENMP)

  int hid = omp_get_initial_device();
  int did = omp_get_default_device();

  allocAndInitOpenMPDeviceData(da, a, N, did, hid);
  allocAndInitOpenMPDeviceData(db, b, N, did, hid);


//----------------------------------------------------------------------------//
// Base OpenMP target variant
//----------------------------------------------------------------------------//

  std::memset(c_base_omptarget, 0, N * sizeof(int));
  allocAndInitOpenMPDeviceData(dc, c_base_omptarget, N, did, hid);

  std::cout << "\n Running base OpenMP target vector addition...\n";

  #pragma omp target is_device_ptr(da, db, dc) device( did )
  #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1)
  for (int i = 0; i < N; ++i ) {
    dc[i] = da[i] + db[i];
  }

  getOpenMPDeviceData(c_base_omptarget, dc, N, hid, did);

  checkResult(c_base_omptarget, c_ref, N);
//printArray(c_base_omptarget, N);


//----------------------------------------------------------------------------//
// RAJA OpenMP target variant
//----------------------------------------------------------------------------//

  std::memset(c_raja_omptarget, 0, N * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget, N, did, hid);

  std::cout << "\n Running RAJA OpenMP target vector addition...\n";

  RAJA::forall<RAJA::omp_target_parallel_for_exec<threads_per_team>>(
  RAJA::RangeSegment(0, N), [=](int i) {
    dc[i] = da[i] + db[i];
  });

  getOpenMPDeviceData(c_raja_omptarget, dc, N, hid, did);

  checkResult(c_raja_omptarget, c_ref, N);
//printArray(c_raja_omptarget, N);

#endif

// cleanup

  delete [] a;
  delete [] b;

  delete [] c_ref;
  delete [] c_raja_seq;
  delete [] c_raja_omp;
  delete [] c_base_omptarget;
  delete [] c_raja_omptarget;

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  deallocOpenMPDeviceData(da, did);
  deallocOpenMPDeviceData(db, did);
  deallocOpenMPDeviceData(dc, did);
#endif

  std::cout << "\n 1D case DONE!...\n";
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

  std::cout << "\n\n2D case using RAJA::kernel...\n";

//
// Define matrix extents
//
  constexpr int Nj = 40;
  constexpr int Ni = 100;
  constexpr int N_2D = Nj * Ni;

  // Host allocations for variant results
  c_ref = new int[N_2D];
  int* c_base_omp = new int[N_2D];
  c_raja_omp = new int[N_2D];
  int* c_raja_omp_no_collapse = new int[N_2D];
  c_base_omptarget = new int[N_2D];
  c_raja_omptarget = new int[N_2D];
  int* c_raja_omptarget_no_collapse = new int[N_2D];

//----------------------------------------------------------------------------//
// C-style sequential variant establishes reference solution to compare with.
//----------------------------------------------------------------------------//

  std::memset(c_ref, 0, N_2D * sizeof(int));

  std::cout << "\n Running C-style matrix init...\n";

  for (int j = 0; j < Nj; ++j) {
    for (int i = 0; i < Ni; ++i) {
      c_ref[i + Ni*j] = i + Ni*j;
    }
  }

//printArray(c_ref, N_2D);


//----------------------------------------------------------------------------//
// Base OpenMP variant
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c_base_omp, 0, N_2D * sizeof(int));

  std::cout << "\n Running base OpenMP matrix init...\n";

  #pragma omp parallel for collapse(2)
  for (int j = 0; j < Nj; ++j) {
    for (int i = 0; i < Ni; ++i) {
      c_base_omp[i + Ni*j] = i + Ni*j;
    }
  }

  checkResult(c_base_omp, c_ref, N_2D);
//printArray(c_base_omp, N_2D);

#endif


//----------------------------------------------------------------------------//
// RAJA OpenMP variants
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c_raja_omp, 0, N_2D * sizeof(int));

  std::cout << "\n Running RAJA OpenMP collapse matrix init...\n";

  using EXEC_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<0, 1>,
        RAJA::statement::Lambda<0>
      >
    >;

  RAJA::kernel<EXEC_POL1>( 
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {  
      c_raja_omp[i + Ni*j] = i + Ni*j;
  });

  checkResult(c_raja_omp, c_ref, N_2D);
//printArray(c_raja_omp, N_2D);

#endif

//////////////////////////////////////////////

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c_raja_omp, 0, N_2D * sizeof(int));

  std::cout << "\n Running RAJA OpenMP collapse (Segs) matrix init...\n";

  using EXEC_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_parallel_collapse_exec,
                                RAJA::ArgList<0, 1>,
        RAJA::statement::Lambda<0, RAJA::Segs<0, 1>>
      >
    >;

  RAJA::kernel<EXEC_POL2>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      c_raja_omp[i + Ni*j] = i + Ni*j;
  });

  checkResult(c_raja_omp, c_ref, N_2D);
//printArray(c_raja_omp, N_2D);

#endif

//////////////////////////////////////////////

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c_raja_omp_no_collapse, 0, N_2D * sizeof(int));
  
  std::cout << "\n Running RAJA OpenMP matrix init (no collapse)...\n";

  using EXEC_POL3 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<1, RAJA::loop_exec,
          RAJA::statement::Lambda<0>
        >
      >
    >;

  RAJA::kernel<EXEC_POL3>(     
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      c_raja_omp_no_collapse[i + Ni*j] = i + Ni*j;
  });

  checkResult(c_raja_omp_no_collapse, c_ref, N_2D);
//printArray(c_raja_omp_no_collapse, N_2D);

#endif

//////////////////////////////////////////////

#if defined(RAJA_ENABLE_OPENMP)

  std::memset(c_raja_omp_no_collapse, 0, N_2D * sizeof(int));

  std::cout << "\n Running RAJA OpenMP matrix init (no collapse, segs)...\n";

  using EXEC_POL4 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<1, RAJA::loop_exec,
          RAJA::statement::Lambda<0, RAJA::Segs<0, 1>>
        >
      >
    >;

  RAJA::kernel<EXEC_POL4>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      c_raja_omp_no_collapse[i + Ni*j] = i + Ni*j;
  });

  checkResult(c_raja_omp_no_collapse, c_ref, N_2D);
//printArray(c_raja_omp_no_collapse, N_2D);

#endif


//----------------------------------------------------------------------------//
// OpenMP target variants
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_TARGET_OPENMP)

  hid = omp_get_initial_device();
  did = omp_get_default_device();

//----------------------------------------------------------------------------//
// Base OpenMP target variant
//----------------------------------------------------------------------------//

  std::memset(c_base_omptarget, 0, N_2D * sizeof(int));
  allocAndInitOpenMPDeviceData(dc, c_base_omptarget, N_2D, did, hid);

  std::cout << "\n Running base OpenMP target matrix init...\n";

  #pragma omp target is_device_ptr(dc) device( did )
  #pragma omp teams distribute parallel for thread_limit(threads_per_team) schedule(static, 1) collapse(2)
  for (int j = 0; j < Nj; ++j) {
    for (int i = 0; i < Ni; ++i) {
      dc[i + Ni*j] = i + Ni*j;
    }
  }

  getOpenMPDeviceData(c_base_omptarget, dc, N_2D, hid, did);

  checkResult(c_base_omptarget, c_ref, N_2D);
//printArray(c_base_omptarget, N_2D);

#endif
 
//----------------------------------------------------------------------------//
// RAJA OpenMP target variants
//----------------------------------------------------------------------------//

#if defined(RAJA_ENABLE_TARGET_OPENMP)

  std::memset(c_raja_omptarget, 0, N_2D * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget, N_2D, did, hid);

  using KERNEL_POL1 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                RAJA::ArgList<0, 1>,
        RAJA::statement::Lambda<0>
      >
    >;

  std::cout << "\n Running RAJA OpenMP target matrix init (collapse)...\n";

  RAJA::kernel<KERNEL_POL1>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      dc[i + Ni*j] = i + Ni*j;
  });

  getOpenMPDeviceData(c_raja_omptarget, dc, N_2D, hid, did);

  checkResult(c_raja_omptarget, c_ref, N_2D);
//printArray(c_raja_omptarget, N_2D);

#endif

////////////////////////////////////////////

#if defined(RAJA_ENABLE_TARGET_OPENMP)

  std::memset(c_raja_omptarget, 0, N_2D * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget, N_2D, did, hid);

  using KERNEL_POL2 =
    RAJA::KernelPolicy<
      RAJA::statement::Collapse<RAJA::omp_target_parallel_collapse_exec,
                                RAJA::ArgList<0, 1>,
        RAJA::statement::Lambda<0, RAJA::Segs<0, 1>>
      >
    >;

  std::cout << "\n Running RAJA OpenMP target matrix init (collapse, Segs)...\n";

  RAJA::kernel<KERNEL_POL2>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      dc[i + Ni*j] = i + Ni*j;
  });

  getOpenMPDeviceData(c_raja_omptarget, dc, N_2D, hid, did);

  checkResult(c_raja_omptarget, c_ref, N_2D);
//printArray(c_raja_omptarget, N_2D);

#endif

////////////////////////////////////////////

#if defined(RAJA_ENABLE_TARGET_OPENMP)

  std::memset(c_raja_omptarget_no_collapse, 0, N_2D * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget_no_collapse, N_2D, did, hid);

  using KERNEL_POL3 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
        RAJA::statement::For<1, RAJA::seq_exec,
          RAJA::statement::Lambda<0>
        >
      >
    >;

  std::cout << "\n Running RAJA OpenMP target matrix init (no collapse)...\n";

  RAJA::kernel<KERNEL_POL3>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      dc[i + Ni*j] = i + Ni*j;
  });

  getOpenMPDeviceData(c_raja_omptarget_no_collapse, dc, N_2D, hid, did);

  checkResult(c_raja_omptarget_no_collapse, c_ref, N_2D);
//printArray(c_raja_omptarget_no_collapse, N_2D);

#endif

////////////////////////////////////////////

#if defined(RAJA_ENABLE_TARGET_OPENMP)

  std::memset(c_raja_omptarget_no_collapse, 0, N_2D * sizeof(int));
  initOpenMPDeviceData(dc, c_raja_omptarget_no_collapse, N_2D, did, hid);

  using KERNEL_POL4 =
    RAJA::KernelPolicy<
      RAJA::statement::For<0, RAJA::omp_target_parallel_for_exec<threads_per_team>,
        RAJA::statement::For<1, RAJA::seq_exec,
          RAJA::statement::Lambda<0, RAJA::Segs<0, 1>>
        >
      >
    >;

  std::cout << "\n Running RAJA OpenMP target matrix init (no collapse, Segs)...\n";

  RAJA::kernel<KERNEL_POL4>(
    RAJA::make_tuple(RAJA::RangeSegment{0, Nj},
                     RAJA::RangeSegment{0, Ni}),
    [=](int j, int i) {
      dc[i + Ni*j] = i + Ni*j;
  });

  getOpenMPDeviceData(c_raja_omptarget_no_collapse, dc, N_2D, hid, did);

  checkResult(c_raja_omptarget_no_collapse, c_ref, N_2D);
//printArray(c_raja_omptarget_no_collapse, N_2D);

#endif


// cleanup

  delete [] c_ref;
  delete [] c_base_omp;
  delete [] c_raja_omp;
  delete [] c_raja_omp_no_collapse;
  delete [] c_base_omptarget;
  delete [] c_raja_omptarget;
  delete [] c_raja_omptarget_no_collapse;

#if defined(RAJA_ENABLE_TARGET_OPENMP)
  deallocOpenMPDeviceData(dc, did);
#endif

  std::cout << "\n 2D case DONE!...\n";
///////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////

  return 0;
}

//
// Function to check result and report P/F.
//
void checkResult(int* c, int* c_ref, int len)
{
  bool correct = true;
  for (int i = 0; i < len; i++) {
    if ( correct && c[i] != c_ref[i] ) { correct = false; }
  }
  if ( correct ) {
    std::cout << "\n\t result -- PASS\n";
  } else {
    std::cout << "\n\t result -- FAIL\n";
  }
}

//
// Function to print array.
//
void printArray(int* v, int len)
{
  std::cout << std::endl;
  for (int i = 0; i < len; i++) {
    std::cout << "v[" << i << "] = " << v[i] << std::endl;
  }
  std::cout << std::endl;
}

