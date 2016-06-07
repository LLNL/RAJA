#ifndef KRIPKERAJA_H__
#define KRIPKERAJA_H__

#include<RAJA/RAJA.hxx>
#include<Kripke/Grid.h>
#include<Kripke/Subdomain.h>


//#define RAJA_INLINE __attribute__((always_inline))


#define RAJA_LAMBDA [=]
//#define RAJA_LAMBDA [=] __device__


// All of our OpenMP execution policies are swapped out with sequential if
// an OpenMP compiler is not available.
// Note:  It is always safe to replace OpenMP loops with sequential loops for
// this code.
#ifdef RAJA_ENABLE_OPENMP
using kripke_omp_for_nowait_exec = RAJA::omp_for_nowait_exec;
using kripke_omp_collapse_nowait_exec = RAJA::omp_collapse_nowait_exec;

template<typename T>
using kripke_OMP_Parallel = RAJA::OMP_Parallel<T>;

#else
typedef RAJA::simd_exec kripke_omp_for_nowait_exec;
typedef RAJA::simd_exec kripke_omp_collapse_nowait_exec;

template<typename T>
using kripke_OMP_Parallel = RAJA::Execute;

#endif

// Subdomain loops
template<typename SubdomainPolicy, typename BODY>
RAJA_INLINE void forallSubdomains(Grid_Data *grid_data, BODY body){

  RAJA::forall<SubdomainPolicy>(
    RAJA::RangeSegment(0, grid_data->subdomains.size()),
    [=](int sdom_id){
      // get subdomain object
      Subdomain &sdom = grid_data->subdomains[sdom_id];

      body(sdom_id, sdom);
    });

}

// Loop over zoneset subdomains
template<typename SubdomainPolicy, typename BODY>
RAJA_INLINE void forallZoneSets(Grid_Data *grid_data, BODY body){

  RAJA::forall<SubdomainPolicy>(
    RAJA::RangeSegment(0, grid_data->num_zone_sets),
    [=](int zs){
      // get material mix information
      int sdom_id = grid_data->zs_to_sdomid[zs];
      Subdomain &sdom = grid_data->subdomains[sdom_id];

      body(zs, sdom_id, sdom);
    });

}



//#define KRIPKE_USE_PSCOPE

#ifdef KRIPKE_USE_PSCOPE


#define FORALL_SUBDOMAINS(SDOM_POL, DOMAIN, ID, SDOM) \
  forallSubdomains<SDOM_POL>(DOMAIN, [&](int ID, Subdomain &SDOM){

#define FORALL_ZONESETS(SDOM_POL, DOMAIN, ID, SDOM) \
  forallZoneSets<SDOM_POL>(DOMAIN, [&](int zone_set, int ID, Subdomain &SDOM){


#define END_FORALL });

#else

// Eliminates policy scope outer lambda,
#define BEGIN_POLICY(NVAR, NTYPE) \
  { \
    typedef NEST_DGZ_T NTYPE;

#define END_POLICY }


#define FORALL_SUBDOMAINS(SDOM_POL, DOMAIN, ID, SDOM) \
  for(int ID = 0;ID < DOMAIN.subdomains.size();++ ID){ \
    Subdomain &SDOM = DOMAIN.subdomains[ID];

#define FORALL_ZONESETS(SDOM_POL, DOMAIN, ID, SDOM) \
  for(int _zset_idx = 0;_zset_idx < DOMAIN.num_zone_sets;++ _zset_idx){ \
    int ID = DOMAIN.zs_to_sdomid[_zset_idx]; \
    Subdomain &SDOM = DOMAIN.subdomains[ID];


#define END_FORALL }

#endif

#endif
