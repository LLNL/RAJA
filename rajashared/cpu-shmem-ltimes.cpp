#include <stddef.h>
#include <stdio.h>
struct CopyTest {
    static size_t default_count;
    static size_t copy_count;
    static size_t move_count;
    static size_t call_count;

    inline
    CopyTest(){
      default_count ++;
    }

    inline
    CopyTest(CopyTest const &){
      copy_count ++;
    }

    inline
    CopyTest(CopyTest &&){
      move_count ++;
    }

    inline
    void call() const {
      call_count ++;
    }

    static void print(){
      printf("CopyTest:  default=%ld, copy=%ld, move=%ld, call=%ld\n",
          (long)default_count, (long)copy_count, (long)move_count, (long)call_count);
    }
};


#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"


#include <cstdio>
#include <cmath>


 size_t CopyTest::default_count = 0;
 size_t CopyTest::copy_count = 0;
 size_t CopyTest::move_count = 0;
 size_t CopyTest::call_count = 0;


using namespace RAJA;


RAJA_INDEX_VALUE(IMoment, "IMoment");
RAJA_INDEX_VALUE(IDirection, "IDirection");
RAJA_INDEX_VALUE(IGroup, "IGroup");
RAJA_INDEX_VALUE(IZone, "IZone");


void runLTimesBare(bool ,
                          Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{

  using namespace RAJA::nested;


  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones, 0.0);


  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = i; //drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = 2*i; //drand48();
  }

  for (size_t i = 0; i < phi_data.size(); ++i) {
    phi_data[i] = 0; //drand48();
  }


  double * __restrict__ d_ell = &ell_data[0];
  double * __restrict__ d_phi = &phi_data[0];
  double * __restrict__ d_psi = &psi_data[0];



  RAJA::Timer timer;
  timer.start();

  for (int m(0); m < num_moments; ++m) {
    for (int g(0); g < num_groups; ++g) {
      for (int d(0); d < num_directions; ++d) {
        for (int z(0); z < num_zones; ++z) {
          d_phi[m*num_groups*num_zones + g*num_zones + z] +=
              d_ell[m*num_directions + d] *
              d_psi[d*num_groups*num_zones + g*num_zones + z];
        }

      }
    }
  }



  timer.stop();
  printf("LTimes took %lf seconds using bare loops\n",
      timer.elapsed());



  // Check correctness
  //printf("%e, %e, %e\n", ell_data[1], phi_data[1], psi_data[1]);

}

void runLTimesBareView(bool debug,
                          Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{

  using namespace RAJA::nested;

#if 1
  // psi[direction, group, zone]
  using PsiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IMoment, IGroup, IZone>;

  // ell[moment, direction]
  using EllView = RAJA::TypedView<double, Layout<2, Index_type, 1>, IMoment, IDirection>;
#else
  using PsiView = RAJA::TypedView<double, StaticLayout<80,32,16*1024>, IDirection, IGroup, IZone>;
  using PhiView = RAJA::TypedView<double, StaticLayout<25,32,16*1024>, IMoment, IGroup, IZone>;
  using EllView = RAJA::TypedView<double, StaticLayout<25,80>, IMoment, IDirection>;
#endif

  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones, 0.0);


  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = i; //drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = 2*i; //drand48();
  }

  for (size_t i = 0; i < phi_data.size(); ++i) {
    phi_data[i] = 0; //drand48();
  }


  double *d_ell = &ell_data[0];
  double *d_phi = &phi_data[0];
  double *d_psi = &psi_data[0];



  // create views on data
  std::array<camp::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({{num_moments, num_directions}}, ell_perm));

  std::array<camp::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(
      d_psi,
      make_permuted_layout({{num_directions, num_groups, num_zones}}, psi_perm));

  std::array<camp::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(
      d_phi,
      make_permuted_layout({{num_moments, num_groups, num_zones}}, phi_perm));



  RAJA::Timer timer;
  timer.start();

  for (IMoment m(0); m < num_moments; ++m) {
    for (IGroup g(0); g < num_groups; ++g) {
      for (IDirection d(0); d < num_directions; ++d) {
        for (IZone z(0); z < num_zones; ++z) {
#if 1
          phi(m, g, z) += ell(m, d) * psi(d, g, z);
#else
          d_phi[(*m)*num_groups*num_zones + (*g)*num_zones + *z] +=
              d_ell[(*m)*num_directions + *d] *
              d_psi[(*d)*num_groups*num_zones + (*g)*num_zones + *z];
#endif

        }

      }
    }
  }



  timer.stop();
  printf("LTimes took %lf seconds using bare loops and RAJA::Views\n",
      timer.elapsed());



  // Check correctness
  if(debug){

    size_t errors = 0;
    double total_error = 0.;
    for (IZone z(0); z < num_zones; ++z) {
      for (IGroup g(0); g < num_groups; ++g) {
        for (IMoment m(0); m < num_moments; ++m) {
          double total = 0.0;
          for (IDirection d(0); d < num_directions; ++d) {
            double val = ell(m, d) * psi(d, g, z);
            total += val;
          }
          if(std::abs(total-phi(m,g,z)) > 1e-9){
            ++ errors;
          }
          total_error += std::abs(total-phi(m,g,z));
        }
      }
    }
    if(errors == 0){
      printf("  -- no errors (%e)\n", total_error);
    }
    else{
      printf("  -- failed : %ld errors\n", (long)errors);
    }
  }
  //printf("%e, %e, %e\n", ell_data[1], phi_data[1], psi_data[1]);

}




void runLTimesRajaNested(bool debug,
                          Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{

  using namespace RAJA::nested;

#if 1
  // psi[direction, group, zone]
  using PsiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IMoment, IGroup, IZone>;

  // ell[moment, direction]
  using EllView = RAJA::TypedView<double, Layout<2, Index_type, 1>, IMoment, IDirection>;
#else
  using PsiView = RAJA::TypedView<double, StaticLayout<80,32,32*1024>, IDirection, IGroup, IZone>;
  using PhiView = RAJA::TypedView<double, StaticLayout<25,32,32*1024>, IMoment, IGroup, IZone>;
  using EllView = RAJA::TypedView<double, StaticLayout<25,80>, IMoment, IDirection>;
#endif

  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones, 0.0);


  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = i; //drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = 2*i; //drand48();
  }

  for (size_t i = 0; i < phi_data.size(); ++i) {
    phi_data[i] = 0; //drand48();
  }


  double *d_ell = &ell_data[0];
  double *d_phi = &phi_data[0];
  double *d_psi = &psi_data[0];



  // create views on data
  std::array<camp::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({{num_moments, num_directions}}, ell_perm));

  std::array<camp::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(
      d_psi,
      make_permuted_layout({{num_directions, num_groups, num_zones}}, psi_perm));

  std::array<camp::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(
      d_phi,
      make_permuted_layout({{num_moments, num_groups, num_zones}}, phi_perm));



#if 1
  using Pol = RAJA::nested::Policy<
    For<0, loop_exec,
      For<1, loop_exec,
        For<2, loop_exec,
          For<3, loop_exec, Lambda<0>>
        >
      >
    >>;
#else
  using Pol = RAJA::nested::Policy<
    Collapse<seq_exec, ArgList<0,1,2,3>, Lambda<0>>
    >;
#endif


  RAJA::Timer timer;
  timer.start();

  auto segments =  camp::make_tuple(TypedRangeSegment<IMoment>(0, num_moments),
      TypedRangeSegment<IDirection>(0, num_directions),
      TypedRangeSegment<IGroup>(0, num_groups),
      TypedRangeSegment<IZone>(0, num_zones));


  nested::forall(
      Pol{},

      segments,

      // Lambda_CalcPhi
      [=] (IMoment m, IDirection d, IGroup g, IZone z) {
        phi(m, g, z) += ell(m, d) * psi(d, g, z);
      });



  timer.stop();
  printf("LTimes took %lf seconds using RAJA::nested::forall\n",
      timer.elapsed());


  CopyTest::print();

  // Check correctness
  if(debug){

    size_t errors = 0;
    double total_error = 0.;
    for (IZone z(0); z < num_zones; ++z) {
      for (IGroup g(0); g < num_groups; ++g) {
        for (IMoment m(0); m < num_moments; ++m) {
          double total = 0.0;
          for (IDirection d(0); d < num_directions; ++d) {
            double val = ell(m, d) * psi(d, g, z);
            total += val;
          }
          if(std::abs(total-phi(m,g,z)) > 1e-9){
            ++ errors;
          }
          total_error += std::abs(total-phi(m,g,z));
        }
      }
    }
    if(errors == 0){
      printf("  -- no errors (%e)\n", total_error);
    }
    else{
      printf("  -- failed : %ld errors\n", (long)errors);
    }
  }
  //printf("%e, %e, %e\n", ell_data[1], phi_data[0], psi_data[1]);

}

#if 1

void runLTimesRajaNestedShmem(bool debug,
                          Index_type num_moments,
                          Index_type num_directions,
                          Index_type num_groups,
                          Index_type num_zones)
{

  using namespace RAJA::nested;

#if 1
  // psi[direction, group, zone]
  using PsiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IDirection, IGroup, IZone>;

  // phi[moment, group, zone]
  using PhiView = RAJA::TypedView<double, Layout<3, Index_type, 2>, IMoment, IGroup, IZone>;

  // ell[moment, direction]
  using EllView = RAJA::TypedView<double, Layout<2, Index_type, 1>, IMoment, IDirection>;
#else
  using PsiView = RAJA::TypedView<double, StaticLayout<80,32,16*1024>, IDirection, IGroup, IZone>;
  using PhiView = RAJA::TypedView<double, StaticLayout<25,32,16*1024>, IMoment, IGroup, IZone>;
  using EllView = RAJA::TypedView<double, StaticLayout<25,80>, IMoment, IDirection>;
#endif


  // allocate data
  // phi is initialized to all zeros, the others are randomized
  std::vector<double> ell_data(num_moments * num_directions);
  std::vector<double> psi_data(num_directions * num_groups * num_zones);
  std::vector<double> phi_data(num_moments * num_groups * num_zones, 0.0);


  // randomize data
  for (size_t i = 0; i < ell_data.size(); ++i) {
    ell_data[i] = i; //drand48();
  }

  for (size_t i = 0; i < psi_data.size(); ++i) {
    psi_data[i] = 2*i; //drand48();
  }

  for (size_t i = 0; i < phi_data.size(); ++i) {
    phi_data[i] = 0; //drand48();
  }


  double *d_ell = &ell_data[0];
  double *d_phi = &phi_data[0];
  double *d_psi = &psi_data[0];



  // create views on data
  std::array<camp::idx_t, 2> ell_perm {{0, 1}};
  EllView ell(
      d_ell,
      make_permuted_layout({{num_moments, num_directions}}, ell_perm));

  std::array<camp::idx_t, 3> psi_perm {{0, 1, 2}};
  PsiView psi(
      d_psi,
      make_permuted_layout({{num_directions, num_groups, num_zones}}, psi_perm));

  std::array<camp::idx_t, 3> phi_perm {{0, 1, 2}};
  PhiView phi(
      d_phi,
      make_permuted_layout({{num_moments, num_groups, num_zones}}, phi_perm));


  constexpr size_t tile_moments = 25;
  constexpr size_t tile_directions = 80;
  constexpr size_t tile_zones = 32;
  constexpr size_t tile_groups = 0;


  using Pol = RAJA::nested::Policy<
    nested::Tile<0, nested::tile_fixed<tile_moments>, seq_exec,
      nested::Tile<1, nested::tile_fixed<tile_directions>, seq_exec,
        SetShmemWindow<

          // Load shmem L
          For<0, loop_exec, For<1, loop_exec, Lambda<0>>>,

          For<2, loop_exec,
            nested::Tile<3, nested::tile_fixed<tile_zones>, seq_exec,
            SetShmemWindow<
              // Load Psi into shmem
              For<1, loop_exec, For<3, loop_exec, Lambda<1> >>,

              // Zero shmem phi
              For<0, loop_exec, For<3, loop_exec, Lambda<2> >>,

              // Compute L*Psi
              For<0, loop_exec, For<1, loop_exec, For<3, loop_exec, Lambda<3> >>>,

              // Write shmem phi
              For<0, loop_exec, For<3, loop_exec, Lambda<4> >>
            >
            > // Tile zones
          > // for g
        > // Shmem Window (mom, dir)
      > // Tile directions
    > // Tile moments
  >; // Pol


  RAJA::Timer timer;
  timer.start();

  auto segments =  camp::make_tuple(TypedRangeSegment<IMoment>(0, num_moments),
      TypedRangeSegment<IDirection>(0, num_directions),
      TypedRangeSegment<IGroup>(0, num_groups),
      TypedRangeSegment<IZone>(0, num_zones));


  using shmem_ell_t = SharedMemory<seq_shmem, double, tile_moments*tile_directions>;
  ShmemWindowView<shmem_ell_t, ArgList<0,1>, SizeList<tile_moments, tile_directions>, decltype(segments)> shmem_ell;

  using shmem_psi_t = SharedMemory<seq_shmem, double, tile_zones*tile_directions>;
  ShmemWindowView<shmem_psi_t, ArgList<1, 2, 3>, SizeList<tile_directions, tile_groups, tile_zones>, decltype(segments)> shmem_psi;

  using shmem_phi_t = SharedMemory<seq_shmem, double, tile_zones*tile_moments>;
  ShmemWindowView<shmem_phi_t, ArgList<0, 2, 3>, SizeList<tile_moments, tile_groups, tile_zones>, decltype(segments)> shmem_phi;


  nested::forall(
      Pol{},

      segments,

      // Lambda_LoadEll
      [=] (IMoment m, IDirection d, IGroup, IZone) {
        shmem_ell(m, d) = ell(m, d);
      },

      // Lambda_LoadPsi
      [=] (IMoment, IDirection d, IGroup g, IZone z) {
        shmem_psi(d, g, z) = psi(d, g, z);
      },

      // Lambda_LoadPhi
      [=] (IMoment m, IDirection, IGroup g, IZone z) {
        shmem_phi(m, g, z) = phi(m,g,z);
      },

      // Lambda_CalcPhi
      [=] (IMoment m, IDirection d, IGroup g, IZone z) {
        shmem_phi(m, g, z) += shmem_ell(m, d) * shmem_psi(d, g, z);
      },

      // Lambda_SavePhi
      [=] (IMoment m, IDirection, IGroup g, IZone z) {
        phi(m,g,z) = shmem_phi(m, g, z);
      });



  timer.stop();
  printf("LTimes took %lf seconds using RAJA::nested::forall and shmem\n",
      timer.elapsed());



  // Check correctness
  if(debug){

    size_t errors = 0;
    double total_error = 0.;
    for (IZone z(0); z < num_zones; ++z) {
      for (IGroup g(0); g < num_groups; ++g) {
        for (IMoment m(0); m < num_moments; ++m) {
          double total = 0.0;
          for (IDirection d(0); d < num_directions; ++d) {
            double val = ell(m, d) * psi(d, g, z);
            total += val;
          }
          if(std::abs(total-phi(m,g,z)) > 1e-9){
            ++ errors;
          }
          total_error += std::abs(total-phi(m,g,z));
        }
      }
    }
    if(errors == 0){
      printf("  -- no errors (%e)\n", total_error);
    }
    else{
      printf("  -- failed : %ld errors\n", (long)errors);
    }
  }
  //printf("%e, %e, %e\n", ell_data[1], phi_data[0], psi_data[1]);

}


#endif

int main(){

  bool debug = false;

  int m = 25;
  int d = 80;
  int g = 32;
  int z = 32*1024;

  printf("m=%d, d=%d, g=%d, z=%d\n", m, d, g, z);

  runLTimesBare(debug, m, d, g, z);
  runLTimesBareView(debug, m, d, g, z);
  runLTimesRajaNested(debug, m, d, g, z);
  runLTimesRajaNestedShmem(debug, m, d, g, z);



  return 0;
}


