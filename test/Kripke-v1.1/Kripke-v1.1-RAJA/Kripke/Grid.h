/*
 * NOTICE
 *
 * This work was produced at the Lawrence Livermore National Laboratory (LLNL)
 * under contract no. DE-AC-52-07NA27344 (Contract 44) between the U.S.
 * Department of Energy (DOE) and Lawrence Livermore National Security, LLC
 * (LLNS) for the operation of LLNL. The rights of the Federal Government are
 * reserved under Contract 44.
 *
 * DISCLAIMER
 *
 * This work was prepared as an account of work sponsored by an agency of the
 * United States Government. Neither the United States Government nor Lawrence
 * Livermore National Security, LLC nor any of their employees, makes any
 * warranty, express or implied, or assumes any liability or responsibility
 * for the accuracy, completeness, or usefulness of any information, apparatus,
 * product, or process disclosed, or represents that its use would not infringe
 * privately-owned rights. Reference herein to any specific commercial products,
 * process, or service by trade name, trademark, manufacturer or otherwise does
 * not necessarily constitute or imply its endorsement, recommendation, or
 * favoring by the United States Government or Lawrence Livermore National
 * Security, LLC. The views and opinions of authors expressed herein do not
 * necessarily state or reflect those of the United States Government or
 * Lawrence Livermore National Security, LLC, and shall not be used for
 * advertising or product endorsement purposes.
 *
 * NOTIFICATION OF COMMERCIAL USE
 *
 * Commercialization of this product is prohibited without notifying the
 * Department of Energy (DOE) or Lawrence Livermore National Security.
 */

#ifndef KRIPKE_GRID_DATA_H__
#define KRIPKE_GRID_DATA_H__

#include <Kripke/Directions.h>
#include <Kripke/Kernel.h>
#include <Kripke/Subdomain.h>
#include <Kripke/Timing.h>
#include <RAJA/IndexValue.hxx>
#include <mpi.h>
#include <vector>

// Foreward Decl
struct Input_Variables;
struct SubTVec;


/**
 * Contains all grid parameters and variables.
 */
struct Grid_Data {
public:
  explicit Grid_Data(Input_Variables *input_vars);
  ~Grid_Data();

  void randomizeData(void);
  void copy(Grid_Data const &b);
  bool compare(Grid_Data const &b, double tol, bool verbose);
  double particleEdit(void);
#ifdef KRIPKE_USE_SILO
  void writeSilo(std::string const &fname);
#endif

  Timing timing;

  int niter;

  double source_value;

  std::vector<double> sigma_tot;            // Cross section data

  int num_group_sets;                       // Number of group-sets
  int num_groups_per_set;                   // How many groups in each set
  int num_direction_sets;                   // Number of direction-sets
  int num_directions_per_set;               // Number of directions per dir set
  int num_zone_sets;                        // Number of zone sets
  int legendre_order;                       // Legendra expansion order ( >= 0 )
  int total_num_moments;                    // Number of spherical harmonic moments

  std::vector<int> moment_to_coeff;         // Map from harmonic moments to legendre coefficients

  std::vector<Directions> directions;       // Quadrature point data, for all directions
  Kernel *kernel;                           // Layout-specific math kernels

  std::vector<Subdomain> subdomains;        // Group/Angle/Zone set data
  std::vector<int> zs_to_sdomid;            // map of zonesets to subdomains with ds=gs=0

  // Variables:
  SubTVec *sigs;                            // scattering lookup table for each material
                                            // G=g->gp, D=legendre coeff, Z=matidx

  // Per directionset ell and ell_plus matrices (Subdomain point into these arrays)
  std::vector<SubTVec *> ell;               // L matrix in nm_offset coordinates
  std::vector<SubTVec *> ell_plus;          // L+ matrix in nm_offset coordinates

  // Per zoneset phi and phi_out (Subdomains point into these arrays)
  std::vector<SubTVec *> phi;               // Moments of psi
  std::vector<SubTVec *> phi_out;           // Scattering source

  template<typename T>
  inline int indexSize(int sdom_id){

    // Get size of index from hash in the Subdomain object
    //Subdomain &sdom = subdomains[sdom_id];
    //sdom.index_size[T::getName()];
    return subdomains[sdom_id].index_size[T::getName()];
  }

  template<typename T>
  inline RAJA::RangeSegment indexRange(int sdom_id){

    // Get size of index from hash in the Subdomain object
    int len = indexSize<T>(sdom_id);

    // Construct a range covering that Index
    return RAJA::RangeSegment(0, len);
  }
};

#endif
