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

#ifndef KRIPKE_SUBDOMAIN_H__
#define KRIPKE_SUBDOMAIN_H__

#include <vector>
#include <Kripke/Layout.h>

// Foreward Decl
struct Directions;
struct SubTVec;
struct Input_Variables;
class Kernel;

/**
 * Provides sweep index sets for a given octant.
 * This generalizes the sweep pattern, and allows for experimenting with
 * a tiled approach to on-node sweeps.
 */
struct Grid_Sweep_Block {
  int start_i, start_j, start_k; // starting index
  int end_i, end_j, end_k; // termination conditon (one past)
  int inc_i, inc_j, inc_k; // increment
};



/**
 * Contains parameters and variables that describe a single Group Set and
 * Direction Set.
 */
struct Subdomain {
  Subdomain();
  ~Subdomain();

  void setup(int sdom_id, Input_Variables *input_vars, int gs, int ds, int zs,
    std::vector<Directions> &direction_list, Kernel *kernel, Layout *layout);

  void setVars(SubTVec *ell_ptr, SubTVec *ell_plus_ptr,
    SubTVec *phi_ptr, SubTVec *phi_out_ptr);

  void randomizeData(void);
  void copy(Subdomain const &b);
  bool compare(Subdomain const &b, double tol, bool verbose);
  void computeSweepIndexSet(void);
  void computeLLPlus(int legendre_order);

  int idx_group_set;
  int idx_dir_set;
  int idx_zone_set;

  int num_groups;       // Number of groups in this set
  int num_directions;   // Number of directions in this set
  int num_zones;        // Number of zones in this set

  double zeros[3];                     // origin of local mesh
  int nzones[3];                    // Number of zones in each dimension
  std::vector<double> deltas[3];    // Spatial grid deltas in each dimension (including ghost zones)

  int group0;           // Starting global group id
  int direction0;       // Starting global direction id

  Grid_Sweep_Block sweep_block;

  // Neighbors
  Neighbor upwind[3];   // Upwind dependencies in x,y,z
  Neighbor downwind[3]; // Downwind neighbors in x,y,z

  // Sweep boundary data
  SubTVec *plane_data[3];
  SubTVec *old_plane_data[3];

  // Variables
  SubTVec *psi;         // Solution
  SubTVec *rhs;         // RHS, source term
  SubTVec *sigt;        // Zonal per-group cross-section

  // Pointers into directions and directionset data from Grid_Data
  Directions *directions;
  SubTVec *ell;
  SubTVec *ell_plus;
  SubTVec *phi;
  SubTVec *phi_out;

  // Materials on the mesh, used for scattering lookup
  double reg_volume[3];               // volume of each material region
  std::vector<double> volume;         // volume of each zone
  std::vector<int> mixed_to_zones;    // mapping from mixed slot to zones
  std::vector<int> num_mixed;         // mapping from mixed slot to zones
  std::vector<int> zones_to_mixed;    // mapping from zones to first mixed slot
  std::vector<int> mixed_material;    // material number for each mixed slot
  std::vector<double> mixed_fraction; // volume fraction each mixed slot
};

#endif
