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

#ifndef KRIPKE_INPUT_VARIABLES_H__
#define KRIPKE_INPUT_VARIABLES_H__

#include<Kripke.h>

/**
 * This structure defines the input parameters to setup a problem.
 */

struct Input_Variables {
  Input_Variables();
  
  bool checkValues(void) const;
  
  // Problem Description
  int nx, ny, nz;               // Number of spatial zones in x,y,z
  int num_directions;           // Total number of directions
  int num_groups;               // Total number of energy groups
  int legendre_order;           // Scattering order (number Legendre coeff's - 1)
  int quad_num_polar;           // Number of polar quadrature points (0 for dummy)
  int quad_num_azimuthal;       // Number of azimuthal quadrature points (0 for dummy)

  // On-Node Options
  Nesting_Order nesting;        // Data layout and loop ordering (of Psi)
  
  // Parallel Decomp
  int npx, npy, npz;            // The number of processors in x,y,z
  int num_dirsets;              // Number of direction sets
  int num_groupsets;            // Number of energy group sets
  int num_zonesets_dim[3];      // Number of zoneset in x, y, z  
  int layout_pattern;           // Which subdomain/task layout to use
  
  // Physics and Solver Options
  int niter;                    // number of solver iterations to run
  ParallelMethod parallel_method;
  double sigt[3];               // total cross section for 3 materials
  double sigs[3];               // total scattering cross section for 3 materials
  
  // Output Options
  std::string run_name;         // Name to use when generating output files
#ifdef KRIPKE_USE_SILO
  std::string silo_basename;    // name prefix for silo output files
#endif

};

#endif
