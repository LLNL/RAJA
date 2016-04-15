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

#ifndef KRIPKE_DIRECTIONS_H__
#define KRIPKE_DIRECTIONS_H__

#include <vector>

struct Grid_Data;
struct Input_Variables;

/**
 * Contains information needed for one quadrature set direction.
 */
struct Directions{
  double xcos;              /* Absolute value of the x-direction cosine. */
  double ycos;              /* Absolute value of the y-direction cosine. */
  double zcos;              /* Absolute value of the z-direction cosine. */
  double w;                 /* weight for the quadrature rule.*/
  int id;                   /* direction flag (= 1 if x-direction
                            cosine is positive; = -1 if not). */
  int jd;                   /* direction flag (= 1 if y-direction
                            cosine is positive; = -1 if not). */
  int kd;                   /* direction flag (= 1 if z-direction
                            cosine is positive; = -1 if not). */
  int octant;
};


void InitDirections(Grid_Data *grid_data, Input_Variables *input_vars);

#endif
