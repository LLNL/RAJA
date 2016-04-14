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

#include <Kripke/Directions.h>
#include <Kripke/Grid.h>
#include <Kripke/Input_Variables.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
#include <algorithm>

namespace {
  /*
    GaussLegendre returns the n point Gauss-Legendre quadrature rule for
    the integral between x1 and x2.
  */
  void GaussLegendre(double x1, double x2, std::vector<double> &x,
      std::vector<double> &w, double eps)
  {
    int n = x.size();
    int m, j, i;
    double z1, z, xm, xl, pp, p3, p2, p1;

    m=(n+1)/2;
    xm=0.5*(x2+x1);
    xl=0.5*(x2-x1);
    for(i=1; i<=m; i++){
      z=cos(M_PI*(i-0.25)/(n+0.5));
      do {
        p1=1.0;
        p2=0.0;
        for(j=1; j<=n; j++){
          p3=p2;
          p2=p1;
          p1=((2.0*j-1.0)*z*p2-(j-1.0)*p3)/j;
        }
        pp=n*(z*p1-p2)/(z*z-1.0);
        z1=z;
        z=z1-p1/pp;
      } while(fabs(z-z1) > eps);
      x[i-1]=xm-xl*z;
      x[n-i]=xm+xl*z;
      w[i-1]=2.0*xl/((1.0-z*z)*pp*pp);

      w[n-i]=w[i-1];
    }
  }


  bool dirSortFcn(Directions const &a, Directions const &b){
    return b.octant < a.octant;
  }
}

/**
 * Initializes the quadrature set information for a Grid_Data object.
 * This guarantees that each <GS,DS> pair have a single originating octant.
 */
void InitDirections(Grid_Data *grid_data, Input_Variables *input_vars)
{
  std::vector<Directions> &directions = grid_data->directions;

  // Get set description from user
  int num_directions_per_octant = input_vars->num_directions/8;
  int num_directions = input_vars->num_directions;

  // allocate storage
  directions.resize(num_directions);

  // Are we running a REAL quadrature set?
  int num_polar = input_vars->quad_num_polar;
  int num_azimuth = input_vars->quad_num_azimuthal;

  std::vector<double> polar_cos;
  std::vector<double> polar_weight;
  if(num_polar > 0){
    // make sure the user specified the correct number of quadrature points
    if(num_polar % 4 != 0){
      printf("Must have number of polar angles be a multiple of 4\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(num_azimuth % 2 != 0){
      printf("Must have number of azimuthal angles be a multiple of 2\n");
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    if(num_polar*num_azimuth != num_directions){
      printf("You need to specify %d total directions, not %d\n",
          num_polar*num_azimuth, num_directions);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Compute gauss legendre weights
    polar_cos.resize(num_polar);
    polar_weight.resize(num_polar);
    GaussLegendre(-1.0, 1.0, polar_cos, polar_weight, DBL_EPSILON);

    // compute azmuhtal angles and weights
    std::vector<double> az_angle(num_azimuth);
    std::vector<double> az_weight(num_azimuth);
    double dangle = 2.0*M_PI/((double) num_azimuth);

    for(int i=0; i<num_azimuth; i++){
      if(i == 0){
        az_angle[0] = dangle/2.0;
      }
      else{
        az_angle[i] = az_angle[i-1] + dangle;
      }
      az_weight[i] = dangle;
    }


    // Loop over polar 'octants
    int d = 0;
    for(int i=0; i< num_polar; i++){
      for(int j=0; j< num_azimuth; j++){
        double xcos = sqrt(1.0-polar_cos[i]*polar_cos[i]) * cos(az_angle[j]);
        double ycos = sqrt(1.0-polar_cos[i]*polar_cos[i]) * sin(az_angle[j]);
        double zcos = polar_cos[i];
        double w = polar_weight[i]*az_weight[j];

        directions[d].id = (xcos > 0.) ? 1 : -1;
        directions[d].jd = (ycos > 0.) ? 1 : -1;
        directions[d].kd = (zcos > 0.) ? 1 : -1;

        directions[d].octant = 0;
        if(directions[d].id == -1){
          directions[d].octant += 1;
        }
        if(directions[d].jd == -1){
          directions[d].octant += 2;
        }
        if(directions[d].kd == -1){
          directions[d].octant += 4;
        }

        directions[d].xcos = std::abs(xcos);
        directions[d].ycos = std::abs(ycos);
        directions[d].zcos = std::abs(zcos);
        directions[d].w = w;

        ++ d;
      }
    }

    // Sort by octant.. so each set has same directions
    std::sort(directions.begin(), directions.end(), dirSortFcn);
  }
  else{
    // Do (essentialy) an S2 quadrature.. but with repeated directions

    // Compute x,y,z cosine values
    double mu  = cos(M_PI/4);
    double eta = sqrt(1-mu*mu) * cos(M_PI/4);
    double xi  = sqrt(1-mu*mu) * sin(M_PI/4);
    int d = 0;
    for(int octant = 0;octant < 8;++ octant){
      double omegas[3];
      omegas[0] = octant & 0x1;
      omegas[1] = (octant>>1) & 0x1;
      omegas[2] = (octant>>2) & 0x1;

      for(int sd=0; sd<num_directions_per_octant; sd++, d++){
        // Store which logical direction of travel we have
        directions[d].id = (omegas[0] > 0.) ? 1 : -1;
        directions[d].jd = (omegas[1] > 0.) ? 1 : -1;
        directions[d].kd = (omegas[2] > 0.) ? 1 : -1;

        // Store quadrature point's weight
        directions[d].w = 4.0*M_PI / (double)num_directions;
        directions[d].xcos = mu;
        directions[d].ycos = eta;
        directions[d].zcos = xi;
      }
    }
  }
}




