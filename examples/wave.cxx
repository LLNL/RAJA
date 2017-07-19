//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For additional details, please also read RAJA/LICENSE.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the disclaimer below.
//
// * Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the disclaimer (as noted below) in the
//   documentation and/or other materials provided with the distribution.
//
// * Neither the name of the LLNS/LLNL nor the names of its contributors may
//   be used to endorse or promote products derived from this software without
//   specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
// LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
// OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
// IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#include <cstdlib>
#include <iostream>
#include <cmath>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

#define sr 2 //Stencil Radius
#define PI 3.14159265359

double wave_sol(double t, double x, double y){
  return cos(2.*PI*t)*sin(2.*PI*x)*sin(2.*PI*y);
}

//P_3 = 2*P2 - P1 + laplace(P2)

//Assume periodic boundary conditions
//void cpp_solver(double *P1, double *P2, int nx, int ny,double dt,double dx,double cc){
void cpp_solver(double *P1, double *P2, int nx, double ct){
  
  //double coeff[3] = {1.0,-2.0,1.0}; //second order scheme
  double coeff[5] = {-1.0/12.0,4.0/3.0,-5.0/2.0,4.0/3.0,-1.0/12.0}; //fourth order scheme
  
  // loop over points
  for(int ty=0; ty<nx; ++ty){
    for(int tx=0; tx<nx; ++tx){

      const int id = tx + ty*nx;
      double P_old  = P1[id]; 
      double P_curr = P2[id]; 
      
      //Compute laplacian
      double lap = 0.0;

      for(int r=-sr; r<=sr; ++r){
        const int xi  = (tx+r+nx)%nx;
        const int idx = xi + nx*ty;
        lap += coeff[r+sr]*P2[idx]; 

        const int yi  = (ty+r+nx)%nx;
        const int idy = tx + nx*yi;
        lap += coeff[r+sr]*P2[idy];
      }

      P1[id] = 2*P_curr - P_old + ct*lap;
    }
  }
 
}



//Two dimensional solver for the acoustic wave eqation P_tt = cc(P_xx + P_yy)
//Second order in time and fourth order in space
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  double factor = 8.0;
  
  //Wave speed
  double cc = 1./2.0;

  //Domain details
  double xo = -1; 
  double dx = 0.1250/factor;
  int nx    = 16*factor;
  double T     = 0.82;
  int entries = nx*nx;

  //Storage for Approximated Solution
  double *P1 = new double[entries];
  double *P2 = new double[entries];

  
  //Time Stepping details
  double dt, nt;
  dt = 0.01*(dx/sqrt(cc));
  nt = ceil(T/dt);
  dt = T/nt;
  std::cout<<"nt: "<<nt<<std::endl;

  //poulate field
  int iter=0; 
  for(int ty=0; ty<nx; ty++){
    for(int tx=0; tx<nx; tx++){
      
      int id   = tx + nx*ty;
      double x = xo + tx*dx;
      double y = xo + ty*dx;
      
      P1[iter] = wave_sol(-dt,x,y);
      P2[iter] = wave_sol(0,x,y);
      iter++;
    }
  }
  

  //Evolve the solution
  double time = 0; 
  double ct = (cc*dt*dt)/(dx*dx);

  for(int k=0; k<nt; ++k){
    cpp_solver(P1,P2,nx,ct);
    time += dt; 
    
    double *Temp = P2;
    P2 = P1; 
    P1 = Temp;
  }

  std::cout<<"Evolved solution to time: "<<time<<std::endl;

  
  //Compute l_{inf} err
  double err=-1; 
  for(int ty=0; ty<nx; ty++){
    for(int tx=0; tx<nx; tx++){
      
      int id   = tx + nx*ty;
      double x = xo + tx*dx;
      double y = xo + ty*dx;
            
      double myErr = fabs(P2[id] - wave_sol(time,x,y));
      if(myErr > err) err = myErr; 
    }
  }

  std::cout<<"Err: "<<err<<" hx:"<<dx<<std::endl;  
  delete[] P1, P2; 


  return 0;
}

#if 0      
      for(int r = -sr; r<=sr; ++r){                
        int x   = (tx + r)%nx; 
        int id0 = x + ty*nx; 
        lap += ct*P2[id0];
        int y   = (ty + r)%nx; 
        int id1 = tx  + y%nx; 
        lap += ct*P2[id1];
      }
#endif
