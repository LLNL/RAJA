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

//Stencil Radius
#define sr 1
#define PI 3.14159265359

#if 0
for(int r = -sr; r<=sr; ++r){
  const int x   = (tx + r)%nx; 
  const int id0 = x + ty*nx; 
  lap += ct*coeff[sr+r]*P2[id0]; 
  
  const int y   = (ty + r)%nx; 
  const int id1 = tx + y*nx; 
  lap += ct*coeff[sr+r]*P2[id1];
 }
#endif

double wave_sol(double t, double x, double y){
  return cos(2*M_PI*t)*sin(2*M_PI*x)*sin(2*M_PI*y);
}

//P_3 = 2*P2 - P1 + laplace(P2)

//Assume periodic boundary conditions
void cpp_solver(double *P1, double *P2, int nx, int ny,double dt,double dx,double cc){
  
  //double coeff[3] = {-5.0/2.0,4.0/3.0,-1.0/12.0};
  //double coeff[3] = {1,-2,1};

  double ct = (cc*dt*dt)/(dx*dx);

  // loop over points
  for(int ty=0; ty<ny; ++ty){
    for(int tx=0; tx<nx; ++tx){

      const int id = tx + ty*nx;
      double P_old  = P1[id]; 
      double P_curr = P2[id]; 
      
      //Compute Stencil
      double lap = 0.0;

      //lap = ct*(P2[id-1] - 2*P2[id] + P2[id+1] + P2[id+nx] - 2*P2[id] + P2[id-nx]); 
      //lap += ct*(P2[id-1] + P2[id+1] + P2[id+nx] + P2[id-nx]);  
#if 0      
      int left  = ((tx-1) + ty*nx)%(nx*ny);
      int right = ((tx+1) + ty*nx)%(nx*ny);
      int up    = ((tx)   + (ty+1)*nx)%(nx*ny);
      int down  = ((tx)   + (ty-1)*nx)%(nx*ny);
#endif

      int left  = ( (tx-1)%nx) + ty*nx;
      int right = ( (tx+1+nx)%nx) + ty*nx;
      int up    = tx + ( (ty+1)%ny )*nx;
      int down  = tx + ( (ty-1+nx)%ny )*nx;



      lap = ct*(P2[left] -2*P2[id] + P2[right] + P2[up] -2*P2[id] + P2[down]);            
      
      P1[id] = 2*P_curr - P_old + lap; 

    }
  }
 
}



//Two dimensional solver for the acoustic wave eqation P_tt = cc(P_xx + P_yy)
//Second order in time and fourth order in space
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  double factor = 80;
  
  //Wave speed
  double cc = 1./2.0;

  //Domain details
  double xo = -1; 
  double dx = 0.1250/factor;
  int nx    = 16*factor;
  int nt    = 10;

  int entries = nx*nx;

  //Storage for Approximated Solution
  double *P1 = new double[entries];
  double *P2 = new double[entries];

  //Storage for Analytic Solution
  
  //Time Stepping details
  double dt = 0.01*(dx/sqrt(cc)); 

  //poulate field
  int iter=0; 
  for(int ty=0; ty<nx; ty++){
    for(int tx=0; tx<nx; tx++){
      
      int id   = tx + nx*ty;
      double x = xo + tx*dx;
      double y = xo + ty*dx;
      
      P1[iter] = wave_sol(0,x,y);
      P2[iter] = wave_sol(dt,x,y);
      iter++;
    }
  }
  

  //Evolve the solution
  double time = 0; 
  for(int k=0; k<nt; ++k){

    cpp_solver(P1,P2,nx,nx,dt,dx,cc);
    time += 2*dt; 

    double *Temp = P2;
    P2 = P1; 
    P1 = Temp;
  }

  std::cout<<"Evolved solution to time: "<<time<<std::endl;

  
  //Compute l_{inf} err
  double err=-1; 
  for(int ty=0; ty<nx; ty++){
    for(int tx=0; tx<nx; tx++){
      
      int id = tx + nx*ty;      
      double x = xo + tx*dx;
      double y = xo + ty*dx;
      
      double myErr = abs(P1[id] - wave_sol(time,x,y));
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
