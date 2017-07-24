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
#define CUDA_BLOCK_SIZE 16


typedef struct grid{
  double ox,dx;
  int nx;
}grid_s;


template <typename T>
T* allocate(RAJA::Index_type size) {
  T* ptr;
#if defined(RAJA_ENABLE_CUDA)
  cudaMallocManaged((void**)&ptr, sizeof(T)*size,cudaMemAttachGlobal);
#else
  ptr = new T[size];
#endif
  return ptr;
}

template <typename T>
void deallocate(T* &ptr) {
  if (ptr) {
#if defined(RAJA_ENABLE_CUDA)
    cudaFree(ptr);
#else
    delete[] ptr;
#endif
    ptr = nullptr;
  }
}



struct DefaultCPP{};

// Generic RAJA implementation for CPU                                                                                       
template <typename Policy1, typename Policy2>
struct Wave {
  template <typename T>  
    void operator()(T* P1, T* P2, int nx, double ct) {

    RAJA::forallN< RAJA::NestedPolicy<
    RAJA::ExecList<RAJA::seq_exec,RAJA::seq_exec >>>(
    RAJA::RangeSegment(0, nx),
    RAJA::RangeSegment(0, nx),
    [=](int ty, int tx) {
    
    double coeff[5] = {-1.0/12.0,4.0/3.0,-5.0/2.0,4.0/3.0,-1.0/12.0}; //fourth order scheme
    
    // loop over points        
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
    
    //finite difference stencil
    P1[id] = 2*P_curr - P_old + ct*lap;
          
    }); 
            
  }
};


// Generic CPP implementation                                                                                                
template <>
struct Wave<DefaultCPP, DefaultCPP> {
  template <typename T>
  void operator()(T* P1, T* P2, int nx, double ct) {

    double coeff[5] = {-1.0/12.0,4.0/3.0,-5.0/2.0,4.0/3.0,-1.0/12.0}; //fourth order scheme
    
    // loop over points
    for(int ty=0; ty<nx; ++ty){
      for(int tx=0; tx<nx; ++tx){
        
        const int id = tx + ty*nx;
        T P_old  = P1[id]; 
        T P_curr = P2[id]; 
        
        //Compute laplacian
        T lap = 0.0;
        
        for(int r=-sr; r<=sr; ++r){
          const int xi  = (tx+r+nx)%nx;
          const int idx = xi + nx*ty;
          lap += coeff[r+sr]*P2[idx]; 
          
          const int yi  = (ty+r+nx)%nx;
          const int idy = tx + nx*yi;
          lap += coeff[r+sr]*P2[idy];
        }
        
        //finite difference stencil
        P1[id] = 2*P_curr - P_old + ct*lap;
      }
    }
    
  }

};

// RAJA implementation for CUDA                                                                                              
template <size_t BLOCK_SIZE>
struct Wave<RAJA::cuda_exec<BLOCK_SIZE>, RAJA::cuda_exec<BLOCK_SIZE> > {
  template <typename T>
    void operator()(T* P1, T* P2, int nx, double ct){

    RAJA::forallN<RAJA::NestedPolicy<
    RAJA::ExecList<
    RAJA::cuda_threadblock_y_exec<BLOCK_SIZE>,
    RAJA::cuda_threadblock_x_exec<BLOCK_SIZE> >>>(
    RAJA::RangeSegment(0, nx), RAJA::RangeSegment(0, nx), [=] __device__ (int tidy, int tidx) {
                                             
    double coeff[5] = {-1.0/12.0,4.0/3.0,-5.0/2.0,4.0/3.0,-1.0/12.0}; //fourth order scheme
    
    // loop over points        
    const int id = tidx + tidy*nx;
    T P_old  = P1[id]; 
    T P_curr = P2[id]; 
    
    //Compute laplacian
    T lap = 0.0;
    
    for(int r=-sr; r<=sr; ++r){
      const int xi  = (tidx+r+nx)%nx;
      const int idx = xi + nx*tidy;
      lap += coeff[r+sr]*P2[idx]; 
      
      const int yi  = (tidy+r+nx)%nx;
      const int idy = tidx + nx*yi;
      lap += coeff[r+sr]*P2[idy];
    }
    
    //finite difference stencil
    P1[id] = 2*P_curr - P_old + ct*lap;
          
  }); 
    

  }
};


double wave_sol(double t, double x, double y);
void set_ic(double *P1, double *P2, double t0, double t1, grid_s grid);
void compute_err(double *P, double tf, grid_s grid);

/* Example 4: Two dimensional solver for the acoustic wave eqation P_tt = c^2(P_xx + P_yy)
   Scheme uses a second order spatial discretization for P_tt and a
   fourth order spatial discretization in space. Periodic boundary conditions are assumed.
 */

//
//Second order in time and fourth order in space
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  std::cout<<"Example 4. Time-Domain Finite Difference Solver For The Acoustic Wave Equation"<<std::endl;
  
  //Wave speed squared
  double cc = 1./2.0;

  //Multiplier for spatial discretization
  double factor = 8.0;

  //Discretization of the domain. 
  //The same discretization of the x-dimension wil be used for the y-dimension
  grid_s grid; 
  grid.ox   = -1; 
  grid.dx   = 0.1250/factor;
  grid.nx      = 16*factor;
  double T    = 0.82;
  int entries = grid.nx*grid.nx;

  double *P1 = allocate<double>(entries);
  double *P2 = allocate<double>(entries);
  
  //Parameters for time stepping
  double dt, nt,time, ct;
  dt = 0.01*(grid.dx/sqrt(cc));  //intial step size, we must obey CFL condition
  nt = ceil(T/dt); //total number of time steps
  dt = T/nt; //final time-step 
  ct = (cc*dt*dt)/(grid.dx*grid.dx); //Merge coefficients to a single coefficient

  
  //----[C++-Loop Style]------
  std::cout<<"C++-Style Loop"<<std::endl;
  time = 0; 
  set_ic(P1,P2,(time-dt),time,grid);
  for(int k=0; k<nt; ++k){
    
    Wave<DefaultCPP, DefaultCPP>()(P1,P2,grid.nx,ct);
    time += dt; 

    double *Temp = P2;
    P2 = P1; 
    P1 = Temp;
  }  
  compute_err(P2, time, grid);
  std::cout<<"Evolved solution to time: "<<time<<"\n \n"<<std::endl;
  //========================

  //---[RAJA-Style Sequential Policy]--
  std::cout<<"RAJA-Style Sequential Loop"<<std::endl;
  time = 0; 
  set_ic(P1,P2,(time-dt),time,grid);
  for(int k=0; k<nt; ++k){
    
    Wave<RAJA::seq_exec, RAJA::seq_exec>()(P1,P2,grid.nx,ct);
    time += dt; 
    
    double *Temp = P2;
    P2 = P1; 
    P1 = Temp;
  }  
  compute_err(P2, time, grid);
  std::cout<<"Evolved solution to time: "<<time<<"\n \n"<<std::endl;
  //===================================
  
  //---[RAJA-Style Omp/Seq Policy]--
  std::cout<<"RAJA-Style OMP/Seq Loop"<<std::endl;
  time = 0; 
  set_ic(P1,P2,(time-dt),time,grid);
  for(int k=0; k<nt; ++k){
    
    Wave<RAJA::omp_parallel_for_exec, RAJA::seq_exec>()(P1,P2,grid.nx,ct);
    time += dt; 
    
    double *Temp = P2;
    P2 = P1; 
    P1 = Temp;
  }  
  compute_err(P2, time, grid);
  std::cout<<"Evolved solution to time: "<<time<<"\n \n"<<std::endl;
  //===================================

  //---[RAJA-Style CUDA Policy]--
  std::cout<<"RAJA-Style CUDA Loop"<<std::endl;
  time = 0; 

  time = 0; 
  set_ic(P1,P2,(time-dt),time,grid);
  for(int k=0; k<nt; ++k){
    
    Wave<RAJA::cuda_exec<CUDA_BLOCK_SIZE>,RAJA::cuda_exec<CUDA_BLOCK_SIZE> >()(P1,P2,grid.nx,ct);
    time += dt; 
    
    double *Temp = P2;
    P2 = P1; 
    P1 = Temp;
  }  
  cudaDeviceSynchronize();
  compute_err(P2, time, grid); 
  std::cout<<"Evolved solution to time: "<<time<<"\n \n"<<std::endl;
  //===================================

  deallocate(P1);
  deallocate(P2);

  return 0;
}


//Analytic solution is assumed to be 
//P(t,x,y) = Cos(2*PI*t)*Sin(2*PI*x)*Sin(2*PI*y)
double wave_sol(double t, double x, double y){
  return cos(2.*PI*t)*sin(2.*PI*x)*sin(2.*PI*y);
}

//Compute ||P_{analytic} - P_{approx}||_{inf} 
void compute_err(double *P, double tf, grid_s grid){

  double err=-1; 
  for(int ty=0; ty<grid.nx; ty++){
    for(int tx=0; tx<grid.nx; tx++){
      
      int id   = tx + grid.nx*ty;
      double x = grid.ox + tx*grid.dx;
      double y = grid.ox + ty*grid.dx;
            
      double myErr = fabs(P[id] - wave_sol(tf,x,y));
      if(myErr > err) err = myErr; 
    }
  }
  std::cout<<"Max err: "<<err<<" hx: "<<grid.dx<<std::endl;
}


//Set initial condition. 
void set_ic(double *P1, double *P2, double t0, double t1, grid_s grid){

  //poulate field
  int iter=0; 
  for(int ty=0; ty<grid.nx; ty++){
    for(int tx=0; tx<grid.nx; tx++){
      
      double x = grid.ox + tx*grid.dx;
      double y = grid.ox + ty*grid.dx;
      
      P1[iter] = wave_sol(-t0,x,y);
      P2[iter] = wave_sol(t1,x,y);
      iter++;
    }
  }

}
