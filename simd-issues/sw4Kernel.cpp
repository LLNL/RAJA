//  SW4 LICENSE
// # ----------------------------------------------------------------------
// # SW4 - Seismic Waves, 4th order
// # ----------------------------------------------------------------------
// # Copyright (c) 2013, Lawrence Livermore National Security, LLC. 
// # Produced at the Lawrence Livermore National Laboratory. 
// # 
// # Written by:
// # N. Anders Petersson (petersson1@llnl.gov)
// # Bjorn Sjogreen      (sjogreen2@llnl.gov)
// # 
// # LLNL-CODE-643337 
// # 
// # All rights reserved. 
// # 
// # This file is part of SW4, Version: 1.0
// # 
// # Please also read LICENCE.txt, which contains "Our Notice and GNU General Public License"
// # 
// # This program is free software; you can redistribute it and/or modify
// # it under the terms of the GNU General Public License (as published by
// # the Free Software Foundation) version 2, dated June 1991. 
// # 
// # This program is distributed in the hope that it will be useful, but
// # WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF
// # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the terms and
// # conditions of the GNU General Public License for more details. 
// # 
// # You should have received a copy of the GNU General Public License
// # along with this program; if not, write to the Free Software
// # Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307, USA


#include "RAJA/RAJA.hpp"
#include <cstdio>
#include <time.h>       /* time */
#include <cstdlib>
#include <iostream>
#include <chrono>
#include <cmath>

#define float_sw4 double

/// - Generates incorrect ouput
using POL = 
   RAJA::KernelPolicy<
    RAJA::statement::For<2, RAJA::omp_parallel_for_exec,
    RAJA::statement::For<1, RAJA::loop_exec,
    RAJA::statement::For<0, RAJA::simd_exec,
    RAJA::statement::Lambda<0> > > > >;


//using POL =  //Produces incorrect output
//RAJA::KernelPolicy<
//RAJA::statement::For<2, RAJA::omp_parallel_for_exec,
//    RAJA::statement::For<1, RAJA::loop_exec,
//    RAJA::statement::For<0, RAJA::loop_exec,
//    RAJA::statement::Lambda<0> > > > >;


#define KERNEL //produces incorrect ouput
//#undef KERNEL //traditional nesting - produces correct ouput 

using POL2 = RAJA::omp_parallel_for_exec;
using POL1 = RAJA::loop_exec;
using POL0 = RAJA::simd_exec;

#define mu(i,j,k)     a_mu[base+i+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+i+ni*(j)+nij*(k)]

// Reversed indexation                                                                                            
#define u(c,i,j,k)   a_u[base3+i+ni*(j)+nij*(k)+nijk*(c)]
#define lu(c,i,j,k) a_lu[base3+i+ni*(j)+nij*(k)+nijk*(c)]
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define strz(k) a_strz[k-kfirst0]
#define acof(i,j,k) a_acof[(i-1)+6*(j-1)+48*(k-1)]
#define bope(i,j) a_bope[i-1+6*(j-1)]
#define ghcof(i) a_ghcof[i-1]


void rhs4sg_rev_native( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                        int nk, float_sw4* __restrict__ a_acof, float_sw4 *__restrict__ a_bope,
                        float_sw4* __restrict__ a_ghcof, float_sw4* __restrict__ a_lu, float_sw4* __restrict__ a_u,
                        float_sw4* __restrict__ a_mu, float_sw4* __restrict__ a_lambda, 
                        float_sw4 h, float_sw4* __restrict__ a_strx, float_sw4* __restrict__ a_stry, 
                        float_sw4* __restrict__ a_strz );


void rhs4sg_rev_raja( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
                      int nk, float_sw4* __restrict__ a_acof, float_sw4 *__restrict__ a_bope,
                      float_sw4* __restrict__ a_ghcof, float_sw4* __restrict__ a_lu, float_sw4* __restrict__ a_u,
                      float_sw4* __restrict__ a_mu, float_sw4* __restrict__ a_lambda, 
                      float_sw4 h, float_sw4* __restrict__ a_strx, float_sw4* __restrict__ a_stry, 
                      float_sw4* __restrict__ a_strz );





int main(int argc, char *argv[])
{

   long int kfirst=-1, klast=103;
   long int jfirst=-1, jlast=203;
   long int ifirst=-1, ilast=203;
   double   h  = 0.04;
   long int nk = 101;

   long int k_len = klast-kfirst + 10;
   long int j_len = jlast-jfirst + 10;
   long int i_len = ilast-ifirst + 10;
   long int arr_len = k_len*j_len*i_len;

   //Allocate arrays
   //long int dim = 3;
   //double * a_lu_native  = new double[dim*arr_len]; //ref solution
   //double * a_lu_raja    = new double[dim*arr_len]; //SIMD solution

   size_t lu_arraySz = 13153412 + 10;  //padding
   double * a_lu_native  = new double[lu_arraySz]; //ref solution
   double * a_lu_raja    = new double[lu_arraySz]; //SIMD solution
   
   double * a_acof       = new double[arr_len];
   double * a_bope       = new double[arr_len];
   double * a_ghcof      = new double[arr_len];
   double * a_u          = new double[arr_len];
   double * a_mu         = new double[arr_len];
   double * a_lambda     = new double[arr_len];
   double * a_strx       = new double[arr_len];
   double * a_stry       = new double[arr_len];
   double * a_strz       = new double[arr_len];
   for(auto i=0; i<lu_arraySz; ++i)
      {
         a_lu_native[i]  = 0.0; //output for the native version
         a_lu_raja[i]    = 0.0; //output for the raja version
      }

   for(auto i=0; i<arr_len; ++i)
      {
         a_acof[i]  = rand() % 10 + 1;
         a_bope[i]    = rand() % 10 + 1;
         a_ghcof[i]   = rand() % 10 + 1;
         a_u[i]       = rand() % 10 + 1;
         a_mu[i]      = rand() % 10 + 1;
         a_lambda[i]  = rand() % 10 + 1;
         a_strx[i]    = rand() % 10 + 1;
         a_stry[i]    = rand() % 10 + 1;
         a_strz[i]    = rand() % 10 + 1;
      }
   
   
   std::cout<<"Calling native version..."<<std::endl;
   //Call native version
   rhs4sg_rev_native(ifirst, ilast, jfirst, jlast, kfirst, klast,
                     nk, a_acof, a_bope, a_ghcof, a_lu_native, a_u,
                     a_mu, a_lambda, h, a_strx, a_stry, a_strz );

   

   std::cout<<"Calling RAJA version..."<<std::endl;
   //Call RAJA version
   rhs4sg_rev_raja(ifirst, ilast, jfirst, jlast, kfirst, klast,
                   nk, a_acof, a_bope, a_ghcof, a_lu_raja, a_u,
                   a_mu, a_lambda, h, a_strx, a_stry, a_strz );


   bool pass = true;

   for(auto i=0; i<lu_arraySz; ++i)
      {
        double err = std::abs(a_lu_native[i]-a_lu_raja[i]);
         if(err > 1e-8){
            pass = false; 
         }
      }

   if(pass)
      {
         std::cout<<"RAJA and native versions produced the same output"<<std::endl;
      }else
      {
         std::cout<<"RAJA and native versions did NOT produce the same output"<<std::endl;
      }

   delete[] a_lu_native;
   delete[] a_lu_raja;
   
   delete[] a_acof;
   delete[] a_bope;
   delete[] a_ghcof;
   delete[] a_u;
   delete[] a_mu;
   delete[] a_lambda;
   delete[] a_strx;
   delete[] a_stry;
   delete[] a_strz;

   return 0;
}

void rhs4sg_rev_native( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		 int nk, float_sw4* __restrict__ a_acof, float_sw4 *__restrict__ a_bope,
		 float_sw4* __restrict__ a_ghcof, float_sw4* __restrict__ a_lu, float_sw4* __restrict__ a_u,
		 float_sw4* __restrict__ a_mu, float_sw4* __restrict__ a_lambda, 
		 float_sw4 h, float_sw4* __restrict__ a_strx, float_sw4* __restrict__ a_stry, 
		 float_sw4* __restrict__ a_strz )
{//raja fun start


 // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+i+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+i+ni*(j)+nij*(k)]
   // Reversed indexation
#define u(c,i,j,k)   a_u[base3+i+ni*(j)+nij*(k)+nijk*(c)]   
#define lu(c,i,j,k) a_lu[base3+i+ni*(j)+nij*(k)+nijk*(c)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define strz(k) a_strz[k-kfirst0]
#define acof(i,j,k) a_acof[(i-1)+6*(j-1)+48*(k-1)]
#define bope(i,j) a_bope[i-1+6*(j-1)]
#define ghcof(i) a_ghcof[i-1]
   
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 i12  = 1.0/12;
   const float_sw4 i144 = 1.0/144;
   const float_sw4 tf   = 0.75;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = base-nijk;

   const int nic  = 3*ni;
   const int nijc = 3*nij;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;


   const float_sw4 cof = 1.0/(h*h);
   
   int k1 = 7; 
   int k2 = 101;   
   RAJA::RangeSegment k_range(k1,k2+1);
   RAJA::RangeSegment j_range(jfirst+2,jlast-1);
   RAJA::RangeSegment i_range(ifirst+2,ilast-1);

#pragma omp parallel for                                                                                             
   for(int  k= k1; k <= k2 ; k++ ){
      for(int j=jfirst+2; j <= jlast-2 ; j++ ) {
#pragma omp simd                                                                      
#pragma ivdep                                                                                                     
         //#pragma forceinline recursive 
         for(int i=ifirst+2; i <= ilast-2 ; i++ )
            {

      float_sw4 mux1,mux2,mux3,mux4,muy1,muy2,muy3,muy4;
      float_sw4 r1,r2,r3,mucof,mu1zz,mu2zz,mu3zz,lap2mu,q,u3zip2,u3zip1;
      float_sw4 u3zim1,u3zim2,lau3zx,mu3xz,u3zjp2,u3zjp1,u3zjm1,u3zjm2,lau3zy;
      float_sw4 mu3yz,mu1zx,u1zip2,u1zip1,u1zim1,u1zim2;
      float_sw4 u2zjp2,u2zjp1,u2zjm1,u2zjm2,mu2zy,lau1xz,lau2yz,muz1,muz2,muz3,muz4;
      
/* from inner_loop_4a, 28x3 = 84 ops */
            mux1 = mu(i-1,j,k)*strx(i-1)-
	       tf*(mu(i,j,k)*strx(i)+mu(i-2,j,k)*strx(i-2));
            mux2 = mu(i-2,j,k)*strx(i-2)+mu(i+1,j,k)*strx(i+1)+
	       3*(mu(i,j,k)*strx(i)+mu(i-1,j,k)*strx(i-1));
            mux3 = mu(i-1,j,k)*strx(i-1)+mu(i+2,j,k)*strx(i+2)+
	       3*(mu(i+1,j,k)*strx(i+1)+mu(i,j,k)*strx(i));
            mux4 = mu(i+1,j,k)*strx(i+1)-
	       tf*(mu(i,j,k)*strx(i)+mu(i+2,j,k)*strx(i+2));

            muy1 = mu(i,j-1,k)*stry(j-1)-
	       tf*(mu(i,j,k)*stry(j)+mu(i,j-2,k)*stry(j-2));
            muy2 = mu(i,j-2,k)*stry(j-2)+mu(i,j+1,k)*stry(j+1)+
	       3*(mu(i,j,k)*stry(j)+mu(i,j-1,k)*stry(j-1));
            muy3 = mu(i,j-1,k)*stry(j-1)+mu(i,j+2,k)*stry(j+2)+
	       3*(mu(i,j+1,k)*stry(j+1)+mu(i,j,k)*stry(j));
            muy4 = mu(i,j+1,k)*stry(j+1)-
	       tf*(mu(i,j,k)*stry(j)+mu(i,j+2,k)*stry(j+2));

            muz1 = mu(i,j,k-1)*strz(k-1)-
	       tf*(mu(i,j,k)*strz(k)+mu(i,j,k-2)*strz(k-2));
            muz2 = mu(i,j,k-2)*strz(k-2)+mu(i,j,k+1)*strz(k+1)+
	       3*(mu(i,j,k)*strz(k)+mu(i,j,k-1)*strz(k-1));
            muz3 = mu(i,j,k-1)*strz(k-1)+mu(i,j,k+2)*strz(k+2)+
	       3*(mu(i,j,k+1)*strz(k+1)+mu(i,j,k)*strz(k));
            muz4 = mu(i,j,k+1)*strz(k+1)-
	       tf*(mu(i,j,k)*strz(k)+mu(i,j,k+2)*strz(k+2));
/* xx, yy, and zz derivatives:*/
/* 75 ops */
            r1 = i6*( strx(i)*( (2*mux1+la(i-1,j,k)*strx(i-1)-
               tf*(la(i,j,k)*strx(i)+la(i-2,j,k)*strx(i-2)))*
                              (u(1,i-2,j,k)-u(1,i,j,k))+
           (2*mux2+la(i-2,j,k)*strx(i-2)+la(i+1,j,k)*strx(i+1)+
                3*(la(i,j,k)*strx(i)+la(i-1,j,k)*strx(i-1)))*
                              (u(1,i-1,j,k)-u(1,i,j,k))+ 
           (2*mux3+la(i-1,j,k)*strx(i-1)+la(i+2,j,k)*strx(i+2)+
                3*(la(i+1,j,k)*strx(i+1)+la(i,j,k)*strx(i)))*
                              (u(1,i+1,j,k)-u(1,i,j,k))+
                (2*mux4+ la(i+1,j,k)*strx(i+1)-
               tf*(la(i,j,k)*strx(i)+la(i+2,j,k)*strx(i+2)))*
                (u(1,i+2,j,k)-u(1,i,j,k)) ) + stry(j)*(
                     muy1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                     muy2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                     muy3*(u(1,i,j+1,k)-u(1,i,j,k)) +
                     muy4*(u(1,i,j+2,k)-u(1,i,j,k)) ) + strz(k)*(
                     muz1*(u(1,i,j,k-2)-u(1,i,j,k)) + 
                     muz2*(u(1,i,j,k-1)-u(1,i,j,k)) + 
                     muz3*(u(1,i,j,k+1)-u(1,i,j,k)) +
                     muz4*(u(1,i,j,k+2)-u(1,i,j,k)) ) );

/* 75 ops */
            r2 = i6*( strx(i)*(mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                      mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                      mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                      mux4*(u(2,i+2,j,k)-u(2,i,j,k)) ) + stry(j)*(
                  (2*muy1+la(i,j-1,k)*stry(j-1)-
                      tf*(la(i,j,k)*stry(j)+la(i,j-2,k)*stry(j-2)))*
                          (u(2,i,j-2,k)-u(2,i,j,k))+
           (2*muy2+la(i,j-2,k)*stry(j-2)+la(i,j+1,k)*stry(j+1)+
                     3*(la(i,j,k)*stry(j)+la(i,j-1,k)*stry(j-1)))*
                          (u(2,i,j-1,k)-u(2,i,j,k))+ 
           (2*muy3+la(i,j-1,k)*stry(j-1)+la(i,j+2,k)*stry(j+2)+
                     3*(la(i,j+1,k)*stry(j+1)+la(i,j,k)*stry(j)))*
                          (u(2,i,j+1,k)-u(2,i,j,k))+
                  (2*muy4+la(i,j+1,k)*stry(j+1)-
                    tf*(la(i,j,k)*stry(j)+la(i,j+2,k)*stry(j+2)))*
                          (u(2,i,j+2,k)-u(2,i,j,k)) ) + strz(k)*(
                     muz1*(u(2,i,j,k-2)-u(2,i,j,k)) + 
                     muz2*(u(2,i,j,k-1)-u(2,i,j,k)) + 
                     muz3*(u(2,i,j,k+1)-u(2,i,j,k)) +
                     muz4*(u(2,i,j,k+2)-u(2,i,j,k)) ) );

/* 75 ops */
            r3 = i6*( strx(i)*(mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                      mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                      mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                      mux4*(u(3,i+2,j,k)-u(3,i,j,k))  ) + stry(j)*(
                     muy1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                     muy2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                     muy3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                     muy4*(u(3,i,j+2,k)-u(3,i,j,k)) ) + strz(k)*(
                  (2*muz1+la(i,j,k-1)*strz(k-1)-
                      tf*(la(i,j,k)*strz(k)+la(i,j,k-2)*strz(k-2)))*
                          (u(3,i,j,k-2)-u(3,i,j,k))+
           (2*muz2+la(i,j,k-2)*strz(k-2)+la(i,j,k+1)*strz(k+1)+
                      3*(la(i,j,k)*strz(k)+la(i,j,k-1)*strz(k-1)))*
                          (u(3,i,j,k-1)-u(3,i,j,k))+ 
           (2*muz3+la(i,j,k-1)*strz(k-1)+la(i,j,k+2)*strz(k+2)+
                      3*(la(i,j,k+1)*strz(k+1)+la(i,j,k)*strz(k)))*
                          (u(3,i,j,k+1)-u(3,i,j,k))+
                  (2*muz4+la(i,j,k+1)*strz(k+1)-
                    tf*(la(i,j,k)*strz(k)+la(i,j,k+2)*strz(k+2)))*
		  (u(3,i,j,k+2)-u(3,i,j,k)) ) );


/* Mixed derivatives: */
/* 29ops /mixed derivative */
/* 116 ops for r1 */
/*   (la*v_y)_x */
            r1 = r1 + strx(i)*stry(j)*
                 i144*( la(i-2,j,k)*(u(2,i-2,j-2,k)-u(2,i-2,j+2,k)+
                             8*(-u(2,i-2,j-1,k)+u(2,i-2,j+1,k))) - 8*(
                        la(i-1,j,k)*(u(2,i-1,j-2,k)-u(2,i-1,j+2,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i-1,j+1,k))) )+8*(
                        la(i+1,j,k)*(u(2,i+1,j-2,k)-u(2,i+1,j+2,k)+
                             8*(-u(2,i+1,j-1,k)+u(2,i+1,j+1,k))) ) - (
                        la(i+2,j,k)*(u(2,i+2,j-2,k)-u(2,i+2,j+2,k)+
                             8*(-u(2,i+2,j-1,k)+u(2,i+2,j+1,k))) )) 
/*   (la*w_z)_x */
               + strx(i)*strz(k)*       
                 i144*( la(i-2,j,k)*(u(3,i-2,j,k-2)-u(3,i-2,j,k+2)+
                             8*(-u(3,i-2,j,k-1)+u(3,i-2,j,k+1))) - 8*(
                        la(i-1,j,k)*(u(3,i-1,j,k-2)-u(3,i-1,j,k+2)+
                             8*(-u(3,i-1,j,k-1)+u(3,i-1,j,k+1))) )+8*(
                        la(i+1,j,k)*(u(3,i+1,j,k-2)-u(3,i+1,j,k+2)+
                             8*(-u(3,i+1,j,k-1)+u(3,i+1,j,k+1))) ) - (
                        la(i+2,j,k)*(u(3,i+2,j,k-2)-u(3,i+2,j,k+2)+
                             8*(-u(3,i+2,j,k-1)+u(3,i+2,j,k+1))) )) 
/*   (mu*v_x)_y */
               + strx(i)*stry(j)*       
                 i144*( mu(i,j-2,k)*(u(2,i-2,j-2,k)-u(2,i+2,j-2,k)+
                             8*(-u(2,i-1,j-2,k)+u(2,i+1,j-2,k))) - 8*(
                        mu(i,j-1,k)*(u(2,i-2,j-1,k)-u(2,i+2,j-1,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i+1,j-1,k))) )+8*(
                        mu(i,j+1,k)*(u(2,i-2,j+1,k)-u(2,i+2,j+1,k)+
                             8*(-u(2,i-1,j+1,k)+u(2,i+1,j+1,k))) ) - (
                        mu(i,j+2,k)*(u(2,i-2,j+2,k)-u(2,i+2,j+2,k)+
                             8*(-u(2,i-1,j+2,k)+u(2,i+1,j+2,k))) )) 
/*   (mu*w_x)_z */
               + strx(i)*strz(k)*       
                 i144*( mu(i,j,k-2)*(u(3,i-2,j,k-2)-u(3,i+2,j,k-2)+
                             8*(-u(3,i-1,j,k-2)+u(3,i+1,j,k-2))) - 8*(
                        mu(i,j,k-1)*(u(3,i-2,j,k-1)-u(3,i+2,j,k-1)+
                             8*(-u(3,i-1,j,k-1)+u(3,i+1,j,k-1))) )+8*(
                        mu(i,j,k+1)*(u(3,i-2,j,k+1)-u(3,i+2,j,k+1)+
                             8*(-u(3,i-1,j,k+1)+u(3,i+1,j,k+1))) ) - (
                        mu(i,j,k+2)*(u(3,i-2,j,k+2)-u(3,i+2,j,k+2)+
				     8*(-u(3,i-1,j,k+2)+u(3,i+1,j,k+2))) )) ;

/* 116 ops for r2 */
/*   (mu*u_y)_x */
            r2 = r2 + strx(i)*stry(j)*
                 i144*( mu(i-2,j,k)*(u(1,i-2,j-2,k)-u(1,i-2,j+2,k)+
                             8*(-u(1,i-2,j-1,k)+u(1,i-2,j+1,k))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j-2,k)-u(1,i-1,j+2,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i-1,j+1,k))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j-2,k)-u(1,i+1,j+2,k)+
                             8*(-u(1,i+1,j-1,k)+u(1,i+1,j+1,k))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j-2,k)-u(1,i+2,j+2,k)+
                             8*(-u(1,i+2,j-1,k)+u(1,i+2,j+1,k))) )) 
/* (la*u_x)_y */
              + strx(i)*stry(j)*
                 i144*( la(i,j-2,k)*(u(1,i-2,j-2,k)-u(1,i+2,j-2,k)+
                             8*(-u(1,i-1,j-2,k)+u(1,i+1,j-2,k))) - 8*(
                        la(i,j-1,k)*(u(1,i-2,j-1,k)-u(1,i+2,j-1,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i+1,j-1,k))) )+8*(
                        la(i,j+1,k)*(u(1,i-2,j+1,k)-u(1,i+2,j+1,k)+
                             8*(-u(1,i-1,j+1,k)+u(1,i+1,j+1,k))) ) - (
                        la(i,j+2,k)*(u(1,i-2,j+2,k)-u(1,i+2,j+2,k)+
                             8*(-u(1,i-1,j+2,k)+u(1,i+1,j+2,k))) )) 
/* (la*w_z)_y */
               + stry(j)*strz(k)*
                 i144*( la(i,j-2,k)*(u(3,i,j-2,k-2)-u(3,i,j-2,k+2)+
                             8*(-u(3,i,j-2,k-1)+u(3,i,j-2,k+1))) - 8*(
                        la(i,j-1,k)*(u(3,i,j-1,k-2)-u(3,i,j-1,k+2)+
                             8*(-u(3,i,j-1,k-1)+u(3,i,j-1,k+1))) )+8*(
                        la(i,j+1,k)*(u(3,i,j+1,k-2)-u(3,i,j+1,k+2)+
                             8*(-u(3,i,j+1,k-1)+u(3,i,j+1,k+1))) ) - (
                        la(i,j+2,k)*(u(3,i,j+2,k-2)-u(3,i,j+2,k+2)+
                             8*(-u(3,i,j+2,k-1)+u(3,i,j+2,k+1))) ))
/* (mu*w_y)_z */
               + stry(j)*strz(k)*
                 i144*( mu(i,j,k-2)*(u(3,i,j-2,k-2)-u(3,i,j+2,k-2)+
                             8*(-u(3,i,j-1,k-2)+u(3,i,j+1,k-2))) - 8*(
                        mu(i,j,k-1)*(u(3,i,j-2,k-1)-u(3,i,j+2,k-1)+
                             8*(-u(3,i,j-1,k-1)+u(3,i,j+1,k-1))) )+8*(
                        mu(i,j,k+1)*(u(3,i,j-2,k+1)-u(3,i,j+2,k+1)+
                             8*(-u(3,i,j-1,k+1)+u(3,i,j+1,k+1))) ) - (
                        mu(i,j,k+2)*(u(3,i,j-2,k+2)-u(3,i,j+2,k+2)+
				     8*(-u(3,i,j-1,k+2)+u(3,i,j+1,k+2))) )) ;
/* 116 ops for r3 */
/*  (mu*u_z)_x */
            r3 = r3 + strx(i)*strz(k)*
                 i144*( mu(i-2,j,k)*(u(1,i-2,j,k-2)-u(1,i-2,j,k+2)+
                             8*(-u(1,i-2,j,k-1)+u(1,i-2,j,k+1))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j,k-2)-u(1,i-1,j,k+2)+
                             8*(-u(1,i-1,j,k-1)+u(1,i-1,j,k+1))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j,k-2)-u(1,i+1,j,k+2)+
                             8*(-u(1,i+1,j,k-1)+u(1,i+1,j,k+1))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j,k-2)-u(1,i+2,j,k+2)+
                             8*(-u(1,i+2,j,k-1)+u(1,i+2,j,k+1))) )) 
/* (mu*v_z)_y */
              + stry(j)*strz(k)*
                 i144*( mu(i,j-2,k)*(u(2,i,j-2,k-2)-u(2,i,j-2,k+2)+
                             8*(-u(2,i,j-2,k-1)+u(2,i,j-2,k+1))) - 8*(
                        mu(i,j-1,k)*(u(2,i,j-1,k-2)-u(2,i,j-1,k+2)+
                             8*(-u(2,i,j-1,k-1)+u(2,i,j-1,k+1))) )+8*(
                        mu(i,j+1,k)*(u(2,i,j+1,k-2)-u(2,i,j+1,k+2)+
                             8*(-u(2,i,j+1,k-1)+u(2,i,j+1,k+1))) ) - (
                        mu(i,j+2,k)*(u(2,i,j+2,k-2)-u(2,i,j+2,k+2)+
                             8*(-u(2,i,j+2,k-1)+u(2,i,j+2,k+1))) ))
/*   (la*u_x)_z */
              + strx(i)*strz(k)*
                 i144*( la(i,j,k-2)*(u(1,i-2,j,k-2)-u(1,i+2,j,k-2)+
                             8*(-u(1,i-1,j,k-2)+u(1,i+1,j,k-2))) - 8*(
                        la(i,j,k-1)*(u(1,i-2,j,k-1)-u(1,i+2,j,k-1)+
                             8*(-u(1,i-1,j,k-1)+u(1,i+1,j,k-1))) )+8*(
                        la(i,j,k+1)*(u(1,i-2,j,k+1)-u(1,i+2,j,k+1)+
                             8*(-u(1,i-1,j,k+1)+u(1,i+1,j,k+1))) ) - (
                        la(i,j,k+2)*(u(1,i-2,j,k+2)-u(1,i+2,j,k+2)+
                             8*(-u(1,i-1,j,k+2)+u(1,i+1,j,k+2))) )) 
/* (la*v_y)_z */
              + stry(j)*strz(k)*
                 i144*( la(i,j,k-2)*(u(2,i,j-2,k-2)-u(2,i,j+2,k-2)+
                             8*(-u(2,i,j-1,k-2)+u(2,i,j+1,k-2))) - 8*(
                        la(i,j,k-1)*(u(2,i,j-2,k-1)-u(2,i,j+2,k-1)+
                             8*(-u(2,i,j-1,k-1)+u(2,i,j+1,k-1))) )+8*(
                        la(i,j,k+1)*(u(2,i,j-2,k+1)-u(2,i,j+2,k+1)+
                             8*(-u(2,i,j-1,k+1)+u(2,i,j+1,k+1))) ) - (
                        la(i,j,k+2)*(u(2,i,j-2,k+2)-u(2,i,j+2,k+2)+
				     8*(-u(2,i,j-1,k+2)+u(2,i,j+1,k+2))) )) ;

/* 9 ops */
//	    lu(1,i,j,k) = a1*lu(1,i,j,k) + cof*r1;
//            lu(2,i,j,k) = a1*lu(2,i,j,k) + cof*r2;
//            lu(3,i,j,k) = a1*lu(3,i,j,k) + cof*r3;
            lu(1,i,j,k) =  cof*r1;
            lu(2,i,j,k) =  cof*r2;
            lu(3,i,j,k) =  cof*r3;
            }
      }
   }


            

}//end brace


void rhs4sg_rev_raja( int ifirst, int ilast, int jfirst, int jlast, int kfirst, int klast,
		 int nk, float_sw4* __restrict__ a_acof, float_sw4 *__restrict__ a_bope,
		 float_sw4* __restrict__ a_ghcof, float_sw4* __restrict__ a_lu, float_sw4* __restrict__ a_u,
		 float_sw4* __restrict__ a_mu, float_sw4* __restrict__ a_lambda, 
		 float_sw4 h, float_sw4* __restrict__ a_strx, float_sw4* __restrict__ a_stry, 
		 float_sw4* __restrict__ a_strz )
{//raja fun start


 // Direct reuse of fortran code by these macro definitions:
#define mu(i,j,k)     a_mu[base+i+ni*(j)+nij*(k)]
#define la(i,j,k) a_lambda[base+i+ni*(j)+nij*(k)]
   // Reversed indexation
#define u(c,i,j,k)   a_u[base3+i+ni*(j)+nij*(k)+nijk*(c)]   
#define lu(c,i,j,k) a_lu[base3+i+ni*(j)+nij*(k)+nijk*(c)]   
#define strx(i) a_strx[i-ifirst0]
#define stry(j) a_stry[j-jfirst0]
#define strz(k) a_strz[k-kfirst0]
#define acof(i,j,k) a_acof[(i-1)+6*(j-1)+48*(k-1)]
#define bope(i,j) a_bope[i-1+6*(j-1)]
#define ghcof(i) a_ghcof[i-1]
   
   const float_sw4 a1   = 0;
   const float_sw4 i6   = 1.0/6;
   const float_sw4 i12  = 1.0/12;
   const float_sw4 i144 = 1.0/144;
   const float_sw4 tf   = 0.75;

   const int ni    = ilast-ifirst+1;
   const int nij   = ni*(jlast-jfirst+1);
   const int nijk  = nij*(klast-kfirst+1);
   const int base  = -(ifirst+ni*jfirst+nij*kfirst);
   const int base3 = base-nijk;

   const int nic  = 3*ni;
   const int nijc = 3*nij;
   const int ifirst0 = ifirst;
   const int jfirst0 = jfirst;
   const int kfirst0 = kfirst;


   const float_sw4 cof = 1.0/(h*h);
   
   int k1 = 7; 
   int k2 = 101;   
   RAJA::RangeSegment k_range(k1,k2+1);
   RAJA::RangeSegment j_range(jfirst+2,jlast-1);
   RAJA::RangeSegment i_range(ifirst+2,ilast-1);

#ifdef KERNEL
   std::cout<<"using RAJA Kernel"<<std::endl;
   RAJA::kernel<POL>(RAJA::make_tuple(i_range,j_range,k_range),
                     [=] (int i, int j, int k) {
#else
  std::cout<<"using RAJA Forall nesting"<<std::endl;
  RAJA::forall<POL2>(k_range, [=] (int k){
        RAJA::forall<POL1>(j_range, [=] (int j){
              RAJA::forall<POL0>(i_range, [=] (int i){
#endif

      float_sw4 mux1,mux2,mux3,mux4,muy1,muy2,muy3,muy4;
      float_sw4 r1,r2,r3,mucof,mu1zz,mu2zz,mu3zz,lap2mu,q,u3zip2,u3zip1;
      float_sw4 u3zim1,u3zim2,lau3zx,mu3xz,u3zjp2,u3zjp1,u3zjm1,u3zjm2,lau3zy;
      float_sw4 mu3yz,mu1zx,u1zip2,u1zip1,u1zim1,u1zim2;
      float_sw4 u2zjp2,u2zjp1,u2zjm1,u2zjm2,mu2zy,lau1xz,lau2yz,muz1,muz2,muz3,muz4;
      
/* from inner_loop_4a, 28x3 = 84 ops */
            mux1 = mu(i-1,j,k)*strx(i-1)-
	       tf*(mu(i,j,k)*strx(i)+mu(i-2,j,k)*strx(i-2));
            mux2 = mu(i-2,j,k)*strx(i-2)+mu(i+1,j,k)*strx(i+1)+
	       3*(mu(i,j,k)*strx(i)+mu(i-1,j,k)*strx(i-1));
            mux3 = mu(i-1,j,k)*strx(i-1)+mu(i+2,j,k)*strx(i+2)+
	       3*(mu(i+1,j,k)*strx(i+1)+mu(i,j,k)*strx(i));
            mux4 = mu(i+1,j,k)*strx(i+1)-
	       tf*(mu(i,j,k)*strx(i)+mu(i+2,j,k)*strx(i+2));

            muy1 = mu(i,j-1,k)*stry(j-1)-
	       tf*(mu(i,j,k)*stry(j)+mu(i,j-2,k)*stry(j-2));
            muy2 = mu(i,j-2,k)*stry(j-2)+mu(i,j+1,k)*stry(j+1)+
	       3*(mu(i,j,k)*stry(j)+mu(i,j-1,k)*stry(j-1));
            muy3 = mu(i,j-1,k)*stry(j-1)+mu(i,j+2,k)*stry(j+2)+
	       3*(mu(i,j+1,k)*stry(j+1)+mu(i,j,k)*stry(j));
            muy4 = mu(i,j+1,k)*stry(j+1)-
	       tf*(mu(i,j,k)*stry(j)+mu(i,j+2,k)*stry(j+2));

            muz1 = mu(i,j,k-1)*strz(k-1)-
	       tf*(mu(i,j,k)*strz(k)+mu(i,j,k-2)*strz(k-2));
            muz2 = mu(i,j,k-2)*strz(k-2)+mu(i,j,k+1)*strz(k+1)+
	       3*(mu(i,j,k)*strz(k)+mu(i,j,k-1)*strz(k-1));
            muz3 = mu(i,j,k-1)*strz(k-1)+mu(i,j,k+2)*strz(k+2)+
	       3*(mu(i,j,k+1)*strz(k+1)+mu(i,j,k)*strz(k));
            muz4 = mu(i,j,k+1)*strz(k+1)-
	       tf*(mu(i,j,k)*strz(k)+mu(i,j,k+2)*strz(k+2));
/* xx, yy, and zz derivatives:*/
/* 75 ops */
            r1 = i6*( strx(i)*( (2*mux1+la(i-1,j,k)*strx(i-1)-
               tf*(la(i,j,k)*strx(i)+la(i-2,j,k)*strx(i-2)))*
                              (u(1,i-2,j,k)-u(1,i,j,k))+
           (2*mux2+la(i-2,j,k)*strx(i-2)+la(i+1,j,k)*strx(i+1)+
                3*(la(i,j,k)*strx(i)+la(i-1,j,k)*strx(i-1)))*
                              (u(1,i-1,j,k)-u(1,i,j,k))+ 
           (2*mux3+la(i-1,j,k)*strx(i-1)+la(i+2,j,k)*strx(i+2)+
                3*(la(i+1,j,k)*strx(i+1)+la(i,j,k)*strx(i)))*
                              (u(1,i+1,j,k)-u(1,i,j,k))+
                (2*mux4+ la(i+1,j,k)*strx(i+1)-
               tf*(la(i,j,k)*strx(i)+la(i+2,j,k)*strx(i+2)))*
                (u(1,i+2,j,k)-u(1,i,j,k)) ) + stry(j)*(
                     muy1*(u(1,i,j-2,k)-u(1,i,j,k)) + 
                     muy2*(u(1,i,j-1,k)-u(1,i,j,k)) + 
                     muy3*(u(1,i,j+1,k)-u(1,i,j,k)) +
                     muy4*(u(1,i,j+2,k)-u(1,i,j,k)) ) + strz(k)*(
                     muz1*(u(1,i,j,k-2)-u(1,i,j,k)) + 
                     muz2*(u(1,i,j,k-1)-u(1,i,j,k)) + 
                     muz3*(u(1,i,j,k+1)-u(1,i,j,k)) +
                     muz4*(u(1,i,j,k+2)-u(1,i,j,k)) ) );

/* 75 ops */
            r2 = i6*( strx(i)*(mux1*(u(2,i-2,j,k)-u(2,i,j,k)) + 
                      mux2*(u(2,i-1,j,k)-u(2,i,j,k)) + 
                      mux3*(u(2,i+1,j,k)-u(2,i,j,k)) +
                      mux4*(u(2,i+2,j,k)-u(2,i,j,k)) ) + stry(j)*(
                  (2*muy1+la(i,j-1,k)*stry(j-1)-
                      tf*(la(i,j,k)*stry(j)+la(i,j-2,k)*stry(j-2)))*
                          (u(2,i,j-2,k)-u(2,i,j,k))+
           (2*muy2+la(i,j-2,k)*stry(j-2)+la(i,j+1,k)*stry(j+1)+
                     3*(la(i,j,k)*stry(j)+la(i,j-1,k)*stry(j-1)))*
                          (u(2,i,j-1,k)-u(2,i,j,k))+ 
           (2*muy3+la(i,j-1,k)*stry(j-1)+la(i,j+2,k)*stry(j+2)+
                     3*(la(i,j+1,k)*stry(j+1)+la(i,j,k)*stry(j)))*
                          (u(2,i,j+1,k)-u(2,i,j,k))+
                  (2*muy4+la(i,j+1,k)*stry(j+1)-
                    tf*(la(i,j,k)*stry(j)+la(i,j+2,k)*stry(j+2)))*
                          (u(2,i,j+2,k)-u(2,i,j,k)) ) + strz(k)*(
                     muz1*(u(2,i,j,k-2)-u(2,i,j,k)) + 
                     muz2*(u(2,i,j,k-1)-u(2,i,j,k)) + 
                     muz3*(u(2,i,j,k+1)-u(2,i,j,k)) +
                     muz4*(u(2,i,j,k+2)-u(2,i,j,k)) ) );

/* 75 ops */
            r3 = i6*( strx(i)*(mux1*(u(3,i-2,j,k)-u(3,i,j,k)) + 
                      mux2*(u(3,i-1,j,k)-u(3,i,j,k)) + 
                      mux3*(u(3,i+1,j,k)-u(3,i,j,k)) +
                      mux4*(u(3,i+2,j,k)-u(3,i,j,k))  ) + stry(j)*(
                     muy1*(u(3,i,j-2,k)-u(3,i,j,k)) + 
                     muy2*(u(3,i,j-1,k)-u(3,i,j,k)) + 
                     muy3*(u(3,i,j+1,k)-u(3,i,j,k)) +
                     muy4*(u(3,i,j+2,k)-u(3,i,j,k)) ) + strz(k)*(
                  (2*muz1+la(i,j,k-1)*strz(k-1)-
                      tf*(la(i,j,k)*strz(k)+la(i,j,k-2)*strz(k-2)))*
                          (u(3,i,j,k-2)-u(3,i,j,k))+
           (2*muz2+la(i,j,k-2)*strz(k-2)+la(i,j,k+1)*strz(k+1)+
                      3*(la(i,j,k)*strz(k)+la(i,j,k-1)*strz(k-1)))*
                          (u(3,i,j,k-1)-u(3,i,j,k))+ 
           (2*muz3+la(i,j,k-1)*strz(k-1)+la(i,j,k+2)*strz(k+2)+
                      3*(la(i,j,k+1)*strz(k+1)+la(i,j,k)*strz(k)))*
                          (u(3,i,j,k+1)-u(3,i,j,k))+
                  (2*muz4+la(i,j,k+1)*strz(k+1)-
                    tf*(la(i,j,k)*strz(k)+la(i,j,k+2)*strz(k+2)))*
		  (u(3,i,j,k+2)-u(3,i,j,k)) ) );


/* Mixed derivatives: */
/* 29ops /mixed derivative */
/* 116 ops for r1 */
/*   (la*v_y)_x */
            r1 = r1 + strx(i)*stry(j)*
                 i144*( la(i-2,j,k)*(u(2,i-2,j-2,k)-u(2,i-2,j+2,k)+
                             8*(-u(2,i-2,j-1,k)+u(2,i-2,j+1,k))) - 8*(
                        la(i-1,j,k)*(u(2,i-1,j-2,k)-u(2,i-1,j+2,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i-1,j+1,k))) )+8*(
                        la(i+1,j,k)*(u(2,i+1,j-2,k)-u(2,i+1,j+2,k)+
                             8*(-u(2,i+1,j-1,k)+u(2,i+1,j+1,k))) ) - (
                        la(i+2,j,k)*(u(2,i+2,j-2,k)-u(2,i+2,j+2,k)+
                             8*(-u(2,i+2,j-1,k)+u(2,i+2,j+1,k))) )) 
/*   (la*w_z)_x */
               + strx(i)*strz(k)*       
                 i144*( la(i-2,j,k)*(u(3,i-2,j,k-2)-u(3,i-2,j,k+2)+
                             8*(-u(3,i-2,j,k-1)+u(3,i-2,j,k+1))) - 8*(
                        la(i-1,j,k)*(u(3,i-1,j,k-2)-u(3,i-1,j,k+2)+
                             8*(-u(3,i-1,j,k-1)+u(3,i-1,j,k+1))) )+8*(
                        la(i+1,j,k)*(u(3,i+1,j,k-2)-u(3,i+1,j,k+2)+
                             8*(-u(3,i+1,j,k-1)+u(3,i+1,j,k+1))) ) - (
                        la(i+2,j,k)*(u(3,i+2,j,k-2)-u(3,i+2,j,k+2)+
                             8*(-u(3,i+2,j,k-1)+u(3,i+2,j,k+1))) )) 
/*   (mu*v_x)_y */
               + strx(i)*stry(j)*       
                 i144*( mu(i,j-2,k)*(u(2,i-2,j-2,k)-u(2,i+2,j-2,k)+
                             8*(-u(2,i-1,j-2,k)+u(2,i+1,j-2,k))) - 8*(
                        mu(i,j-1,k)*(u(2,i-2,j-1,k)-u(2,i+2,j-1,k)+
                             8*(-u(2,i-1,j-1,k)+u(2,i+1,j-1,k))) )+8*(
                        mu(i,j+1,k)*(u(2,i-2,j+1,k)-u(2,i+2,j+1,k)+
                             8*(-u(2,i-1,j+1,k)+u(2,i+1,j+1,k))) ) - (
                        mu(i,j+2,k)*(u(2,i-2,j+2,k)-u(2,i+2,j+2,k)+
                             8*(-u(2,i-1,j+2,k)+u(2,i+1,j+2,k))) )) 
/*   (mu*w_x)_z */
               + strx(i)*strz(k)*       
                 i144*( mu(i,j,k-2)*(u(3,i-2,j,k-2)-u(3,i+2,j,k-2)+
                             8*(-u(3,i-1,j,k-2)+u(3,i+1,j,k-2))) - 8*(
                        mu(i,j,k-1)*(u(3,i-2,j,k-1)-u(3,i+2,j,k-1)+
                             8*(-u(3,i-1,j,k-1)+u(3,i+1,j,k-1))) )+8*(
                        mu(i,j,k+1)*(u(3,i-2,j,k+1)-u(3,i+2,j,k+1)+
                             8*(-u(3,i-1,j,k+1)+u(3,i+1,j,k+1))) ) - (
                        mu(i,j,k+2)*(u(3,i-2,j,k+2)-u(3,i+2,j,k+2)+
				     8*(-u(3,i-1,j,k+2)+u(3,i+1,j,k+2))) )) ;

/* 116 ops for r2 */
/*   (mu*u_y)_x */
            r2 = r2 + strx(i)*stry(j)*
                 i144*( mu(i-2,j,k)*(u(1,i-2,j-2,k)-u(1,i-2,j+2,k)+
                             8*(-u(1,i-2,j-1,k)+u(1,i-2,j+1,k))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j-2,k)-u(1,i-1,j+2,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i-1,j+1,k))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j-2,k)-u(1,i+1,j+2,k)+
                             8*(-u(1,i+1,j-1,k)+u(1,i+1,j+1,k))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j-2,k)-u(1,i+2,j+2,k)+
                             8*(-u(1,i+2,j-1,k)+u(1,i+2,j+1,k))) )) 
/* (la*u_x)_y */
              + strx(i)*stry(j)*
                 i144*( la(i,j-2,k)*(u(1,i-2,j-2,k)-u(1,i+2,j-2,k)+
                             8*(-u(1,i-1,j-2,k)+u(1,i+1,j-2,k))) - 8*(
                        la(i,j-1,k)*(u(1,i-2,j-1,k)-u(1,i+2,j-1,k)+
                             8*(-u(1,i-1,j-1,k)+u(1,i+1,j-1,k))) )+8*(
                        la(i,j+1,k)*(u(1,i-2,j+1,k)-u(1,i+2,j+1,k)+
                             8*(-u(1,i-1,j+1,k)+u(1,i+1,j+1,k))) ) - (
                        la(i,j+2,k)*(u(1,i-2,j+2,k)-u(1,i+2,j+2,k)+
                             8*(-u(1,i-1,j+2,k)+u(1,i+1,j+2,k))) )) 
/* (la*w_z)_y */
               + stry(j)*strz(k)*
                 i144*( la(i,j-2,k)*(u(3,i,j-2,k-2)-u(3,i,j-2,k+2)+
                             8*(-u(3,i,j-2,k-1)+u(3,i,j-2,k+1))) - 8*(
                        la(i,j-1,k)*(u(3,i,j-1,k-2)-u(3,i,j-1,k+2)+
                             8*(-u(3,i,j-1,k-1)+u(3,i,j-1,k+1))) )+8*(
                        la(i,j+1,k)*(u(3,i,j+1,k-2)-u(3,i,j+1,k+2)+
                             8*(-u(3,i,j+1,k-1)+u(3,i,j+1,k+1))) ) - (
                        la(i,j+2,k)*(u(3,i,j+2,k-2)-u(3,i,j+2,k+2)+
                             8*(-u(3,i,j+2,k-1)+u(3,i,j+2,k+1))) ))
/* (mu*w_y)_z */
               + stry(j)*strz(k)*
                 i144*( mu(i,j,k-2)*(u(3,i,j-2,k-2)-u(3,i,j+2,k-2)+
                             8*(-u(3,i,j-1,k-2)+u(3,i,j+1,k-2))) - 8*(
                        mu(i,j,k-1)*(u(3,i,j-2,k-1)-u(3,i,j+2,k-1)+
                             8*(-u(3,i,j-1,k-1)+u(3,i,j+1,k-1))) )+8*(
                        mu(i,j,k+1)*(u(3,i,j-2,k+1)-u(3,i,j+2,k+1)+
                             8*(-u(3,i,j-1,k+1)+u(3,i,j+1,k+1))) ) - (
                        mu(i,j,k+2)*(u(3,i,j-2,k+2)-u(3,i,j+2,k+2)+
				     8*(-u(3,i,j-1,k+2)+u(3,i,j+1,k+2))) )) ;
/* 116 ops for r3 */
/*  (mu*u_z)_x */
            r3 = r3 + strx(i)*strz(k)*
                 i144*( mu(i-2,j,k)*(u(1,i-2,j,k-2)-u(1,i-2,j,k+2)+
                             8*(-u(1,i-2,j,k-1)+u(1,i-2,j,k+1))) - 8*(
                        mu(i-1,j,k)*(u(1,i-1,j,k-2)-u(1,i-1,j,k+2)+
                             8*(-u(1,i-1,j,k-1)+u(1,i-1,j,k+1))) )+8*(
                        mu(i+1,j,k)*(u(1,i+1,j,k-2)-u(1,i+1,j,k+2)+
                             8*(-u(1,i+1,j,k-1)+u(1,i+1,j,k+1))) ) - (
                        mu(i+2,j,k)*(u(1,i+2,j,k-2)-u(1,i+2,j,k+2)+
                             8*(-u(1,i+2,j,k-1)+u(1,i+2,j,k+1))) )) 
/* (mu*v_z)_y */
              + stry(j)*strz(k)*
                 i144*( mu(i,j-2,k)*(u(2,i,j-2,k-2)-u(2,i,j-2,k+2)+
                             8*(-u(2,i,j-2,k-1)+u(2,i,j-2,k+1))) - 8*(
                        mu(i,j-1,k)*(u(2,i,j-1,k-2)-u(2,i,j-1,k+2)+
                             8*(-u(2,i,j-1,k-1)+u(2,i,j-1,k+1))) )+8*(
                        mu(i,j+1,k)*(u(2,i,j+1,k-2)-u(2,i,j+1,k+2)+
                             8*(-u(2,i,j+1,k-1)+u(2,i,j+1,k+1))) ) - (
                        mu(i,j+2,k)*(u(2,i,j+2,k-2)-u(2,i,j+2,k+2)+
                             8*(-u(2,i,j+2,k-1)+u(2,i,j+2,k+1))) ))
/*   (la*u_x)_z */
              + strx(i)*strz(k)*
                 i144*( la(i,j,k-2)*(u(1,i-2,j,k-2)-u(1,i+2,j,k-2)+
                             8*(-u(1,i-1,j,k-2)+u(1,i+1,j,k-2))) - 8*(
                        la(i,j,k-1)*(u(1,i-2,j,k-1)-u(1,i+2,j,k-1)+
                             8*(-u(1,i-1,j,k-1)+u(1,i+1,j,k-1))) )+8*(
                        la(i,j,k+1)*(u(1,i-2,j,k+1)-u(1,i+2,j,k+1)+
                             8*(-u(1,i-1,j,k+1)+u(1,i+1,j,k+1))) ) - (
                        la(i,j,k+2)*(u(1,i-2,j,k+2)-u(1,i+2,j,k+2)+
                             8*(-u(1,i-1,j,k+2)+u(1,i+1,j,k+2))) )) 
/* (la*v_y)_z */
              + stry(j)*strz(k)*
                 i144*( la(i,j,k-2)*(u(2,i,j-2,k-2)-u(2,i,j+2,k-2)+
                             8*(-u(2,i,j-1,k-2)+u(2,i,j+1,k-2))) - 8*(
                        la(i,j,k-1)*(u(2,i,j-2,k-1)-u(2,i,j+2,k-1)+
                             8*(-u(2,i,j-1,k-1)+u(2,i,j+1,k-1))) )+8*(
                        la(i,j,k+1)*(u(2,i,j-2,k+1)-u(2,i,j+2,k+1)+
                             8*(-u(2,i,j-1,k+1)+u(2,i,j+1,k+1))) ) - (
                        la(i,j,k+2)*(u(2,i,j-2,k+2)-u(2,i,j+2,k+2)+
				     8*(-u(2,i,j-1,k+2)+u(2,i,j+1,k+2))) )) ;

/* 9 ops */
//	    lu(1,i,j,k) = a1*lu(1,i,j,k) + cof*r1;
//            lu(2,i,j,k) = a1*lu(2,i,j,k) + cof*r2;
//            lu(3,i,j,k) = a1*lu(3,i,j,k) + cof*r3;
            lu(1,i,j,k) =  cof*r1;
            lu(2,i,j,k) =  cof*r2;
            lu(3,i,j,k) =  cof*r3;

#ifdef KERNEL            
                 });
#else
                 });
              });
        });
#endif  

}//end brace

