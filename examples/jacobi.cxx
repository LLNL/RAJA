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
#include <cstdio> 
#include <iostream>

#include "RAJA/RAJA.hpp"
#include "RAJA/util/defines.hpp"

// Solve for loop currents in structured resistor array
//Similar discretizationt to solving the Possion Equation
//with zero dirichlet boundary condtions
int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{

  //----[Setting up solver]-------
  int N  = 1000;
  int NN = (N+2)*(N+2);
  double tol = 1e-5;  
  double resI2 = 1, V, invD; 
  double *I    = new double [(N+2)*(N+2)];
  double *Iold = new double [(N+2)*(N+2)];

  unsigned int iteration=0;
  //----[Standard C approach]-----
  while(resI2>tol*tol){
    resI2 = 0; 
    
    invD = 1./4;
    V    = 0;

    for(unsigned int n=1; n<=N; ++n){
      for(unsigned int m=1; m<=N; ++m){

        unsigned id = n*(N+2) + m;
        I[id] = invD*(V-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);

        if(m==1 && n==1){
          V    = 1;
          invD = 1./3;
          I[id] = invD*(V-Iold[id-N-2]-Iold[id+N+2]-Iold[id-1]-Iold[id+1]);
        }

      }
    }
    
    //Reduction step
    for(int k=0; k<NN; ++k){
      resI2 += (I[k]-Iold[k])*(I[k]-Iold[k]);
      Iold[k]   = I[k];
    }
    
    ++iteration;
    if(!(iteration%100)){
      printf("at iteration %d resI = %g\n", iteration, sqrt(resI2));
    }
  }

  printf("Iterations: %d\n", iteration);
  printf("Top right current: %lg\n", I[N+N*(N+2)]);
  printf("Memory usage: %lg GB\n", (N+2)*(N+2)*sizeof(double)/1.e9);

  




  
  
  //Clean up
  delete [] I, Iold;


  return 0;
}
