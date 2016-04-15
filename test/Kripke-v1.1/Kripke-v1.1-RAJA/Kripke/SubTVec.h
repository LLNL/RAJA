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

#ifndef KRIPKE_SUBTVEC_H__
#define KRIPKE_SUBTVEC_H__

#define KRIPKE_ALIGN_DATA

#define KRIPKE_ALIGN 64

#include <Kripke/Kernel.h>
#include <algorithm>
#include <vector>
#include <stdlib.h>

/**
 *  A transport vector (used for Psi and Phi, RHS, etc.)
 *
 *  This provides the inner most three strides of
 *    Psi[GS][DS][G][D][Z]
 *  but in whatever nesting order is specified.
 */
struct SubTVec {
private:
  // disallow
  SubTVec(SubTVec const &c);
  SubTVec &operator=(SubTVec const &c);

public:
  SubTVec(Nesting_Order nesting, int ngrps, int ndir_mom, int nzones):
    groups(ngrps),
    directions(ndir_mom),
    zones(nzones),
    elements(groups*directions*zones),
    data_linear(NULL)
  {
#ifdef KRIPKE_ALIGN_DATA
    int status = posix_memalign((void**)&data_linear, KRIPKE_ALIGN, sizeof(double)*elements);
    if(status != 0){
    	printf("Error allocating data\n");
    	data_linear = NULL;
    }
#else
    data_linear = (double *) malloc(sizeof(double)*elements);
#endif
    setupIndices(nesting, data_linear);
  }


  /**
   * ALIASING version of constructor.
   * Use this when you have a data buffer already, and don't want this class
   * to do any memory management.
   */
  SubTVec(Nesting_Order nesting, int ngrps, int ndir_mom, int nzones, double *ptr):
    groups(ngrps),
    directions(ndir_mom),
    zones(nzones),
    elements(groups*directions*zones),
    data_linear(NULL)
  {
    setupIndices(nesting, ptr);
  }

  ~SubTVec(){
    if(data_linear != NULL){
      free(data_linear);
    }
  }

  void setupIndices(Nesting_Order nesting, double *ptr){
    // setup nesting order
    switch(nesting){
      case NEST_GDZ:
        ext_to_int[0] = 0;
        ext_to_int[1] = 1;
        ext_to_int[2] = 2;
        break;
      case NEST_GZD:
        ext_to_int[0] = 0;
        ext_to_int[2] = 1;
        ext_to_int[1] = 2;
        break;
      case NEST_DZG:
        ext_to_int[1] = 0;
        ext_to_int[2] = 1;
        ext_to_int[0] = 2;
        break;
      case NEST_DGZ:
        ext_to_int[1] = 0;
        ext_to_int[0] = 1;
        ext_to_int[2] = 2;
        break;
      case NEST_ZDG:
        ext_to_int[2] = 0;
        ext_to_int[1] = 1;
        ext_to_int[0] = 2;
        break;
      case NEST_ZGD:
        ext_to_int[2] = 0;
        ext_to_int[0] = 1;
        ext_to_int[1] = 2;
        break;
    }

    // setup dimensionality
    int size_ext[3];
    size_ext[0] = groups;
    size_ext[1] = directions;
    size_ext[2] = zones;

    // map to internal indices
    for(int i = 0; i < 3; ++i){
      size_int[ext_to_int[i]] = size_ext[i];
    }

    data_pointer = ptr;
  }

  inline double* ptr(void){
    return data_pointer;
  }

  inline double* ptr(int g, int d, int z){
    return &(*this)(g,d,z);
  }

  // These are NOT efficient.. just used to re-stride data for comparisons
  inline double &operator()(int g, int d, int z) {
    int idx[3];
    idx[ext_to_int[0]] = g;
    idx[ext_to_int[1]] = d;
    idx[ext_to_int[2]] = z;
    int offset = idx[0] * size_int[1]*size_int[2] +
                 idx[1] * size_int[2] +
                 idx[2];
    return data_pointer[offset];
  }
  inline double operator()(int g, int d, int z) const {
    return (*const_cast<SubTVec*>(this))(g,d,z);
  }

  inline double sum(void) const {
    double s = 0.0;
    for(size_t i = 0;i < elements;++ i){
      s+= data_linear[i];
    }
    return s;
  }

  inline void clear(double v){
#ifdef KRIPKE_USE_OPENMP
#pragma omp parallel for
#endif
    for(int i = 0;i < elements;++ i){
      data_linear[i] = v;
    }
  }

  inline void randomizeData(void){
    for(int i = 0;i < elements;++ i){
      data_linear[i] = drand48();
    }
  }

  inline void copy(SubTVec const &b){
    for(int g = 0;g < groups;++ g){
      for(int d = 0;d < directions; ++ d){
        for(int z = 0;z < zones;++ z){
          // Copy using abstract indexing
          (*this)(g,d,z) = b(g,d,z);
        }
      }
    }
  }

  inline bool compare(std::string const &name, SubTVec const &b,
      double tol, bool verbose){

    bool is_diff = false;
    int num_wrong = 0;
    for(int g = 0;g < groups;++ g){
      for(int d = 0;d < directions; ++ d){
        for(int z = 0;z < zones;++ z){
          // Copy using abstract indexing
          double err = std::abs((*this)(g,d,z) - b(g,d,z));
          if(err > tol){
            is_diff = true;
            if(verbose){
              printf("%s[g=%d, d=%d, z=%d]: |%e - %e| = %e\n",
                  name.c_str(), g,d,z, (*this)(g,d,z), b(g,d,z), err);
              num_wrong ++;
              if(num_wrong > 100){
                return true;
              }
            }
          }
        }
      }
    }
    return is_diff;
  }

  int ext_to_int[3]; // external index to internal index mapping
  int size_int[3]; // size of each dimension in internal indices

  int groups, directions, zones, elements;
  double *data_pointer;
  //std::vector<double> data_linear;
  double *data_linear;
};


#endif
