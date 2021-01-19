#ifndef ARRAY_REDUCTIONS_HPP
#define ARRAY_REDUCTIONS_HPP

#include "RAJA/RAJA.hpp"

template<typename REDUCE_T>
class CombinableArray{
protected:
  using data_t = std::vector<REDUCE_T>;
  data_t mutable data;
  CombinableArray const *parent = nullptr;

public:
  CombinableArray(){}

  CombinableArray(const CombinableArray &copy) : 
    data(copy.data), 
    parent{copy.parent ? copy.parent : &copy} {}

  REDUCE_T& operator[](size_t i) const { return data[i]; }
  data_t &local() const { return data; }
};



template<typename REDUCE_T>
class VoT_t : public CombinableArray<REDUCE_T> {

  using Base = CombinableArray<REDUCE_T>;
public:

  VoT_t(size_t size){
    Base::data.resize(size);
    for(size_t i = 0; i < Base::data.size(); i++) {
      Base::data[i] = REDUCE_T(0);
    }
  }

  ~VoT_t(){
#pragma omp critical
    {
      if (Base::parent) { 
        #pragma omp parallel for
        for(size_t i = 0; i < Base::data.size(); i++) {
          Base::parent->local()[i] += Base::data[i];
        }
      }
    }
  }
};



template<typename REDUCE_T>
class VoR_t : public CombinableArray<REDUCE_T> {

  using Base = CombinableArray<REDUCE_T>;
public:

  VoR_t(size_t size){
    Base::data.resize(size);
    for(size_t i = 0; i < Base::data.size(); i++) {
      Base::data[i] = REDUCE_T(0);
    }
  }

  ~VoR_t(){}
};
 

// --------------------------------------------------------------------------------
// Some helper function definitions
// --------------------------------------------------------------------------------
using pairlist_t = std::vector<std::pair<int,int>>;

template<typename T1, typename T2>
void checkResults(const  T1& solution, const T2& test, const RAJA::ChronoTimer& timer);
pairlist_t generatePairList(const int n_nodes, const int n_pairs);
std::vector<double> generateSolution(const int n_nodes, const pairlist_t pl);
#endif
