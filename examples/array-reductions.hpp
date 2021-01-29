#ifndef ARRAY_REDUCTIONS_HPP
#define ARRAY_REDUCTIONS_HPP

#include "RAJA/RAJA.hpp"

template<int N, typename T>
struct data_dim { using type = std::vector<typename data_dim< N-1, T >::type>; };

template<typename T>
struct data_dim<0, T> { using type = T; };


template<int N, typename REDUCE_T>
class CombinableArray{
protected:
  using data_t = typename data_dim<N, REDUCE_T>::type;
  data_t mutable data;
  CombinableArray const *parent = nullptr;

public:
  CombinableArray(){}

  CombinableArray(const CombinableArray &copy) : 
    data(copy.data), 
    parent{copy.parent ? copy.parent : &copy} {}

  data_t &local() const { return data; }
};


template<int N, typename REDUCE_T>
class VoT_t : public CombinableArray<N, REDUCE_T> {};

template<typename REDUCE_T>
class VoT_t<1, REDUCE_T> : public CombinableArray<1, REDUCE_T> {
  using Base = CombinableArray<1, REDUCE_T>;
public:

  REDUCE_T& at(size_t i) const { return Base::data[i]; }

  template<typename RHS_T>
  bool operator==(const RHS_T& rhs) const {
    bool correctness = true;
    for(size_t i = 0; i < Base::data.size(); i++) 
      if (Base::data[i] != rhs[i]) correctness=false;
    return correctness; 
  }

  VoT_t(size_t size){
    Base::data.resize(size);
    for(size_t i = 0; i < Base::data.size(); i++) {
      Base::data[i] = REDUCE_T(0);
    }
  }

  ~VoT_t(){
    if (Base::parent) { 
    #pragma omp critical
      #pragma omp parallel for
      for(size_t i = 0; i < Base::data.size(); i++) {
        Base::parent->local()[i] += Base::data[i];
      }
    }
  }
};


template<typename REDUCE_T>
class VoT_t<2, REDUCE_T> : public CombinableArray<2, REDUCE_T> {
  using Base = CombinableArray<2, REDUCE_T>;
public:

  REDUCE_T& at(size_t i, size_t j) const { return Base::data[i][j]; }

  template<typename RHS_T>
  bool operator==(const RHS_T& rhs) const {
    bool correctness = true;
    for(size_t i = 0; i < Base::data.size(); i++) {
      for(size_t j = 0; j < Base::data[0].size(); j++) {
        size_t idx_1d = i * Base::data[0].size() + j;
        if (Base::data[i][j] != rhs[idx_1d]) correctness=false;
      }
    }
    return correctness; 
  }

  VoT_t(size_t size0, size_t size1){
    Base::data.resize(size0);
    for(size_t i = 0; i < size0; i++) {
      Base::data[i].resize(size1);
      for(size_t j = 0; j < size1; j++) {
        Base::data[i][j] = REDUCE_T(0);
      }
    }
  }

  ~VoT_t(){
    #pragma omp critical
    if (Base::parent) { 
      #pragma omp parallel for
      //#pragma omp parallel for collapse(2)
      for(size_t i = 0; i < Base::data.size(); i++) {
        for(size_t j = 0; j < Base::data[0].size(); j++) {
          Base::parent->local()[i][j] += Base::data[i][j];
        }
      }
    }
  }
};



template<typename REDUCE_T>
class VoR_t : public CombinableArray<1, REDUCE_T> {

  using Base = CombinableArray<1, REDUCE_T>;
public:
  REDUCE_T& at(size_t i) const { return Base::data[i]; }

  template<typename RHS_T>
  bool operator==(const RHS_T& rhs) const {
    bool correctness = true;
    for(size_t i = 0; i < Base::data.size(); i++) 
      if (Base::data[i] != rhs[i]) correctness=false;
    return correctness; 
  }

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

pairlist_t generatePairList(const int n_nodes, const int n_pairs);
pairlist_t generate2DPairList(const int n_nodes, const int n_node_lists, const int n_pairs);

std::vector<double> generateSolution(const int n_nodes, const pairlist_t pl);
std::vector<std::vector<double>> generate2DSolution(const int n_nodes, const int n_node_lists, const pairlist_t pl);

template<typename T1, typename T2>
void checkResults(const  T1& solution, const T2& test, const RAJA::ChronoTimer& timer);

#endif
