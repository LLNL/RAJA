#ifndef ARRAY_REDUCTIONS_HPP
#define ARRAY_REDUCTIONS_HPP

#include "RAJA/RAJA.hpp"
#include <iostream>

struct my_type {
  my_type(){ std::cout << "Hazza an object of quality!\n";}
};

template<typename T, int N>
class CombMultiArray { 
public:
  using inner_type = CombMultiArray< T, N-1 >;
  using type = std::vector<inner_type>;
  constexpr static int depth = N;

  inner_type& operator[](size_t idx) const { return data[idx]; }

  CombMultiArray(){};

  template<typename ...Args>
  CombMultiArray(T val, int n, Args ...dims) {
    data.resize(n);
    for(int i =0; i < n; i++){
      data[i] = inner_type(val, dims...);
    }
  }

  CombMultiArray(const CombMultiArray& copy) :
    data(copy.data),
    parent{copy.parent ? copy.parent : &copy} {}

  type &local() const { return data; }
protected:
  type mutable data;
  CombMultiArray const *parent = nullptr;
};

template<typename T>
class CombMultiArray<T, 1> {
public:
  using type = std::vector<T>;
  using inner_type = T;
  T& operator[](size_t idx) const { return data[idx]; }

  CombMultiArray(){};

  CombMultiArray(T val, int n) {
    data.resize(n);
    for(int i =0; i < n; i++){
      data[i] = val;
    }
  }

protected:
  type mutable data;
  CombMultiArray const *parent = nullptr;
};

//template<typename T>
//class CombMultiArray<T, 0> {
//public:
//  using type = T;
//  type data;
//
//  void operator=(const T& rhs){ data = rhs; }
//  CombMultiArray(){};
//  CombMultiArray(T val){
//    data = val; 
//    static int count = 0;
//    std::cout << "Test : " << ++count << " " << val << '\n';
//  }
//};

template<typename T>
std::ostream& operator<<(std::ostream& os, const CombMultiArray<T,0>& dd) {
  return os << dd.data;
}


//template<int N, typename REDUCE_T>
//class CombinableArray{
//protected:
//  using data_t = typename CombMultiArray<REDUCE_T, N>::type;
//  data_t mutable data;
//  CombinableArray const *parent = nullptr;
//
//public:
//  CombinableArray(){}
//
//  CombinableArray(const CombinableArray &copy) : 
//    data(copy.data), 
//    parent{copy.parent ? copy.parent : &copy} {}
//
//  data_t &local() const { return data; }
//};



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
