#ifndef ARRAY_REDUCTIONS_HPP
#define ARRAY_REDUCTIONS_HPP

#include "RAJA/RAJA.hpp"
#include <iostream>

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

  type &local() const { return data; }
  CombMultiArray(const CombMultiArray& copy) :
    data(copy.data){}//,

  static void reduction(type& parent, type& data) {
    #pragma omp parallel for
    for (size_t i = 0; i < data.size(); i++) {
      inner_type::reduction(parent[i].local(), data[i].local());
    }
  }

  void print() {
    for(size_t i =0; i < data.size(); i++) data[i].print();
    std::cout << "\n";
    //std::cout << "Depth : " << depth << "\n";
  }

  template<typename U>
  bool operator==(const U& rhs) const {
    std::cout << "check\n";
    for(size_t i = 0; i < data.size(); i++) 
      if (data[i] != rhs[i]) return false;
    return true;
  }

  //template<typename U>
  //friend bool operator==(type& lhs, const U& rhs) {
  //  for(size_t i = 0; i < lhs.data.size(); i++) 
  //    if (lhs[i] != rhs[i]) return false;
  //  return true;
  //}

  //template<typename U>
  //friend bool operator==(const CombMultiArray<T,N>& lhs, const U& rhs) {
  //  for(size_t i = 0; i < lhs.data.size(); i++) 
  //    if (lhs.data[i] != rhs[i]) return false;
  //  return true;
  //}

protected:
  type mutable data;
};

template<typename T>
class CombMultiArray<T, 1> {
public:
  using type = std::vector<T>;
  using inner_type = T;
  inner_type& operator[](size_t idx) const { return data[idx]; }

  CombMultiArray(){};

  CombMultiArray(T val, int n) {
    data.resize(n);
    for(int i =0; i < n; i++){
      data[i] = val;
    }
  }

  type &local() const { return data; }
  CombMultiArray(const CombMultiArray& copy) : data(copy.data){}

  static void reduction(type& parent, type& data) {
    for (size_t i = 0; i < data.size(); i++) {
      parent[i] += data[i];
    }
  }

  void print() {
    for(size_t i =0; i < data.size(); i++) std::cout << data[i] << " ";
    std::cout << "\n";
  }

  template<typename U>
  bool operator==(const U& rhs) const {
    std::cout << "check2\n";
    for(size_t i = 0; i < data.size(); i++) 
      if (data[i] != rhs[i]) return false;
    return true;
  }

protected:
  type mutable data;
};



template<typename T, int N>
class ContainerReducer : public CombMultiArray<T, N> {
//class ContainerReducer {
  using Base = CombMultiArray<T, N>;
  using Base_inner = typename Base::inner_type;
  ContainerReducer const *parent = nullptr;

public:
  //Base cma;
  Base_inner& operator[](size_t idx) const { 
    return Base::data[idx];
    //return cma.local()[idx];
  }

  ContainerReducer(){}

  template<typename ...Args>
  ContainerReducer(Args ...args)
    : Base(args...) {  
    //{cma = Base(args...);
  }

  ContainerReducer(const ContainerReducer& copy)
    : Base(copy),
    parent{copy.parent ? copy.parent : &copy} {}
    //: parent{copy.parent ? copy.parent : &copy} { cma = Base(copy.cma); }

  void print() { Base::print(); }
  //void print() { cma.print(); }

  template<typename U>
  bool operator==(const U& rhs) const {

    //Base rhs_b = static_cast<Base>(rhs);
    return (Base::data == rhs); }
    //return ((*this) == rhs); }
    //return (cma == rhs); }

  ~ContainerReducer(){
    if (parent)
    #pragma omp critical
    Base::reduction(parent->local(), Base::local()); 
    //Base::reduction(parent->cma.local(), cma.local());
  }
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
