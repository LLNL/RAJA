//Contains code for the symbolic execution of raja kernels

#ifndef RAJA_SymExec_HPP
#define RAJA_SymExec_HPP

#include "RAJA/config.hpp"

#include <string>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>

namespace RAJA
{
struct SymIterator;
struct SymAccess;
struct SymAccessList;


struct SymIterator {
  
  std::string name;
  long int idx;
  std::shared_ptr<std::vector<SymAccess>> accesses; // accesses that use this iterator

  SymIterator(std::string str) : name(str), idx(0) {
    accesses = std::make_shared<std::vector<SymAccess>>();
  }
/*
  SymIterator(long int num) : name("placeholder"), idx(num) {
    accesses = std::make_shared<std::vector<SymAccess>>();
  }
*/
  SymIterator(const SymIterator & other) : name(other.name), idx(other.idx) {
    accesses = other.accesses;
  }

  SymIterator& operator=(const SymIterator& other) = default;

  bool operator == (const int) {
    return 0;
  }
  template <typename T>
  SymIterator operator * (const T & other) {
    std::stringstream b;

    b << name << "*" << other;
    SymIterator newIterator = SymIterator(b.str());
    newIterator.accesses = this->accesses;

    return newIterator;
  }

  template <typename T>
  SymIterator operator + (const T & other) {
    std::stringstream b;

    b << name << "+" << other;
    SymIterator newIterator = SymIterator(b.str());
    newIterator.accesses = this->accesses;

    return newIterator;
  }

  template <typename T>
  SymIterator operator - (const T & other) {
    std::stringstream b;

    b << name << "-" << other;
    SymIterator newIterator = SymIterator(b.str());
    newIterator.accesses = this->accesses;

    return newIterator;
  }  

  template <typename T>
  SymIterator operator / (const T & other) {
    std::stringstream b;

    b << name << "/" << other;
    SymIterator newIterator = SymIterator(b.str());
    newIterator.accesses = this->accesses;

    return newIterator;
  }

  template <typename T>
  SymIterator operator % (const T & other) {
    std::stringstream b;

    b << name << "%" << other;
    SymIterator newIterator = SymIterator(b.str());
    newIterator.accesses = this->accesses;

    return newIterator;
  }

  friend std::ostream& operator<< (std::ostream& s, SymIterator i) {
    s << i.name;
    return s;
  }

}; //SymIterator

template <typename T>
SymIterator operator + (const T & other, const SymIterator & iterator);
template <typename T>
SymIterator operator - (const T & other, const SymIterator & iterator);
template <typename T>
SymIterator operator / (const T & other, const SymIterator & iterator);
template <typename T>
SymIterator operator % (const T & other, const SymIterator & iterator);

template <typename T>
SymIterator operator * (const T & other, const SymIterator & iterator) {
  std::stringstream b;

  b << other << "*" << iterator.name;
  SymIterator newIterator = SymIterator(b.str());
  newIterator.accesses = iterator.accesses;

  return newIterator;
}

struct SymAccess {
  
  void * view;
  std::vector<SymIterator> iterators;
  bool isRead;
  bool isWrite;

  SymAccess(void * _view, std::vector<SymIterator>& _iterators) {
    view = _view;
    iterators = _iterators;
    isRead = false;
    isWrite = false;
  }

  void set_read(); 
  
  void set_write();
  void link_to_iterators();

  std::string access_string();
  operator SymAccessList();

  friend std::ostream& operator<< (std::ostream& s, SymAccess a);
  
  int operator < (const SymAccess other) {
    std::stringstream a, b;
    a << this;
    b << other;
   
    return a.str().compare(b.str());
  }
}; //SymAccess


bool operator < (const SymAccess a, const SymAccess b);
void print_access_list(std::ostream&s, std::vector<SymAccess> accesses, int indent); 

struct SymAccessList {

  std::vector<SymAccess> accesses;

  SymAccessList() {
    accesses = std::vector<SymAccess>();
  }

  SymAccessList(const SymAccess & a) {
    accesses = std::vector<SymAccess>();
    accesses.push_back(a);
  }

  SymAccessList(const SymAccessList& other) {
    accesses = std::vector<SymAccess>();
    for(SymAccess a : other.accesses) {
      accesses.push_back(a);
    }
  } 
  void push_back(const SymAccess & a) {
    accesses.push_back(a);
  }

  //rhs operations 



   //for "a(i) + 2" like statements
  void num_cast() {
    for(SymAccess& a : accesses) {
      a.set_read();
      a.link_to_iterators();
    }
  }

  /*//for "a(i) + i" like statements
  SymAccessList & operator + (const SymIterator &) {
    for(SymAccess& a : accesses) {
      a.set_read();
    }
    return *this;
  } */
  operator double() {num_cast(); return 1.0;}
  
  
  //assignment operations

  SymAccessList operator = (const SymAccessList & other) {
    SymAccessList newList = SymAccessList();

    for(SymAccess & a : accesses) {
      a.set_write();
      newList.push_back(a);
    }

    for(SymAccess a : other.accesses) {
      a.set_read();
      newList.push_back(a);
    }

    for(SymAccess& a : newList.accesses) {
      a.link_to_iterators();
    }

    return newList;
  }

  SymAccessList symbolic_update_equals (const SymAccessList & other) {
    SymAccessList newList = SymAccessList();

    for(SymAccess& a : accesses) {
      a.set_write();
      a.set_read();
      newList.push_back(a);
    }

    for(SymAccess a : other.accesses) {
      a.set_read();
      newList.push_back(a);
    }

    for(SymAccess& a : newList.accesses) {
      a.link_to_iterators();
    }

    return newList;
  }


  SymAccessList num_assign() {
    SymAccessList newList = SymAccessList();

    for(SymAccess& a : accesses) {
      a.set_write();
      newList.push_back(a);
    }

    for(SymAccess& a : newList.accesses) {
      a.link_to_iterators();
    }

    return newList;
  }

  SymAccessList operator = (int) { return num_assign(); }
  SymAccessList operator = (long int) { return num_assign(); }
  SymAccessList operator = (float) { return num_assign(); }
  SymAccessList operator = (double) { return num_assign(); }
  SymAccessList operator = (SymIterator) { return num_assign(); }
  
  SymAccessList num_update_equals() {
    SymAccessList newList = SymAccessList();

    for(SymAccess& a : accesses) {
      a.set_write();
      a.set_read();
      newList.push_back(a);
    }

    for(SymAccess& a : newList.accesses) {
      a.link_to_iterators();
    }

    return newList;
  }

  SymAccessList operator += (int) { return num_update_equals(); }
  SymAccessList operator += (long int) { return num_update_equals(); }
  SymAccessList operator += (float) { return num_update_equals(); }
  SymAccessList operator += (double) { return num_update_equals(); }
  
  SymAccessList operator -= (int) { return num_update_equals(); }
  SymAccessList operator -= (long int) { return num_update_equals(); }
  SymAccessList operator -= (float) { return num_update_equals(); }
  SymAccessList operator -= (double) { return num_update_equals(); }
 
  SymAccessList operator *= (int) { return num_update_equals(); }
  SymAccessList operator *= (long int) { return num_update_equals(); }
  SymAccessList operator *= (float) { return num_update_equals(); }
  SymAccessList operator *= (double) { return num_update_equals(); }
 
  SymAccessList operator /= (int) { return num_update_equals(); }
  SymAccessList operator /= (long int) { return num_update_equals(); }
  SymAccessList operator /= (float) { return num_update_equals(); }
  SymAccessList operator /= (double) { return num_update_equals(); }
 
  SymAccessList operator %= (int) { return num_update_equals(); }
  SymAccessList operator %= (long int) { return num_update_equals(); }
  SymAccessList operator %= (float) { return num_update_equals(); }
  SymAccessList operator %= (double) { return num_update_equals(); }


}; //SymAccessList


SymAccessList list_op_int(const SymAccessList & l0, int scalar);
SymAccessList list_op_double(const SymAccessList & l0, double scalar);
SymAccessList list_op_iterator(const SymAccessList & l0, const SymIterator & i1);
SymAccessList int_op_list(int scalar, const SymAccessList & l1);
SymAccessList double_op_list(double scalar, const SymAccessList & l1);
SymAccessList iterator_op_list(const SymIterator & i0, const SymAccessList & l1);

// Code for expressions involving two symbolic access lists. 
//This should include operations that involve a symoblic access bc it has a conversion function to symaccesslist.
SymAccessList operator + (const SymAccessList & l0, const SymAccessList & l1);
SymAccessList operator - (const SymAccessList & l0, const SymAccessList & l1);
SymAccessList operator * (const SymAccessList & l0, const SymAccessList & l1);
SymAccessList operator / (const SymAccessList & l0, const SymAccessList & l1);
SymAccessList operator % (const SymAccessList & l0, const SymAccessList & l1);


// functions for list op int
SymAccessList operator + (const SymAccessList & l0, int scalar);
SymAccessList operator - (const SymAccessList & l0, int scalar);
SymAccessList operator * (const SymAccessList & l0, int scalar);
SymAccessList operator / (const SymAccessList & l0, int scalar);
SymAccessList operator % (const SymAccessList & l0, int scalar);

//functions for list op double
SymAccessList operator + (const SymAccessList & l0, double scalar);
SymAccessList operator - (const SymAccessList & l0, double scalar);
SymAccessList operator * (const SymAccessList & l0, double scalar);
SymAccessList operator / (const SymAccessList & l0, double scalar);
SymAccessList operator % (const SymAccessList & l0, double scalar);

//functions for list op const SymIterator
SymAccessList operator + (const SymAccessList & l0, const SymIterator & i1);
SymAccessList operator - (const SymAccessList & l0, const SymIterator & i1);
SymAccessList operator * (const SymAccessList & l0, const SymIterator & i1);
SymAccessList operator / (const SymAccessList & l0, const SymIterator & i1);
SymAccessList operator % (const SymAccessList & l0, const SymIterator & i1);


//functions for int op list
SymAccessList operator + (int scalar, const SymAccessList & l1);
SymAccessList operator - (int scalar, const SymAccessList & l1);
SymAccessList operator * (int scalar, const SymAccessList & l1);
SymAccessList operator / (int scalar, const SymAccessList & l1);
SymAccessList operator % (int scalar, const SymAccessList & l1);

//functions for double op list
SymAccessList operator + (double scalar, const SymAccessList & l1);
SymAccessList operator - (double scalar, const SymAccessList & l1);
SymAccessList operator * (double scalar, const SymAccessList & l1);
SymAccessList operator / (double scalar, const SymAccessList & l1);


//functions for SymIterator op list
SymAccessList operator + (const SymIterator & i0, const SymAccessList & l1);
SymAccessList operator - (const SymIterator & i0, const SymAccessList & l1);
SymAccessList operator * (const SymIterator & i0, const SymAccessList & l1);
SymAccessList operator / (const SymIterator & i0, const SymAccessList & l1);
SymAccessList operator % (const SymIterator & i0, const SymAccessList & l1);


// Assignments

SymAccessList list_assign_list(const SymAccessList &l0, const SymAccessList & l1);
SymAccessList list_assign_int(const SymAccessList & l0, int scalar);
SymAccessList list_assign_double(const SymAccessList & l0, double scalar);
SymAccessList list_assign_iterator(const SymAccessList & l0, const SymIterator & i1);

SymAccessList list_update_list(const SymAccessList &l0, const SymAccessList & l1);
SymAccessList list_update_int(const SymAccessList & l0, int scalar);
SymAccessList list_update_double(const SymAccessList & l0, double scalar);
SymAccessList list_update_iterator(const SymAccessList & l0, const SymIterator & i1);

} // namespace RAJA






#endif
