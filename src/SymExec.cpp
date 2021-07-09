
#include "RAJA/loopchain/SymExec.hpp"
#include <set>
namespace RAJA {


void SymAccess::set_read() {
    isRead = true;
  }

void SymAccess::set_write() {
    isWrite = true;
  }

  void SymAccess::link_to_iterators() {
    for(SymIterator i : iterators) {
      i.accesses->push_back(*this);
    }
  }

  std::string SymAccess::access_string() {
    std::stringstream s;
    for(auto i : iterators) {
      s << i.name << ",";
    }
    
    std::string res = s.str();
    res.pop_back();
    return res;
  }

  SymAccess::operator SymAccessList() {
    SymAccessList l = SymAccessList();
    l.push_back(*this);
    return l;
  }

  std::ostream& operator<< (std::ostream& s, SymAccess a) {
    s << " " << a.view << " ";
    s << a.access_string();
    return s;
  }

bool operator < (const SymAccess a, const SymAccess b) {
  std::stringstream i,j;
  i << a;
  j << b;
  return (i.str().compare(j.str()) < 0);
}
void print_access_list(std::ostream&s, std::vector<SymAccess> accesses, int indent) {
  
  std::set<SymAccess> printed = std::set<SymAccess>();

  for (auto a : accesses) {
    if(printed.find(a) == printed.end()) {

      for(int i = 0; i < indent; i++) {s << " ";}
      s << (a.isRead ? "R" : " ");   
      s << (a.isWrite ? "W" : " ");      
      s << a << "\n";
      printed.insert(a);
    }
  }
}





/*
 *
 * Operator Overloading for Symbolic Execution
 *
 *
 */

SymAccessList list_op_list(const SymAccessList & l0, const SymAccessList & l1) {
  SymAccessList newList = SymAccessList();
  for(SymAccess a : l0.accesses) {
    a.set_read();
    newList.push_back(a);
  }
  for(SymAccess a : l1.accesses) {
    a.set_read();
    newList.push_back(a);
  }
  return newList;
}
SymAccessList list_op_int(const SymAccessList & l0, int scalar) {
  SymAccessList newList = SymAccessList();
  for(SymAccess a : l0.accesses) {
    a.set_read();
    newList.push_back(a);
  }
  return newList;
}
SymAccessList list_op_double(const SymAccessList & l0, double scalar) {
  SymAccessList newList = SymAccessList();
  for(SymAccess a : l0.accesses) {
    a.set_read();
    newList.push_back(a);
  }
  return newList;
}

SymAccessList list_op_iterator(const SymAccessList & l0, const SymIterator & i1) {
  SymAccessList newList = SymAccessList();
  for(SymAccess a : l0.accesses) {
    a.set_read();
    newList.push_back(a);
  }
  return newList;
}

SymAccessList int_op_list(int scalar, const SymAccessList & l1) {
  SymAccessList newList = SymAccessList();
  for(SymAccess a : l1.accesses) {
    a.set_read();
    newList.push_back(a);
  }
  return newList;
}

SymAccessList double_op_list(double scalar, const SymAccessList & l1) {
  SymAccessList newList = SymAccessList();
  for(SymAccess a : l1.accesses) {
    a.set_read();
    newList.push_back(a);
  }
  return newList;
}

SymAccessList iterator_op_list(const SymIterator & i0, const SymAccessList & l1) {
  SymAccessList newList = SymAccessList();
  for(SymAccess a : l1.accesses) {
    a.set_read();
    newList.push_back(a);
  }
  return newList;
}


// Code for expressions involving two symbolic access lists. 
//This should include operations that involve a symoblic access bc it has a conversion function to symaccesslist.
SymAccessList operator + (const SymAccessList & l0, const SymAccessList & l1) {return list_op_list(l0,l1);}
SymAccessList operator - (const SymAccessList & l0, const SymAccessList & l1) {return list_op_list(l0,l1);}
SymAccessList operator * (const SymAccessList & l0, const SymAccessList & l1) {return list_op_list(l0,l1);}
SymAccessList operator / (const SymAccessList & l0, const SymAccessList & l1) {return list_op_list(l0,l1);}
SymAccessList operator % (const SymAccessList & l0, const SymAccessList & l1) {return list_op_list(l0,l1);}

// functions for list op int
SymAccessList operator + (const SymAccessList & l0, int scalar) {return list_op_int(l0,scalar);}
SymAccessList operator - (const SymAccessList & l0, int scalar) {return list_op_int(l0,scalar);}
SymAccessList operator * (const SymAccessList & l0, int scalar) {return list_op_int(l0,scalar);}
SymAccessList operator / (const SymAccessList & l0, int scalar) {return list_op_int(l0,scalar);}
SymAccessList operator % (const SymAccessList & l0, int scalar) {return list_op_int(l0,scalar);}

//functions for list op double
SymAccessList operator + (const SymAccessList & l0, double scalar) {return list_op_double(l0,scalar);}
SymAccessList operator - (const SymAccessList & l0, double scalar) {return list_op_double(l0,scalar);}
SymAccessList operator * (const SymAccessList & l0, double scalar) {return list_op_double(l0,scalar);}
SymAccessList operator / (const SymAccessList & l0, double scalar) {return list_op_double(l0,scalar);}
SymAccessList operator % (const SymAccessList & l0, double scalar) {return list_op_double(l0,scalar);}

//functions for list op const SymIterator
SymAccessList operator + (const SymAccessList & l0, const SymIterator & i1) {return list_op_iterator(l0,i1);}
SymAccessList operator - (const SymAccessList & l0, const SymIterator & i1) {return list_op_iterator(l0,i1);}
SymAccessList operator * (const SymAccessList & l0, const SymIterator & i1) {return list_op_iterator(l0,i1);}
SymAccessList operator / (const SymAccessList & l0, const SymIterator & i1) {return list_op_iterator(l0,i1);}
SymAccessList operator % (const SymAccessList & l0, const SymIterator & i1) {return list_op_iterator(l0,i1);}

//functions for int op list
SymAccessList operator + (int scalar, const SymAccessList & l1) {return int_op_list(scalar,l1);}
SymAccessList operator - (int scalar, const SymAccessList & l1) {return int_op_list(scalar,l1);}
SymAccessList operator * (int scalar, const SymAccessList & l1) {return int_op_list(scalar,l1);}
SymAccessList operator / (int scalar, const SymAccessList & l1) {return int_op_list(scalar,l1);}
SymAccessList operator % (int scalar, const SymAccessList & l1) {return int_op_list(scalar,l1);}

//functions for double op list
SymAccessList operator + (double scalar, const SymAccessList & l1) {return double_op_list(scalar,l1);}
SymAccessList operator - (double scalar, const SymAccessList & l1) {return double_op_list(scalar,l1);}
SymAccessList operator * (double scalar, const SymAccessList & l1) {return double_op_list(scalar,l1);}
SymAccessList operator / (double scalar, const SymAccessList & l1) {return double_op_list(scalar,l1);}
SymAccessList operator % (double scalar, const SymAccessList & l1) {return double_op_list(scalar,l1);}

//functions for SymIterator op list
SymAccessList operator + (const SymIterator & i0, const SymAccessList & l1) {return iterator_op_list(i0,l1);}
SymAccessList operator - (const SymIterator & i0, const SymAccessList & l1) {return iterator_op_list(i0,l1);}
SymAccessList operator * (const SymIterator & i0, const SymAccessList & l1) {return iterator_op_list(i0,l1);}
SymAccessList operator / (const SymIterator & i0, const SymAccessList & l1) {return iterator_op_list(i0,l1);}
SymAccessList operator % (const SymIterator & i0, const SymAccessList & l1) {return iterator_op_list(i0,l1);}




} //namespace RAJA
