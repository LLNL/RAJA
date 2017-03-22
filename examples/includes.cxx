#include <iostream>

#include "RAJA/forall.hxx"
#include "RAJA/sequential.hxx"

int main(int RAJA_UNUSED_ARG(argc), char** RAJA_UNUSED_ARG(argv[]))
{
  typedef RAJA::seq_exec execute_policy;

  int N = 512;

  double* a = new double[N];

  RAJA::forall<execute_policy>(0, N, [=](int i) {
    a[i] = 3.14;
  });

  std::cout << "Filled array with PI!" << std::endl;

  return 0;
}
