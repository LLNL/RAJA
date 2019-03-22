
.PHONY: all reproduce real clean

HOST_CXX=/usr/tce/packages/clang/clang-coral-2018.08.08/bin/clang++
CXX=/usr/tce/packages/cuda/cuda-9.2.148/bin/nvcc -ccbin $(HOST_CXX)

RAJA_INCLUDE_FLAGS=-I/path/to/RAJA/install/include
CXX_FLAGS=-arch sm_70 -std=c++11 -lineinfo -g -O3 --expt-extended-lambda -m64

RAJA_DLINK_FLAGS=-L/path/to/RAJA/install/lib -lRAJA


all: reproduce real

reproduce: reproduce_dlink.o

real: real_dlink.o

reproduce.o: reproduce.cu
	$(CXX) -dc -o reproduce.o reproduce.cu

reproduce_dlink.o: reproduce.o
	$(CXX) -dlink -o reproduce_dlink.o reproduce.o -res-usage


real.o: real.cc
	$(CXX) $(CXX_FLAGS) $(RAJA_INCLUDE_FLAGS) -x=cu -dc -o real.o real.cc

real_dlink.o: real.o
	$(CXX) $(CXX_FLAGS) $(RAJA_DLINK_FLAGS) -dlink -o real_dlink.o real.o -res-usage


clean:
	rm -f reproduce.o reproduce_dlink.o real.o real_dlink.o
