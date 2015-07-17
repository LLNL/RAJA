#
#  This is cheesy, but it's simple and easy enough to muck with for testing...
#

#
#  Modify stuff in the following sections to fit your machine and compiler.
#
#  Uncomment ONE AND ONLY ONE opt/debug option and "IndexSet_ARCH" 
#  variable line for the platform you want. 
#

#SIMD_TYPE = -msse4.1
SIMD_TYPE = -mavx

#
# To build and run on IPA:
#   %module load cudatoolkit/7.5
#   %use gcc-4.9.0p
#   %make
#   %salloc -p nvidia -N1
#   %setenv CUDA_VISIBLE_DEVICES 0
#   %<run code here>
#
CXX_COMPILE = nvcc -O2 --expt-extended-lambda -arch compute_35 -std=c++11 -DUSE_CUDA=1 -v -keep 
#CXX_COMPILE = nvcc -O2 --extended-lambda -arch compute_35 -std=c++11 -DUSE_CUDA=1 -v -keep 
#CXX_COMPILE = g++ -g -fpermissive -fopenmp -O0 -std=c++11 -fopenmp 

#CXX 		= icpc
CXX = nvcc

TESTOBJS = IndexSet.o luleshRAJA.o 

lulesh: $(TESTOBJS) 
	$(CXX_COMPILE) $(TESTOBJS) -o  lulesh-RAJA-CUDA.exe

%.o : %.cu ; $(CXX_COMPILE) -c -o $@ $< 
#%.o : %.cxx ; $(CXX_COMPILE) -S -o $@ $< 
## The following assembly options are for Intel compiler...
#%.o : %.cxx ; $(CXX_COMPILE) -S -fcode-asm -vec-report3 -o $@ $< $(CXXFLAGS)

clean: 
	rm -rf *.o *.exe

clean-all: clean 
	rm -rf *.cpp*.i* *.cuda*.* *.fatbin* *.hash* *.module_id* *.ptx *.reg.c


