
default: all

CLANG_JIT ?= clang++
RAJA_DIR ?= /usr/workspace/wsrzd/dzpolia/raja_jit_install/

all: simple_test simple_test_nojit

simple_test: simple_test.cpp
	clang++  -L$(RAJA_DIR)/lib -lRAJA -DGTEST_HAS_DEATH_TEST=0 -I$(RAJA_DIR)/include -isystem /usr/workspace/wsrzd/dzpolia/src/raja/blt/thirdparty_builtin/googletest-master-2018-02-01/googletest/include -I/usr/workspace/wsrzd/dzpolia/src/raja/include -I/usr/workspace/wsrzd/dzpolia/src/raja/build/include -lRAJA -lstdc++ -fjit -O3  -Wall -Wextra  -fPIE -DUSE_JIT -std=c++14 simple_test.cpp -lstdc++ -lRAJA -o simple_test

simple_test_nojit: simple_test.cpp
	/usr/workspace/wsrzd/dzpolia/jit_compiler_squeeee/bin/clang++  -L$(RAJA_DIR)/lib -lRAJA -DGTEST_HAS_DEATH_TEST=0 -I$(RAJA_DIR)/include -lRAJA -lstdc++ -fjit -O3  -Wall -Wextra  -fPIE -DNO_JIT -std=c++14 simple_test.cpp -lstdc++ -lRAJA -o simple_test_nojit
