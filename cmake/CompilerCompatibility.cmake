###############################################################################
# Copyright (c) 2016, Lawrence Livermore National Security, LLC.
#    
# Produced at the Lawrence Livermore National Laboratory
#    
# LLNL-CODE-689114
# 
# All rights reserved.
#  
# This file is part of RAJA.
#
# For additional details, please also read RAJA/LICENSE.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the disclaimer below.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the disclaimer (as noted below) in the
#   documentation and/or other materials provided with the distribution.
#
# * Neither the name of the LLNS/LLNL nor the names of its contributors may
#   be used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL LAWRENCE LIVERMORE NATIONAL SECURITY,
# LLC, THE U.S. DEPARTMENT OF ENERGY OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
# OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
# IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
###############################################################################

include(CheckCXXSourceCompiles)

set(OLD_CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS})
if (NOT MSVC)
  if (CMAKE_CXX_COMPILER_ID MATCHES INTEL)
    set (CMAKE_REQUIRED_FLAGS ${CMAKE_CXX_FLAGS_DEBUG})
  else ()
    set (CMAKE_REQUIRED_FLAGS "-std=c++11")
  endif()
endif()

CHECK_CXX_SOURCE_COMPILES(
"#include <type_traits>
#include <limits>

template <typename T>
struct signed_limits {
  static constexpr T min()
  {
    return static_cast<T>(1llu << ((8llu * sizeof(T)) - 1llu));
  }
  static constexpr T max()
  {
    return static_cast<T>(~(1llu << ((8llu * sizeof(T)) - 1llu)));
  }
};

template <typename T>
struct unsigned_limits {
  static constexpr T min()
  {
    return static_cast<T>(0);
  }
  static constexpr T max()
  {
    return static_cast<T>(0xFFFFFFFFFFFFFFFF);
  }
};

template <typename T>
struct limits : public std::conditional<
  std::is_signed<T>::value,
  signed_limits<T>,
  unsigned_limits<T>>::type {
};

template <typename T>
void check() {
  static_assert(limits<T>::min() == std::numeric_limits<T>::min(), \"min failed\");
  static_assert(limits<T>::max() == std::numeric_limits<T>::max(), \"max failed\");
}

int main() {
  check<char>();
  check<unsigned char>();
  check<short>();
  check<unsigned short>();
  check<int>();
  check<unsigned int>();
  check<long>();
  check<unsigned long>();
  check<long int>();
  check<unsigned long int>();
  check<long long>();
  check<unsigned long long>();
}" check_power_of_two_integral_types)

set(CMAKE_REQUIRED_FLAGS ${OLD_CMAKE_REQUIRED_FLAGS})

if(NOT check_power_of_two_integral_types)
  message(FATAL_ERROR "RAJA fast limits are unsupported for your compiler/architecture")
endif()
