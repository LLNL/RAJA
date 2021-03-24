.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _configopt-label:

****************************
Build Configuration Options
****************************

RAJA uses `BLT <https://github.com/LLNL/blt>`_, a CMake-based build system.
In :ref:`getting_started-label`, we described how to run CMake to configure
RAJA with its default option settings. In this section, we describe all RAJA
configuration options, their defaults, and how to enable or disable features.

=======================
Setting Options
=======================

The RAJA configuration can be set using standard CMake variables along with
BLT and RAJA-specific variables. For example, to make a release build with 
some system default GNU compiler and then install the RAJA header files and
libraries in a specific directory location, you could do the following in 
the top-level RAJA directory::

    $ mkdir build-gnu-release
    $ cd build-gnu-release
    $ cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_C_COMPILER=gcc \
      -DCMAKE_CXX_COMPILER=g++ \
      -DCMAKE_INSTALL_PREFIX=../install-gnu-release ../
    $ make
    $ make install

Following CMake conventions, RAJA supports three build types: ``Release``, 
``RelWithDebInfo``, and ``Debug``. With CMake, compiler flags for each of
these build types are applied automatically and so you do not have to 
specify them. However, if you want to apply other compiler flags, you will
need to do that using appropriate CMake variables.

All RAJA options are set like regular CMake variables. RAJA settings for 
default options, compilers, flags for optimization, etc. can be found in files 
in the ``RAJA/cmake`` directory and top-level ``CMakeLists.txt`` file. 
Configuration variables can be set by passing
arguments to CMake on the command line when CMake is called, or by setting
options in a CMake *cache file* and passing that file to CMake using the 
CMake ``-C`` options. For example, to enable RAJA OpenMP functionality, 
pass the following argument to CMake::

    -DENABLE_OPENMP=On

The RAJA repository contains a collection of CMake cache files 
(we call them *host-config* files) that may be used as a guide for users trying
to set their own options. See :ref:`configopt-raja-hostconfig-label`.

Next, we summarize RAJA options and their defaults.


.. _configopt-raja-features-label:

====================================
Available RAJA Options and Defaults
====================================

RAJA uses a variety of custom variables to control how it is compiled. Many 
of these are used internally to control RAJA compilation and do 
not need to be set by users. Others can be used to enable or disable certain 
RAJA features. Most variables get translated to 
compiler directives and definitions in the RAJA ``config.hpp`` file that is 
generated when CMake runs. The ``config.hpp`` header file is included in other 
RAJA headers as needed so all options propagate consistently through the 
build process for all of the code. Each RAJA variable has a special prefix 
to distinguish it as being specific to RAJA; i.e., it is not a BLT variable
or a standard CMake variable.

The following tables describe which variables set RAJA options and 
and their default settings:

* **Examples, tests, warnings, etc.**

     Variables that control whether RAJA tests, examples, or tutorial
     exercises are built when RAJA is compiled:

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_ENABLE_TESTS        On 
      RAJA_ENABLE_EXAMPLES     On 
      RAJA_ENABLE_EXERCISES    On 
      RAJA_ENABLE_BENCHMARKS   Off
      ======================   ======================

     RAJA can also be configured to build with compiler warnings reported as
     errors, which may be useful to make sure your application builds cleanly:

      =========================   ======================
      Variable                    Default
      =========================   ======================
      ENABLE_WARNINGS_AS_ERRORS   Off
      =========================   ======================

     RAJA Views/Layouts may be configured to check for out of bounds 
     indexing at runtime:

      =========================   ======================
      Variable                    Default
      =========================   ======================
      RAJA_ENABLE_BOUNDS_CHECK    Off
      =========================   ======================

     Note that RAJA bounds checking is a runtime check and will add 
     execution time overhead. Thus, this feature should not be enabled 
     for release builds.
     
* **Programming model back-ends**

     Variables that control which RAJA programming model back-ends are enabled
     are (names are descriptive of what they enable):

      =======================   ============================================
      Variable                  Default
      =======================   ============================================
      ENABLE_OPENMP             On
      ENABLE_TARGET_OPENMP      Off (when on, ENABLE_OPENMP must also be on)
      ENABLE_TBB                Off
      ENABLE_CUDA               Off
      ENABLE_HIP                Off
      =======================   ============================================

     Other compilation options are available via the following:

      =======================   ==========================================
      Variable                  Default
      =======================   ==========================================
      ENABLE_CLANG_CUDA         Off (when on, ENABLE_CUDA must also be on)
      ENABLE_EXTERNAL_CUB       Off (when CUDA enabled)
      CUDA_ARCH                 sm_35 (set based on hardware support)
      ENABLE_EXTERNAL_ROCPRIM   Off (when HIP enabled)
      =======================   ==========================================

      Turning the 'ENABLE_CLANG_CUDA' variable on will build CUDA code with
      the native support in the Clang compiler.

      The 'ENABLE_EXTERNAL_CUB' variable is used to require the use of an
      external install of the NVIDIA CUB support library. Even when Off the CUB
      library included in the CUDA toolkit will still be used if available.
      Starting with CUDA 11, CUB is installed as part of the CUDA toolkit and
      the NVIDIA THRUST library requires that install of CUB. We recommended
      projects use the CUB included with the CUDA toolkit for compatibility with
      THRUST and applications using THRUST. Users should take note of the CUB
      install used by RAJA to ensure they use the same include directories when
      configuring their application.

      The 'ENABLE_EXTERNAL_ROCPRIM' variable is used to require an external
      install of the AMD rocPRIM support library. Even when Off the rocPRIM
      library included in the ROCM install will be used when available. We
      recommend projects use the rocPRIM included with the ROCM install when
      available. Users should take note of the rocPRIM install used by RAJA to
      ensure they use the same include directories when configuring their
      application.

.. note:: See :ref:`getting-started-label` for more information about
          setting other options for RAJA back-ends.

* **Data types, sizes, alignment, etc.**

     RAJA provides type aliases that can be used to parameterize floating 
     point types in applications, which makes it easier to switch between types.

     The following variables are used to set the data type for the type
     alias ``RAJA::Real_type``:

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_USE_DOUBLE          On 
      RAJA_USE_FLOAT           Off 
      ======================   ======================

     Similarly, the 'RAJA::Complex_type' can be enabled to support complex 
     numbers if needed:

      ======================   ======================
      Variable                 Default
      ======================   ======================
      RAJA_USE_COMPLEX         Off 
      ======================   ======================

     When turned on, the RAJA Complex_type is 'std::complex<Real_type>'.

     There are several variables to control the definition of the RAJA 
     floating-point data pointer type ``RAJA::Real_ptr``. The base data type
     is always ``Real_type``. When RAJA is compiled for CPU execution 
     only, the defaults are:

      =============================   ======================
      Variable                        Default
      =============================   ======================
      RAJA_USE_BARE_PTR               Off
      RAJA_USE_RESTRICT_PTR           On
      RAJA_USE_RESTRICT_ALIGNED_PTR   Off
      RAJA_USE_PTR_CLASS              Off
      =============================   ======================

     When RAJA is compiled with CUDA enabled, the defaults are:

      =============================   ======================
      Variable                        Default
      =============================   ======================
      RAJA_USE_BARE_PTR               On
      RAJA_USE_RESTRICT_PTR           Off
      RAJA_USE_RESTRICT_ALIGNED_PTR   Off
      RAJA_USE_PTR_CLASS              Off
      =============================   ======================

     The meaning of these variables is:

      =============================   ========================================
      Variable                        Meaning
      =============================   ========================================
      RAJA_USE_BARE_PTR               Use standard C-style pointer
      RAJA_USE_RESTRICT_PTR           Use C-style pointer with restrict
                                      qualifier
      RAJA_USE_RESTRICT_ALIGNED_PTR   Use C-style pointer with restrict
                                      qualifier and alignment attribute 
                                      (see RAJA_DATA_ALIGN below)
      RAJA_USE_PTR_CLASS              Use pointer class with overloaded `[]` 
                                      operator that applies restrict and 
                                      alignment intrinsics. This is useful 
                                      when a compiler does not support 
                                      attributes in a typedef.
      =============================   ========================================

     RAJA internally uses a parameter to define platform-specific constant
     data alignment. The variable that control this is:

      =============================   ======================
      Variable                        Default
      =============================   ======================
      RAJA_DATA_ALIGN                 64
      =============================   ======================

     What this variable means:

      =============================   ========================================
      Variable                        Meaning
      =============================   ========================================
      RAJA_DATA_ALIGN                 Specifies data alignment used in 
                                      intrinsics and typedefs; 
                                      units of **bytes**.
      =============================   ========================================

     For details on the options in this section are used, please see the 
     header file ``RAJA/include/RAJA/util/types.hpp``.

* **Timer Options**

     RAJA provides a simple portable timer class that is used in RAJA
     example codes to determine execution timing and can be used in other apps
     as well. This timer can use any of three internal timers depending on
     your preferences, and one should be selected by setting the 'RAJA_TIMER'
     variable. If the 'RAJA_USE_CALIPER' variable is turned on (off by default),
     the timer will also offer Caliper-based region annotations. Information
     about using Caliper can be found at 
     `Caliper <https://github.com/LLNL/Caliper>`_ 

      ======================   ======================
      Variable                 Values
      ======================   ======================
      RAJA_TIMER               chrono (default)
                               gettime
                               clock
      ======================   ======================

     What these variables mean:

      =============================   ========================================
      Value                           Meaning
      =============================   ========================================
      chrono                          Use the std::chrono library from the 
                                      C++ standard library
      gettime                         Use `timespec` from the C standard 
                                      library time.h file
      clock                           Use `clock_t` from time.h
      =============================   ========================================

* **Other RAJA Features**
   
     RAJA contains some features that are used mainly for development or may
     not be of general interest to RAJA users. These are turned off be default.
     They are described here for reference and completeness.

      =============================   ========================================
      Variable                        Meaning
      =============================   ========================================
      ENABLE_FT                       Enable/disable RAJA experimental
                                      loop-level fault-tolerance mechanism
      RAJA_REPORT_FT                  Enable/disable a report of fault-
                                      tolerance enabled run (e.g., number of 
                                      faults detected, recovered from, 
                                      recovery overhead, etc.)
     RAJA_ENABLE_RUNTIME_PLUGINS           Enable support for dynamically loading
                                      RAJA plugins.
      =============================   ========================================


.. _configopt-raja-backends-label:

===============================
Setting RAJA Back-End Features
===============================

Various `ENABLE_*` options are listed above for enabling RAJA back-ends,
such as OpenMP and CUDA. To access compiler and hardware optimization features,
it may be necessary to pass additional options to CMake. Please see
:ref:`getting_started-label` for more information. 
