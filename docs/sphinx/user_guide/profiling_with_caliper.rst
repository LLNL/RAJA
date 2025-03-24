.. ##
.. ## Copyright (c) 2016-25, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _profiling-with-Caliper-label:

************************
Profiling with Caliper
************************

To aid in profiling the RAJA abstraction layer has dynamic and static plugin support with the Caliper performance library through RAJA's plugin mechanism :ref:`feat-plugins-label`. Caliper is developed at LLNL and freely available on GitHub, see `Caliper GitHub <https://github.com/LLNL/Caliper>`_ .
Caliper provides a common interface for various vendor profiling tools, its own built in performance reports.

In this section we will demonstrate how to configure RAJA with Caliper, run simple examples with kernel performance,
and finally use the Thicket library, also developed at LLNL and freely available on GitHub, see `Thicket GitHub <https://github.com/LLNL/Thicket>`_ .
For more detailed tutorials we refer the reader to the Caliper and Thicket tutorials.


=================================
Building and running with Caliper
=================================
Caliper serves as a portable profiling library which may be configured with various vendor options. For the most up to date
configuration options we refer the reader to the `Caliper GitHub <https://github.com/LLNL/Caliper>`_  page.
For the following examples we use Caliper v2.12.1 and configure on three, CPU only, NVTX, and ROCTX enabled different platorms::

  //Basic CPU using default build parameters
  cmake ../

  //With NVTX ON
  cmake -DWITH_NVTX=ON ../

  //With ROCTX ON
  cmake -DWITH_ROCTX=ON ../

Building RAJA with Caliper enabled requires pointing RAJA to the Caliper shared cmake file and enabling plugins::

  cmake –DRAJA_ENABLE_RUNTIME_PLUGINS=ON –Dcaliper_DIR=${CALIPER_BUILD_DIR}/share/cmake/caliper ../

As a quick build check we can run the basic Caliper RAJA::forall example::

  //Example make
  make raja-forall-caliper

Finally, we can run Caliper annotated RAJA::

  //Run with raja-forall-caliper example!
  CALI_CONFIG=runtime-report ./bin/raja-forall-caliper

At the end of the the run the program will output the following runtime information::

  Path                     Time (E) Time (I) Time % (E) Time % (I)
  C-version elapsed time   0.000820 0.000820   0.395169   0.395169
  RAJA Seq daxpy Kernel    0.000655 0.000655   0.315502   0.315502
  RAJA SIMD daxpy Kernel   0.000611 0.000611   0.294629   0.294629
  RAJA OpenMP daxpy Kernel 0.013691 0.013691   6.598422   6.598422
  RAJA CUDA daxpy Kernel   0.000118 0.000118   0.056827   0.056827
