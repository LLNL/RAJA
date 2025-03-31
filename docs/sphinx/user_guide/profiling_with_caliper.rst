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

To aid in profiling, the RAJA abstraction layer has dynamic and static plugin support for the Caliper performance library.
We refer the reader to the :ref:`feat-plugins-label` section for more on RAJA plugins. Caliper is developed at LLNL and freely available on GitHub,
see `Caliper GitHub <https://github.com/LLNL/Caliper>`_ . Caliper provides a common interface for various vendor profiling tools and its own built-in performance
reports.

In this section we will demonstrate how to configure RAJA with Caliper, run a simple examples with kernel profiling,
and a quick snapshot of hotspots of using the Thicket libary. The Thicket library is also developed at LLNL and freely available on GitHub,
see `Thicket GitHub <https://github.com/LLNL/Thicket>`_ .
This page is focused as a quickstart guide, for more detailed tutorials we refer the reader to the Caliper and Thicket tutorials.


=================================
Building and running with Caliper
=================================
Caliper serves as a portable profiling library which may be configured with various vendor options. For the most up to date
configuration options we refer the reader to the `Caliper GitHub <https://github.com/LLNL/Caliper>`_  page.
For the following examples we use Caliper v2.12.1 and configure on three different platforms; CPU only, NVTX, and ROCTX::

  //Basic CPU using default build parameters
  cmake ../

  //With NVTX ON
  cmake -DWITH_NVTX=ON ../

  //With ROCTX ON
  cmake -DWITH_ROCTX=ON ../

Building RAJA with Caliper enabled requires pointing RAJA to the Caliper shared cmake file and enabling plugins::

  cmake –DRAJA_ENABLE_RUNTIME_PLUGINS=ON –Dcaliper_DIR=${CALIPER_BUILD_DIR}/share/cmake/caliper ../

As a quick build check we build the basic Caliper RAJA-Caliper example::

  make raja-forall-caliper

Finally, the Caliper annotated RAJA example may be executed via::

  CALI_CONFIG=runtime-report ./bin/raja-forall-caliper

If we build with NVTX enabled Caliper and CUDA enabled RAJA the end of the the run the program should output
the following runtime information::

  Path                     Time (E) Time (I) Time % (E) Time % (I)
  C-version elapsed time   0.000820 0.000820   0.395169   0.395169
  RAJA Seq daxpy Kernel    0.000655 0.000655   0.315502   0.315502
  RAJA SIMD daxpy Kernel   0.000611 0.000611   0.294629   0.294629
  RAJA OpenMP daxpy Kernel 0.013691 0.013691   6.598422   6.598422
  RAJA CUDA daxpy Kernel   0.000118 0.000118   0.056827   0.056827

The (E) column corresponds to exclusive timing; while the (I) column will correspond to inclusive timing.
The left two columns are absolute time in seconds while the right two columns correspond to percentage of time
within the program.

.. note:: RAJA methods which offload to the GPU will require a synchronous GPU policy to ensure that the kernel
          has completed prior to the total runtime being reported. For asynchronous kernels Caliper offers the
          `hip.gputime` or `cuda.gputime` service which may be added to CALI_CONFIG to accurately capture kernel
          time.

========================================
Profiling RAJA kernels via kernel naming
========================================
Caliper annotations of RAJA kernels work through the RAJA kernel naming mechanism currenly only supported in forall
and launch. The ``RAJA::expt::Name`` container holds a string and used for profiling in Caliper. Kernels
which are not provided a name are ommited from Caliper profiling::

    RAJA::forall<RAJA::seq_exec>(RAJA::RangeSegment(0, N),
    RAJA::expt::Name("RAJA Seq daxpy Kernel"), [=] (int i) {

        a[i] += b[i] * c;

  });

.. note:: The RAJA Name feature lives under the expt namespace as it part of a new param reducer interface.
          It will be removed from expt once the new reducer interface has matured.

.. note:: The RAJA Name must be placed before the lambda to ensure proper behavior.


=============================================
Basic integration with vendor profiling tools
=============================================
Once Caliper is configured with RAJA and kernels are provided with a kernel name, the Caliper library provides various
services to assist developers better understand their codes. For example the following command
`CALI_CONFIG=cuda-activity-report,show_kernels ./bin/raja-forall-caliper` will report all CUDA related activity
within an exectuable::

  Path                     Kernel                                           Host Time GPU Time GPU %
  C-version elapsed time                                                     0.000744
  RAJA Seq daxpy Kernel                                                      0.000783
  RAJA SIMD daxpy Kernel                                                     0.000704
  RAJA OpenMP daxpy Kernel                                                   0.009124
  cudaMalloc                                                                 0.128423
  cudaMemcpy                                                                 0.002385 0.001757 73.662910
  cudaStreamCreate                                                           0.000230
  RAJA CUDA daxpy Kernel
  |-                                                                         0.000159
  |-                      void RAJA::policy::cuda~~}::detail::Name>)            0.000038
  cudaLaunchKernel
   |-                                                                        0.000066
   |-                    void RAJA::policy::cuda~~}::detail::Name>)            0.000038
  cudaStreamSynchronize                                                      0.000050
  cudaFree                                                                   0.000495

A simiar command may be found for HIP kernels under the Caliper documentation.
  
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generating an NVIDIA Nsight Systems file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Caliper can also be used to generate a NVIDIA Nsight Systems file. The following command will generate the file and annotate the CUDA
kernel region::

CALI_SERVICES_ENABLE=nvtx,cuda nsys profile -o my_profile ./bin/raja-forall-caliper

The nsys file may then be read into the NVIDIA Nsight toolkit. 

.. image:: figures/CUDA_profiling.png

^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Generating a ROCM trace file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Configuring a ROCM trace file with Caliper may be done using the following command::

  CALI_SERVICES_ENABLE=roctx,rocm,trace rocprof --hip-trace --roctx-trace ./bin/raja-forall-caliper

The trace file may then be read into the Chrome tracer tool as demonstrated below. 

.. image:: figures/ROCM_profiling.png

^^^^^^^^^^^^^^^^^^^^^^^
Generating Hatchet file
^^^^^^^^^^^^^^^^^^^^^^^

Lastly, the following command will generate a .cali file which can be read into the Thicket library::

  CALI_CONFIG=hatchet-region-profile ./bin/raja-forall-caliper


Using python jupyter notebooks and the Hatchet library we can load the .cali file and begin analysing
performance of our application. The following sequence of commands will generate a color coded tree
high-lighting hotspots (in red) of our application::

  caliper_file = region_profile.cali
  gf = ht.GraphFrame.from_caliperreader(caliper_file)
  print(gf.tree())

Within the jupyter notebook the color coded tree will look like this:  
  
.. image:: figures/Hatchet_tree.png

=============
Final remarks
=============
The capabilities of Caliper and Hatchet exceed what has been presented here. Our main goal was to provide
users a quick start guide to building and profiling with Caliper and Hatchet. We highly recommend exploting
the `Caliper GitHub <https://github.com/LLNL/Caliper>`_  and `Thicket GitHub <https://github.com/LLNL/Thicket>`_
pages for the latest versions and links to documentation. Finally, a full-fledged tutorial is avaible of LLNL profiling
tools on YouTube `2024 Caliper, Hatchet, and Thicket <https://youtu.be/qVmxDOxM9Ws?feature=shared>`_. 
