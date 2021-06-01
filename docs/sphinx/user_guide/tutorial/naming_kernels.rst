.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _teamsbasic-label:

------------------------------
Naming kernels with NVTX tools
------------------------------

Key RAJA feature shown in the following example:

  *  Naming kernels using the ``Resources`` object in ``RAJA::ext::Launch`` methods.  

In this example, we illustrate kernel naming capabilities within the RAJA Teams
framework for use with NVIDIA's NVTX tools.  

Recalling the ``RAJA::expt::launch`` API, naming a kernel is done using the third
argument of the ``Resources`` constructor as illustrated below::
  RAJA::expt::launch<launch_policy>(RAJA::expt::ExecPlace ,
  RAJA::expt::Resources(RAJA::expt::Teams(Nteams,Nteams),
                        RAJA::expt::Threads(Nthreads,Nthreads)
                        "myKernel"),
  [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

    /* Express code here */

  });
  
The kernel name is used to create NVTX ranges enabling developers to identify 
kernels using the NVIDIA `Nsight <https://developer.nvidia.com/nsight-visual-studio-edition>`_ 
and NVIDIA `Nvprof <https://docs.nvidia.com/cuda/profiler-users-guide/index.html>`_ profiling
tools. As an illustration, using Nvprof, kernels are identified as ranges of GPU activity through the 
user specified name::

  ==73220== NVTX result:
  ==73220==   Thread "<unnamed>" (id = 290832)
  ==73220==     Domain "<unnamed>"
  ==73220==       Range "myKernel"
              Type  Time(%)      Time     Calls       Avg       Min       Max  Name
              Range:  100.00%  32.868us         1  32.868us  32.868us  32.868us  myKernel
     GPU activities:  100.00%  2.0307ms         1  2.0307ms  2.0307ms  2.0307ms  _ZN4RAJA4expt17launch_global_fcnIZ4mainEUlNS0_13LaunchContextEE_EEvS2_T_
          API calls:  100.00%  27.030us         1  27.030us  27.030us  27.030us  cudaLaunchKernel

As future work we plan to add support to other profiling tools; API changes may occur 
based on user feedback and integration with other tools. Enabling NVTX profiling
with RAJA Teams requires RAJA to be configured with ENABLE_NV_TOOLS_EXT=ON.

The file RAJA/examples/teams_reductions.cpp contains a complete working example code.
