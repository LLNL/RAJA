.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _teamsbasic-label:

------------------------------------
Naming kernels for NVTX/ROCTX tools
------------------------------------

There are no exercise files to work through for this section. Instead, there
is an example source file ``RAJA/examples/teams_reductions.cpp`` which
contains complete code examples of the concepts described here.

Key RAJA feature shown in the following example:

  *  Naming kernels using the ``Grid`` object in ``RAJA::ext::Launch`` methods.

In this example, we illustrate kernel naming capabilities within the RAJA Launch
framework for use with NVTX or ROCTX region naming capabilities.

To name a ``RAJA::expt::launch`` kernel, a string name is passed as the 
third argument to the ``RAJA::expt::Grid`` constructor::

  RAJA::expt::launch<launch_policy>(RAJA::expt::ExecPlace ,
    RAJA::expt::Grid(RAJA::expt::Teams(Nteams,Nteams),
                     RAJA::expt::Threads(Nthreads,Nthreads)
                     "myKernel"),
    [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

      /* Kernel body code goes here */

    }
  );
  
The kernel name is used to create NVTX (NVIDIA) or ROCTX (AMD) ranges enabling
developers to identify kernels using NVIDIA `Nsight <https://developer.nvidia.com/nsight-visual-studio-edition>`_
and NVIDIA `nvprof <https://docs.nvidia.com/cuda/profiler-users-guide/index.html>`_ profiling
tools or `ROCm <https://rocmdocs.amd.com/en/latest/ROCm_Tools/ROCm-Tools.html>`_
profiling tools when using ROCTX.  As an illustration, nvprof
kernels are identified as ranges of GPU activity using the provided kernel 
name::

  ==73220== NVTX result:
  ==73220==   Thread "<unnamed>" (id = 290832)
  ==73220==     Domain "<unnamed>"
  ==73220==       Range "myKernel"
              Type  Time(%)      Time     Calls       Avg       Min       Max  Name
              Range:  100.00%  32.868us         1  32.868us  32.868us  32.868us  myKernel
     GPU activities:  100.00%  2.0307ms         1  2.0307ms  2.0307ms  2.0307ms  _ZN4RAJA4expt17launch_global_fcnIZ4mainEUlNS0_13LaunchContextEE_EEvS2_T_
          API calls:  100.00%  27.030us         1  27.030us  27.030us  27.030us  cudaLaunchKernel

Similarly, ROCm tools can be used to generate traces of a profile and
the resulting json file can be viewed using tools such as `Perfetto
<https://ui.perfetto.dev/#!/>`_.

In future work, we plan to add support to other profiling tools. Thus, API 
changes may occur based on user feedback and integration with other tools. 
Enabling NVTX profiling with RAJA Launch requires RAJA to be configured with 
RAJA_ENABLE_NV_TOOLS_EXT=ON.
or RAJA_ENABLE_ROCTX=ON for ROCTX profiling on AMD platforms platforms.
