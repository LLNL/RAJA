.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _launchintro-label:

------------------------------
``RAJA::expt::Launch`` Basics
------------------------------

There are no exercise files to work through for this section. Instead, there
is an example source file ``RAJA/examples/tut_launch_basic.cpp`` which
contains complete code examples of the concepts described here.

Key RAJA features shown in the following examples are:

  * ``RAJA::expt::launch`` method to create a run-time
    selectable host/device execution space.
  * ``RAJA::expt::loop`` methods to express algorithms
    in terms of nested for loops.

In this example, we introduce the RAJA Launch framework and discuss
hierarchical loop-based parallelism. The RAJA Launch API currently lives
under the RAJA experimental namespace, ``RAJA::expt``, and will be
promoted to the standard RAJA namespace in the next release.
Development with RAJA Launch occurs inside an execution space.
The kernel execution space is dispatched using the
``RAJA::expt::launch`` method::

  RAJA::expt::launch<launch_policy>(RAJA::expt::ExecPlace ,
  RAJA::expt::Grid(RAJA::expt::Teams(Nteams,Nteams),
                        RAJA::expt::Threads(Nthreads,Nthreads)),
  [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

    /* Kernel code goes here */

  });

The ``RAJA::expt::launch`` method can be templated on up to two policies 
(host and device). As an example, the following creates an execution space 
for a sequential and CUDA kernel dispatch::

  using launch_policy = RAJA::expt::LaunchPolicy
    <RAJA::expt::seq_launch_t, RAJA::expt::cuda_launch_t<false>>;

Whether a kernel executes on the host or device is determined by the first 
argument passed to the ``RAJA::expt::launch`` method, which is a 
``RAJA::expt::ExecPlace`` enum value, either ``HOST`` or ``DEVICE``.
Similar to GPU thread and block programming models, RAJA Launch carries out
computation in a predefined compute grid made up of threads which are
then grouped into teams when executing on the device. The execution space is 
then enclosed by a host/device lambda which takes a 
``RAJA::expt::LaunchContext`` object, which may be used to control the flow 
within the kernel, for example by creating thread-team synchronization points.

Inside the execution space, developers express a kernel using nested
``RAJA::expt::loop`` methods. The manner in which each loop is executed 
is determined by the ``RAJA::expt::loop`` template parameter type, which
indicates how the corresponding iterates are mapped to the Teams/Threads
configuration defined by the ``RAJA::expt::Grid`` type passed as the second
argument to the ``RAJA::expt::launch`` method. Following the CUDA/HIP 
programming models we follow a hierarchical structure in which outer loops 
are executed by thread-teams and inner loops are executed by a thread in a team.

.. literalinclude:: ../../../../examples/tut_launch_basic.cpp
   :start-after: // _team_loops_start
   :end-before: // _team_loops_end
   :language: C++

The mapping between teams and threads and teams to the underlying programming 
model depends on how the ``RAJA::expt::loop`` template parameter types are
defined. For example, we may define host and device mapping strategies as::

  using teams_x = RAJA::expt::LoopPolicy<RAJA::loop_exec,
                                         RAJA::cuda_block_x_direct>;
  using thread_x = RAJA::expt::LoopPolicy<RAJA::loop_exec,
                                          RAJA::cuda_block_x_direct>;

Here, the ``RAJA::expt::LoopPolicy`` type holds both the host and device loop 
mapping strategies. On the host, both the team/thread strategies expand
out to standard C-style loops for execution:

.. literalinclude:: ../../../../examples/tut_launch_basic.cpp
   :start-after: // _c_style_loops_start
   :end-before: // _c_style_loops_end
   :language: C++

On the device the ``teams_x/y`` policies will map loop iterations directly to
CUDA (or HIP) thread blocks, while the ``thread_x/y`` policies will map loop 
iterations directly to threads in a CUDA (or HIP) thread block. The direct CUDA 
equivalent of the kernel body using the policy shown above is:

.. literalinclude:: ../../../../examples/tut_launch_basic.cpp
   :start-after: // _device_loop_start
   :end-before: // _device_loop_end
   :language: C++
