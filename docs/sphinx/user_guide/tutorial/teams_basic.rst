.. ##
.. ## Copyright (c) 2016-20, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _teamsbasic-label:

------------------------------
Team based loops (RAJA Teams)
------------------------------

Key RAJA features shown in the following examples:

  * ``RAJA::expt::launch`` method to create a run-time
    selectable host/device execution space.
  * ``RAJA::expt::loop`` methods to express algorithms
    in terms of nested for loops. 

In this example, we introduce the RAJA Teams framework and discuss
hierarchical loop-based parallism. Development with RAJA teams occurs
inside an execution space. The execution space is launched using the 
``RAJA::expt::launch`` method::

  RAJA::expt::launch<launch_policy>(RAJA::expt::ExecPlace ,
  RAJA::expt::Resources(RAJA::expt::Teams(Nteams,Nteams),
                        RAJA::expt::Threads(Nthreads,Nthreads)),
  [=] RAJA_HOST_DEVICE (RAJA::expt::LaunchContext ctx) {

    /* Express code here */

  });

The ``RAJA::expt::launch`` is templated on both a host and a device policy. As 
an example, we may consider enabling an execution space for sequential for loops 
or a CUDA kernel::

  using launch_policy = RAJA::expt::LaunchPolicy
    <RAJA::expt::seq_launch_t, RAJA::expt::cuda_launch_t<false>>;

Kernel execution on either the host or device is driven  
Kernel execution is then driven by the first argument which takes 
a ``RAJA::expt::ExecPlace`` enum type, either ``HOST`` or ``DEVICE``. 
Similar to thread, and block programming models, RAJA Teams carries out
computation in a predefined compute compute grid made of threads which are
then grouped into teams. The execution space is then enclosed by a host/device
lambda which takes a ``RAJA::expt::LaunchContext``. The ``RAJA::expt::LaunchContext``
may then be used to control flow within the kernel, for example creating thread-team
synchronization points. 
   
The building block to kernels with RAJA Teams is the ``RAJA::expt::loop``. 
``RAJA::expt::loops`` enable developers to express their code in terms of
nested for loops within an execution space. Typically, loops within the execution
space are written as nested for loops which loops over teams, and then threads within 
a team. 

.. literalinclude:: ../../../../examples/tut_teams_basic.cpp
   :start-after: // _team_loops_start
   :end-before: // _team_loops_end
   :language: C++
  
The mapping between loop iteration and thread is decided through the thread/team
policy. The team/thread types are an alias for a host and device loop mapping
strategy:: 

  using teams_x = RAJA::expt::LoopPolicy<RAJA::loop_exec,
                                         RAJA::cuda_block_x_direct>;
  using thread_x = RAJA::expt::LoopPolicy<RAJA::loop_exec, 
                                          RAJA::cuda_block_x_direct>;

The ``RAJA::expt::LoopPolicy`` struct holds both a host and device loop mapping
strategy. The ``teams_x`` policy will map loop iterations directly to CUDA thread blocks, 
while the ``thread_x`` policy will map loop iterations directly to threads in a CUDA block.
The CUDA equivalent is illustrated below:   

.. literalinclude:: ../../../../examples/tut_teams_basic.cpp
   :start-after: // _device_loop_start
   :end-before: // _device_loop_end
   :language: C++
   
On the CPU the loop policies will expand to standard C-style for execution. 
The equivalent CUDA kernel. The C-style equivalent loops are illustrated below:

.. literalinclude:: ../../../../examples/tut_teams_basic.cpp
   :start-after: // _c_style_loops_start
   :end-before: // _c_style_loops_end
   :language: C++
   
The file RAJA/examples/tut_teams_basic.cpp contains the complete working example code.
