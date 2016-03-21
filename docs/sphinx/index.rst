.. #######################################################################
.. #
.. # Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. #
.. # Produced at the Lawrence Livermore National Laboratory.
.. #
.. # All rights reserved.
.. #
.. # This source code cannot be distributed without permission and
.. # further review from Lawrence Livermore National Laboratory.
.. #
.. #######################################################################


***********************
What is RAJA?
***********************

RAJA is a collection of C++ software abstractions, being developed at
Lawrence Livermore National Laboratory (LLNL), that enable architecture 
portability for HPC applications. The overarching goals of RAJA are to:

  * make existing (production) applications *portable with minimal disruption*
  * provide a model for new applications so that they are portable from 
    inception. 

RAJA uses standard C++11 -- C++ is the predominant programming language in
which many LLNL codes are written. RAJA shares goals and concepts found in 
other C++ portability abstraction approaches, such as 
`Kokkos <https://github.com/kokkos/kokkos>`_
and `Thrust <https://developer.nvidia.com/thrust>`_. RAJA is rooted in
a perspective based on substantial experience working on production 
mesh-based multiphysics applications at LLNL. It provides constructs that 
are absent in other models and which are fundamental in those codes. Also, 
another goal of RAJA is enable application developers to adapt RAJA concepts 
and specialize them for different code implementation patterns and C++ usage, 
since data structures and algorithms vary widely across applications.

It is important to note that RAJA is very much a work-in-progress.
The community of researchers and application developers at LLNL that are
actively contributing to it and developing new capabilities is growing.
The publically-released version contains only core pieces of RAJA as they
exist today. While the basic interfaces are fairly stable, the implementation
of the underlying concepts is being refined. Additional features will appear 
in future releases.

The core developers of RAJA are:

  * Rich Hornung (hornung1@llnl.gov)
  * Jeff Keasler (keasler1@llnl.gov)

If you have questions, or have ideas about expanding the applicability of
RAJA and are interested in contributing to its development, please do not
hesitate to contact us. We are always interested in exploring new ways to
use RAJA. 
 
Other contributors include:

  * David Beckingsale (beckingsale1@llnl.gov)
  * Holger Jones (jones19@llnl.gov)
  * Adam Kunen (kunen1@llnl.gov)
  * Olga Pearce (pearce8@llnl.gov)


=============
Introduction
=============

The main conceptual abstraction in RAJA is a loop. A typical large 
multiphysics code may contain O(10K) loops and these are where most 
computational work is performed and where most fine-grained parallelism is 
available. RAJA defines a systematic loop encapsulation paradigm that helps 
insulate application developers from implementation details associated with 
software and hardware platform choices. Such details include: non-portable 
compiler and platform-specific directives, parallel programming model usage 
and constraints, and hardware-specific data management. RAJA can be used 
incrementally and selectively in an application and usually works 
seamlessly with native data structures and loop traversal patterns in a code.

Typically, RAJA integration is a multi-step process involving steps of
increasing complexity. First, basic transformation of loops to the "RAJA form" 
is reasonably straightforward and can be even mundane in many cases. The 
initial transformation makes the code portable by enabling it to execute 
on both CPU and GPU hardware by choosing an appropriate parallel programming 
model back-end at compile-time. Next, loop and performance analysis leads to
refinement of *execution policy* choices. Careful categorization of loop 
patterns and workloads is key to selecting the best choices for mapping 
loop execution to hardware resources for high performance. Important 
considerations include: the relationship between data movement and 
computation (e.g., operations performed per byte moved), control flow 
branching intensity, and available parallelism in an algorithm. Earlier, 
we stated that large multiphysics codes contains O(10K) loops. However, 
such codes often contain only O(10) *loop patterns*. Thus, a goal of 
transforming a code to RAJA should be to assign loops to execution 
equivalence classes so that similar loops may be mapped to particular
parallel execution choices or hardware resources in a consistent fashion.

Like other C++-based portability abstraction approaches, RAJA relies 
on decoupling the body of a loop from the mechanism that executes the loop; 
i.e., its *traversal*. The decoupling allows a traversal method to be applied 
to many different loop bodies and different traversals to be explored for a
given loop body for different execution scenarios. In RAJA, the decoupling is 
achieved by re-casting a loop into the generally-accepted "parallel for" idiom.
For example, a traditional C-style for-loop construct is replaced by a call
to a C++ traversal template method and the loop body is passed as a C++
lambda function.

RAJA contains several cooperating encapsulation features, which include:

  * **Traversals and execution policies.** A traversal method specialized 
    with an execution policy template parameter defines how a loop will
    execute; e.g., sequentially, multi-threaded using OpenMP, or run on
    a GPU as a CUDA kernel.

  * **IndexSets.**  IndexSets enable much more flexible and powerful ways 
    to control loop iterations than simple for-loops. IndexSets allow 
    iteration order to be changed in ways which can, for example, enable 
    parallel execution of a non-data-parallel loop without rewriting the
    loop body. Typically, an IndexSet is used to partition an iteration 
    space into *Segments*, or "chunks" of iterations. Then, different 
    iteration subsets may be launched in parallel and dependencies among
    them can be managed. IndexSets also enable coordination between iteration 
    and data placement; specifically, chunks of data and iterations can be 
    mapped to hardware resources in an architecture-specific way. While 
    IndexSets are defined at runtime, code that executes the *Segment types*
    they contain is generated at compile-time. This allows compilers to 
    optimize execution of kernels for different *Segment type* implementations.
    For example, a Segment type that processes a contiguous (stride-1) index 
    range can be optimized in ways that an indirection Segment type cannot.

  * **Data type encapsulation.** RAJA provides simple data and pointer types 
    that can be used to hide non-portable compiler directives and data 
    attributes, such as alignment, restrict, etc. These compiler-specific 
    code decorations often enhance the ability of a compiler to optimize. 
    While use of these types are not required to use RAJA, they are a good 
    idea for HPC codes, in general. Using a centralized collection of 
    types eliminates the need to litter application source code with detailed, 
    non-portable syntax and enables architecture or compiler-specific 
    information to be propagated easily the source code by making local 
    changes in a header file. For any parallel reduction operation, RAJA 
    requires a *reduction class template* to be used. Template specialization 
    of a reduction in a manner similar to the way a traversal method is 
    specialized on an execution policy type enables a portable reduction 
    operation while hiding programming model-specific reduction constructs, 
    such as OpenMP vs. CUDA, from application code. 


.. seealso::

   More detailed discussion of RAJA, including its use in proxy apps included 
   in the RAJA source distribution can be found in these reports:

   `The RAJA Portability Layer: Overview and Status (2014) <file:../RAJAStatus-09.2014_LLNL-TR-661403.pdf>`_

   `RAJA Overview (extracted from ASC Tri-lab Co-design Level 2 Milestone Report 2015) <file:../RAJAOverview-Trilab-09.2015_LLNL-TR-677453.pdf>`_

    Include a recent presentation that shows additional progress after the 
    last L2.


**Contents:**

.. toctree::
   :maxdepth: 2

   code_organization
   config_build
   test_examples
   future

