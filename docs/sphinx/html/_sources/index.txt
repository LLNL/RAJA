
***********************
RAJA User Documentation
***********************

RAJA is a architecture portability abstraction software library for HPC
applications that is being developed at Lawrence Livermore National 
Laboratory. The overarching goals of RAJA are to:

  * make existing (production) applications *portable with minimal disruption*
  * provide a model for new application development so that they are portable 
    from inception. 

RAJA uses standard C++11 -- C++ is the predominant programming language used 
many LLNL codes. RAJA shares goals and concepts found in other C++ portability 
abstraction approaches, such as `Kokkos <https://github.com/kokkos/kokkos>`_
and `Thrust <https://developer.nvidia.com/thrust>`_.  However, RAJA provides 
constructs that are absent in other models and which are used heavily in LLNL 
mesh-based multiphysics codes. Also, another goal of RAJA is to adapt concepts 
and specialize them for different code implementation patterns and C++ usage, 
since data structures and algorithms vary widely across applications.

=============
Introduction
=============

Loops are the main conceptual abstraction in RAJA. A typical large 
multiphysics code contains O(10K) loops. These loops are where most 
computational work is performed and where most fine-grained parallelism is 
available. RAJA defines a systematic loop encapsulation paradigm that helps 
insulate application developers from implementation details associated with 
software and hardware platform choices. Such details include: non-portable 
compiler and platform-specific directives, parallel programming model usage 
and constraints, and hardware-specific data management. RAJA can be used 
incrementally and selectively in an application and works with the native 
data structures and loop traversal patterns in a code.

A typical RAJA integration approach is a multi-step process involving steps of
increasing complexity. First, basic transformation of loops to the "RAJA form" 
is reasonably straightforward and can be even mundane in many cases. The 
initial transformation makes the code portable by enabling it to execute 
on both CPU and GPU hardware by choosing various parallel programming model 
back-ends at compile-time. Second, loop *execution policy* choices can be 
refined based on an analysis of loops. Careful categorization of loop patterns 
and workloads is key to selecting the best choices for mapping loop execution 
to available hardware resources for high performance. Important considerations 
include: the relationship between data movement and computation (operations 
performed per byte moved), control flow branching intensity, and available 
parallelism. Earlier, we stated that large multiphysics codes contains O(10K) 
loops. However, such codes typically contains only O(10) *loop patterns*. 
Thus, a goal of transforming a code to RAJA should be to assign loops to 
execution equivalence classes so that similar loops may be mapped to particular
parallel execution choices or hardware resources in a consistent fashion.

Like other C++-based portability layer abstraction approaches, RAJA relies 
on decoupling the body of a loop from the mechanism that executes the loop; 
i.e., its *traversal*. This allows the same traversal method to be applied 
to many different loop bodies and different traversals to be applied to the 
same loop body for different execution scenarios. In RAJA, the decoupling is 
achieved by re-casting a loop into the generally-accepted "parallel for" idiom.

This loop decoupling enables the cooperating encapsulation features of RAJA
which include:

  * **Trversals and execution policies.** A traversal method specialized 
    with an execution policy template parameter defines how a loop will be 
    executed; e.g., sequentially, multithreaded using OpenMP, or run on
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
    mapped to individual cores in an architecture-specific way. While 
    IndexSets provide the flexibility to be defined at runtime, compilers 
    can optimize execution of kernels for different *Segment type* 
    implementations at compile-time. For example, a Segment type 
    implementation that processes contiguous index ranges can be optimized 
    in ways that an indirection Segment type implementation cannot.

  * **Data type encapsulation.** RAJA provides data and pointer types that 
    can be used to hide non-portable compiler directives and data attributes, 
    such as alignment, restrict, etc. These compiler-specific data decorations 
    often enhance a compiler's ability to optimize. While these types are not 
    required to use RAJA, they are a good idea in general for HPC codes. They 
    eliminate the need to litter a code with detailed, non-portable syntax 
    and enable architecture or compiler-specific information to be propagated 
    throughout an application code with localized changes in a header file. 
    For any parallel reduction operation, RAJA does require a 
    *reduction class template* to be used. Template specialization of a 
    reduction in a manner similar to the way a traversal method is specialized 
    on an execution policy type enables a portable reduction operation while 
    hiding programming model-specific reduction constructs, such as OpenMP
    vs. CUDA, from application code. 


.. seealso::

   More detailed discussion of RAJA, including its use in proxy apps included 
   in the RAJA source distribution can be found in these reports:

   `The RAJA Portability Layer: Overview and Status (2014) <file:../RAJAStatus-09.2014_LLNL-TR-661403.pdf>`_

   `RAJA Overview (extracted from ASC Tri-lab Co-design Level 2 Milestone Report 2015) <file:../RAJAOverview-Trilab-09.2015_LLNL-TR-677453.pdf>`_


**Contents:**

.. toctree::
   :maxdepth: 2

   config_build
   test_examples
   future

