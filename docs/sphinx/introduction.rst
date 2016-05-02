.. ##
.. ## Copyright (c) 2016, Lawrence Livermore National Security, LLC.
.. ##
.. ## Produced at the Lawrence Livermore National Laboratory.
.. ##
.. ## All rights reserved.
.. ##
.. ## For details and restrictions, please read the README-license.txt file.
.. ##


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
