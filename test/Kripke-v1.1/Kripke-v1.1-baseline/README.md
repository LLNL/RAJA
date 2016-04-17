KRIPKE
======

Version 1.1

Release Date 9/13/2015 


Authors
=======
  * Adam J. Kunen [kunen1@llnl.gov](mailto:kunen1@llnl.gov) (Primary point of contact)
  * Peter N. Brown [brown42@llnl.gov](mailto:brown42@llnl.gov)
  * Teresa S. Bailey [bailey42@llnl.gov](mailto:bailey42@llnl.gov)
  * Peter G. Maginot [maginot1@llnl.gov](mailto:maginot1@llnl.gov)


License
=======
See included file NOTICE.md


Overview
========
Kripke is a simple, scalable, 3D Sn deterministic particle transport code.  Its primary purpose is to research how data layout, programming paradigms and architectures effect the implementation and performance of Sn transport.  A main goal of Kripke is investigating how different data-layouts affect instruction, thread and task level parallelism, and what the implications are on overall solver performance.

Kripkie supports storage of angular fluxes (Psi) using all six striding orders (or "nestings") of Directions (D), Groups (G), and Zones (Z), and provides computational kernels specifically written for each of these nestings. Most Sn transport codes are designed around one of these nestings, which is an inflexibility that leads to software engineering compromises when porting to new architectures and programming paradigms.

Early research has found that the problem dimensions (zones, groups, directions, scattering order) and the scaling (number of threads and MPI tasks), can make a profound difference in the performance of each of these nestings. To our knowledge this is a capability unique to Kripke, and should provide key insight into how data-layout effects Sn solver performance. An asynchronous MPI-based parallel sweep algorithm is provided, which employs the concepts of Group Sets (GS) Zone Sets (ZS), and Direction Sets (DS), borrowed from the [Texas A&M code PDT](https://parasol.tamu.edu/asci/).

As we explore new architectures and programming paradigms with Kripke, we will be able to incorporate these findings and ideas into our larger codes. The main advantages of using Kripke for this exploration is that it's light-weight (ie. easily refactored and modified), and it gets us closer to the real question we want answered: "What is the best way to layout and implement an Sn code on a given architecture+programming-model?" instead of the more commonly asked question "What is the best way to map my existing Sn code to a given architecture+programming-model?".


Mini App or Proxy App?
----------------------
Kripke is a Mini-App since it has a very small code base consisting of 4184 lines of C++ code (generated using David A. Wheeler's SLOCCount v2.26).

Kripke is also a Proxy-App since it is a proxy for the LLNL transport code ARDRA.


Analysis
--------
A major challenge of achieving high-performance in an Sn transport (or any physics) code is choosing a data-layout and a parallel decomposition that lends itself to the targeted architecture. Often the data-layout determines the most efficient nesting of loops in computational kernels, which then determines how well your inner-most-loop SIMDizes, how you add threading (pthreads, OpenMP, etc.), and the efficiency and design of your parallel algorithms. Therefore, each nesting produces different loop nesting orders, which provides substantially different performance characteristics. We want to explore how easily and efficiently these different nestings map to different architectures. In particular, we are interested in how we can achieve good parallel efficiency while also achieving efficient use of node resources (such as SIMD units, memory systems, and accelerators).

Parallel sweep algorithms can be explored with Kripke in multiple ways. The core MPI algorithm could be modified or rewritten to explore other approaches, domain overloading, or alternate programming models (such as Charm++). The effect of load-imbalance is an understudied aspect of Sn transport sweeps, and could easily be studied with Kripke by artificially adding more work (ie unknowns) to a subset of MPI tasks. Block-AMR could be added to Kripke, which would be a useful way to explore the cost-benefit analysis of adding AMR to an Sn code, and would be a way to further study load imbalances and AMR effects on sweeps.

The coupling of on-node sweep kernel, the parallel sweep algorithm, and the choices of decomposing the problem phase space into GS's, ZS's and DS's impact the performance of the overall sweep. The tradeoff between large and small "units of work" can be studied. Larger "units of work" provide more opportunity for on-node parallelism, while creating larger messages, less "sends", and less efficient parallel sweeps. Smaller "units of work" make for less efficient on-node kernels, but more efficient parallel sweeps. 

We can also study trading MPI tasks for threads, and the effects this has on our programming models and cache efficiency.

A simple timer infrastructure is provided that measure each compute kernels total time.


Physical Models
---------------

Kripke solves the Discrete Ordinance and Diamond Difference discretized steady-state linear Boltzmann equation. 

        H * Psi = (LPlus * S * L) * Psi + Q

Where:

*   **Psi** is the unknown angular flux discretized over zones, directions, and energy groups

*   **H** is the "streaming-collision" operator.  (Couples zones)

*   **L** is the "discrete-to-moments operator. (Couples directions and moments)

*   **LPlus** is the "moment-to-discrete" operator. (Couples directions and moments)

*   **S** is the (arbitrary) order scattering operator. (Couples groups)

*   **Q** is an external source. In Kripke it is represented in moment space, so really "LPlus*Q"


Kripke is hard-coded to setup and solve the [3D Kobayashi radiation benchmark, problem 3i](https://www.oecd-nea.org/science/docs/2000/nsc-doc2000-4.pdf).  Since Kripke does not have reflecting boundary conditions, the full-space model is solved. Command line arguments allow the user to modify the total and scattering cross-sections.  Since Kripke is a multi-group transport code and the Kobayashi problem is single-group, each energy group is setup to solve the same problem with no group-to-group coupling in the data.


The steady-state solution method uses the source-iteration technique, where each iteration is as follows:

1.  Phi = LTimes(Psi)
2.  PhiOut = Scattering(Phi)
3.  PhiOut = PhiOut + Source()
4.  Rhs = LPlusTimes(PhiOut)
5.  Psi = Sweep(Rhs, Psi)  which is solving Psi=(Hinverse * Rhs) a.k.a _"Inverting H"_



Building and Running
====================

Kripke comes with a simple CMake based build system.

Requirements
------------
*  CMake 3.0 or later
*  C++ Compiler (g++, icpc, etc.)
*  MPI 1.0 or later



Quick Start
-----------
The easiest way to get Kripke running, is to directly invoke CMake and take whatever system defaults you have for compilers and let CMake find MPI for you.

*  Step 1:  Create a build space (assuming you are starting in the Kripke root directory)   
        
        mkdir build

*  Step 2: Run CMake in that build space
        
        cd kripke
        cmake ..

*  Step 3: Now make Kripke:
         
        make -j8
  
*  Step 4: Run the test suite to make sure it works
   
        make test
  
*  Step 5: Run Kripke's default problem:
   
        ./kripke
  

Running Kripke
==============

Environment Variabes
--------------------

If Kripke is build with OpenMP support, then the environment variables ``OMP_NUM_THREADS`` is used to control the number of OpenMP threads.  Kripke does not attempt to modify the OpenMP runtime in anyway, so other ``OMP_*`` environment variables should also work as well.
 

Command Line Options
--------------------
Command line option help can also be viewed by running "./kripke --help"

### Problem Size Options:

*   **``--groups <ngroups>``**     

    Number of energy groups. (Default: --groups 32)

*   **``--legendre <lorder>``**    

    Scattering Legendre Expansion Order (0, 1, ...).  (Default: --legendre 4)

*   **``--quad <ndirs>``**, or **``--quad <polar>:<azim>``**

    Define the quadrature set to use either a fake S2 with <ndirs> points, OR Gauss-Legendre with <polar> by <azim> points.   (Default: --quad 96)

*   **``--zones <x>,<y>,<z>``**

    Number of zones in x,y,z.  (Default: --zones 16,16,16)


### Physics Parameters:

*   **``--sigt <sigt0,sigt1,sigt2>``**
 
    Total material cross-sections.  (Default:   --sigt 0.1,0.0001,0.1)

*   **``--sigs <sigs0,sigs1,sigs2>``**
 
    Total material cross-sections.  (Default:   --sigs 0.05,0.00005,0.05)


### On-Node Options:

*   **``--nest <NEST>``**

    Loop nesting order (and data layout), available are DGZ, DZG, GDZ, GZD, ZDG, and ZGD. (Default: --nest DGZ)


###Parallel Decomposition Options:

*   **``--layout <lout>``**        
    
    Layout of spatial subdomains over mpi ranks. 0 for "Blocked" where local zone sets represent adjacent regions of space. 1 for "Scattered" where adjacent regions of space are distributed to adjacent MPI ranks. (Default: --layout 0)

*   **--procs <npx,npy,npz>**  
    
    Number of MPI ranks in each spatial dimension. (Default:  --procs 1,1,1)

*   **``--dset <ds>``**

    Number of direction-sets.  Must be a factor of 8, and divide evenly the number of quadrature points. (Default:  --dset 8)

*   **``--gset <gs>``**            
    
    Number of energy group-sets.  Must divide evenly the number energy groups. (Default:  --gset 1)

*   **``--zset <zx>,<zy>,<zz>``**  
    
    Number of zone-sets in x, y, and z.  (Default:  --zset 1:1:1)


###Solver Options:

*   **``--niter <NITER>``**

    Number of solver iterations to run. (Default:  --niter 10)

*   **``--pmethod <method>``**     

    Parallel solver method. "sweep" for full up-wind sweep (wavefront algorithm). "bj" for Block Jacobi.  (Default: --pmethod sweep)


### Output and Testing Options:

*   **``--test``**                 

    Run Kernel Test instead of solve

*   **``--silo <siloname>``**                 

    Write SILO output (requires building with LLNL's Silo library)

*   **``--papi <PAPI_XXX_XXX,...>``**

    Track PAPI hardware counters for each timer. (requires building with PAPI library)
    

Test Suite
----------

Running with the ``--test`` command line argument will run a unit-testing frame work that will compare each kernel, using random input data, with the same kernel from a different nesting.  This is very useful for checking correctness of kernels after modification.

Running ``make test`` will use the CMake testing framework, CTest, to run a series of tests outlined in the root ``CMakeLists.txt`` file.


Future Plans
============

Some ideas for future study:

*   Block AMR.

*   More FLOP intensive spatial discretizations such as DFEM's.

*   Programming model abstractions


Retirement
==========

Retirement of this Mini-App should be considered when it is no longer a representative of state-of-the-art transport codes, or when it becomes too cumbersome to adapt to advanced architectures. Also, at the point of retirement it should be clear how to design its successor.


Publications, Presentations, Links
==================================

*  [LLNL Codesign Website](https://codesign.llnl.gov/index.php)

*  A. J. Kunen, Kripke – [An Sn Transport Mini App](https://codesign.llnl.gov/pdfs/Kripke_NECDC2014_Present.pdf), NECDC, October 22, 2014 (LLNL-PRES-661866)

*  A. J. Kunen, [RAJA-Like Transformations in Kripke](https://codesign.llnl.gov/pdfs/Kunen_JOWOG34.pdf), JOWOG34, February 5, 2015 (LLNL-PRES-666686)


Release
=======
LLNL-CODE-658597
