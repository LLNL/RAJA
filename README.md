RAJA v0.1.0
============

[![Build Status](https://travis-ci.org/LLNL/RAJA.svg?branch=develop)](https://travis-ci.org/LLNL/RAJA)
[![Join the chat at https://gitter.im/llnl/raja](https://badges.gitter.im/llnl/raja.svg)](https://gitter.im/llnl/raja?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

RAJA is a collection of C++ software abstractions, being developed at
Lawrence Livermore National Laboratory (LLNL), that enable architecture
portability for HPC applications. The overarching goals of RAJA are to:

  * Make existing (production) applications *portable with minimal disruption*
  * Provide a model for new applications so that they are portable from
    inception.

RAJA uses standard C++11 -- C++ is the predominant programming language in
which many LLNL codes are written. RAJA is rooted in a perspective based on 
substantial experience working on production mesh-based multiphysics 
applications at LLNL. Another goal of RAJA is to enable application developers
to adapt RAJA concepts and specialize them for different code implementation 
patterns and C++ usage, since data structures and algorithms vary widely 
across applications.

RAJA shares goals and concepts found in
other C++ portability abstraction approaches, such as
[Kokkos](https://github.com/kokkos/kokkos)
and [Thrust](https://developer.nvidia.com/thrust). 
However, it includes concepts that are absent in other models and which are 
fundamental to LLNL codes. 

It is important to note that RAJA is very much a work-in-progress.
The community of researchers and application developers at LLNL that are
actively contributing to it and developing new capabilities is growing.
The publicly-released version contains only core pieces of RAJA as they
exist today. While the basic interfaces are fairly stable, the implementation
of the underlying concepts is being refined. Additional features will appear
in future releases.

Quick Start
-----------

The RAJA code lives in a GitHub [repository](https://github.com/llnl/raja).
To clone the repo, use the command:

    git clone https://github.com/llnl/raja.git

Then, you can build RAJA like any other CMake project, provided you have a C++
compiler that supports the C++11 standard. The simplest way to build the code 
is to do the following in the top-level RAJA directory (in-source builds 
are not allowed!):

    mkdir build
    cd build
    cmake ../
    make

More details about RAJA configuration options are located in the User 
Documentation.

User Documentation
-------------------

The [**Documentation**](http://software.llnl.gov/RAJA/) is the best place
to start learning about RAJA.

Other references that may be of interest include:

  * [The RAJA Portability Layer: Overview and Status (2014)](http://software.llnl.gov/RAJA/_static/RAJAStatus-09.2014_LLNL-TR-661403.pdf)
  * [RAJA Overview (extracted from ASC Tri-lab Co-design Level 2 Milestone Report 2015)](http://software.llnl.gov/RAJA/_static/RAJAOverview-Trilab-09.2015_LLNL-TR-677453.pdf)

To cite RAJA, please use the following reference:

* R. D. Hornung and J. A. Keasler, [The RAJA Poratability Layer: Overview and Status](http://software.llnl.gov/RAJA/_static/RAJAStatus-09.2014_LLNL-TR-661403.pdf), Tech Report, LLNL-TR-661403, Sep. 2014.

Mailing List
-----------------

Interested in keep up with RAJA or communicating with its developers and
users? Please join our mailing list over at Google Groups:
- [RAJA Google Group](https://groups.google.com/forum/#!forum/raja-users)

If you have questions, find a bug, or have ideas about expanding the
functionality or applicability of RAJA and are interested in contributing
to its development, please do not hesitate to contact us. We are very
interested in improving RAJA and exploring new ways to use it.

Contributions
---------------

The RAJA team follows the [GitFlow](http://nvie.com/posts/a-successful-git-branching-model/) development model. Folks wishing to contribute to RAJA, should
include their work in a feature branch created from the RAJA `develop` branch.
Then, create a pull request with the `develop` branch as the destination. That
branch contains the latest work in RAJA. Periodically, we will merge the 
develop branch into the `master` branch and tag a new release.

Authors
-----------

The original developers of RAJA are:

  * Rich Hornung (hornung1@llnl.gov)
  * Jeff Keasler (keasler1@llnl.gov)

Contributors include:

  * David Beckingsale (beckingsale1@llnl.gov)
  * Jason Burmark (burmark1@llnl.gov)
  * Holger Jones (jones19@llnl.gov)
  * Will Killian (killian4@llnl.gov)
  * Adam Kunen (kunen1@llnl.gov)
  * David Poliakoff (poliakoff1@llnl.gov)
  * Tom Scogland (scogland1@llnl.gov)


Release
-----------

Copyright (c) 2016, Lawrence Livermore National Security, LLC.

Produced at the Lawrence Livermore National Laboratory.

All rights reserved.

Unlimited Open Source - BSD Distribution

For release details and restrictions, please read the LICENSE.txt file.
It is also linked here:
- [LICENSE](./LICENSE.txt)

`LLNL-CODE-689114`  `OCEC-16-063`
