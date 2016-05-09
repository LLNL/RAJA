RAJA v1.0
============

[![Join the chat at https://gitter.im/llnl/raja](https://badges.gitter.im/llnl/raja.svg)](https://gitter.im/llnl/raja?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Static Analysis Status](https://scan.coverity.com/projects/8825/badge.svg?flat=1)](https://scan.coverity.com/projects/llnl-raja)

RAJA is a collection of C++ software abstractions, being developed at
Lawrence Livermore National Laboratory (LLNL), that enable architecture
portability for HPC applications. The overarching goals of RAJA are to:

  * Make existing (production) applications *portable with minimal disruption*
  * Provide a model for new applications so that they are portable from
    inception.

RAJA uses standard C++11 -- C++ is the predominant programming language in
which many LLNL codes are written. RAJA shares goals and concepts found in
other C++ portability abstraction approaches, such as
[Kokkos](https://github.com/kokkos/kokkos)
and [Thrust](https://developer.nvidia.com/thrust). RAJA is rooted in
a perspective based on substantial experience working on production
mesh-based multiphysics applications at LLNL. It provides constructs that
are absent in other models and which are fundamental in those codes. Also,
another goal of RAJA is enable application developers to adapt RAJA concepts
and specialize them for different code implementation patterns and C++ usage,
since data structures and algorithms vary widely across applications.

Documentation
-----------------

The [**Documentation**](http://software.llnl.gov/RAJA/) is the best place
to start learning about RAJA.

See also:

  * [The RAJA Portability Layer: Overview and Status (2014)](http://software.llnl.gov/RAJA/_static/RAJAStatus-09.2014_LLNL-TR-661403.pdf)
  * [RAJA Overview (extracted from ASC Tri-lab Co-design Level 2 Milestone Report 2015)](http://software.llnl.gov/RAJA/_static/RAJAOverview-Trilab-09.2015_LLNL-TR-677453.pdf)

Authors
-----------

The core developers of RAJA are:

  * Rich Hornung (hornung1@llnl.gov)
  * Jeff Keasler (keasler1@llnl.gov)

Other contributors include:

  * David Beckingsale (beckingsale1@llnl.gov)
  * Holger Jones (jones19@llnl.gov)
  * Adam Kunen (kunen1@llnl.gov)
  * Olga Pearce (pearce8@llnl.gov)
  * Tom Scogland (scogland1@llnl.gov)

Useful information about the RAJA source code, how to compile it,
run tests and examples, etc. can be seen by opening the file
raja/docs/sphinx/html/index.html in the RAJA repo in your web browser. The
HTML files are generated from the *.rst files in the directory raja/docs/sphinx.
You can also look at those text files if you find this more convenient.

If you have questions, find a bug, or have ideas about expanding the
functionality or applicability of RAJA and are interested in contributing
to its development, please do not hesitate to contact us. We are always
interested in improving RAJA and exploring new ways to use it.

Release
-----------

Copyright (c) 2016, Lawrence Livermore National Security, LLC.

Produced at the Lawrence Livermore National Laboratory.

All rights reserved.

Unlimited Open Source - BSD Distribution

For release details and restrictions, please read the README-license.txt file.
It is also linked here:
- [LICENSE](./README-license.txt)

`LLNL-CODE-689114`  `OCEC-16-063`
