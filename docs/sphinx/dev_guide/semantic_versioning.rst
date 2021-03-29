.. ##
.. ## Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/COPYRIGHT file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _semver-label:

***********************
Semantic Versioning
***********************

The Axom team uses the *semantic* versioning scheme for assigning
release numbers. Semantic versioning is a methodology for assigning version
numbers to software releases in a way that conveys specific meaning about
the code and modifications from version to version.
See `Semantic Versioning <http://semver.org>`_ for a more detailed description.

============================
Version Numbers and Meaning
============================

Semantic versioning is based on a three part version number `MM.mm.pp`:

  * `MM` is the *major* version number. It is incremented when an incompatible
    API change is made. That is, the API changes in a way that may break code
    using an earlier release of the software with a smaller major version
    number. Following Gitflow (above), the major version number may be changed
    when the develop branch is merged into the main branch.
  * `mm` is the *minor* version number. It changes when functionality is
    added that is backward-compatible. The API may grow to support new
    functionality. However, the software will function the same as any
    earlier release of the software with a smaller minor version number
    when used through the intersection of two APIs. Following Gitflow (above),
    the minor version number is always changed when the develop branch is
    merged into the main branch, except possibly when the major version
    is changed.
  * `pp` is the *patch* version number. It changes when a bug fix is made that
    is backward compatible. That is, such a bug fix is an internal
    implementation change that fixes incorrect behavior. Following Gitflow
    (above), the patch version number is always changed when a hotfix branch
    is merged into main, or when develop is merged into main and the
    changes only contain bug fixes.

===========================================
What Does a Change in Version Number Mean?
===========================================

A key consideration in meaning for these three version numbers is that
the software has a public API. Changes to the API or code functionality
are communicated by the way the version number is incremented. Some important
conventions followed when using semantic versioning are:

  * Once a version of the software is released, the contents of the release
    *must not* change. If the software is modified, it *must* be released
    as a new version.
  * A major version number of zero (i.e., `0.mm.pp`) is considered initial
    development where anything may change. The API is not considered stable.
  * Version `1.0.0` defines the first stable public API. Version number
    increments beyond this point depend on how the public API changes.
  * When the software is changed so that any API functionality becomes
    deprecated, the minor version number *must* be incremented.
  * A pre-release version may be denoted by appending a hyphen and a series
    of dot-separated identifiers after the patch version. For example,
    `1.0.1-alpha`, `1.0.1-alpha.1`, `1.0.2-0.2.5`.
  * Versions are compared using precedence that is calculated by separating
    major, minor, patch, and pre-release identifiers in that order. Major,
    minor, and patch numbers are compared numerically from left to right. For
    example, 1.0.0 < 2.0.0 < 2.1.0 < 2.1.1. When major, minor, and patch
    numbers are equal, a pre-release version has lower precedence. For
    example, 1.0.0-alpha < 1.0.0.

By following these conventions, it is fairly easy to communicate intent of
version changes to users and it should be straightforward for users
to manage dependencies on Axom. 
