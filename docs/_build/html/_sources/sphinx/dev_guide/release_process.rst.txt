.. ##
.. ## Copyright (c) 2016-22, Lawrence Livermore National Security, LLC
.. ## and RAJA project contributors. See the RAJA/LICENSE file
.. ## for details.
.. ##
.. ## SPDX-License-Identifier: (BSD-3-Clause)
.. ##

.. _release-label:

*******************************************
RAJA Release Process
*******************************************

RAJA is considered part of the **RAJA Portability Suite** set of projects. 
Currently, the Suite includes `Umpire <https://github.com/LLNL/Umpire>`_, `CHAI <https://github.com/LLNL/CHAI>`_, and `camp <https://github.com/LLNL/camp>`_, in addition to RAJA. 

.. important:: Releases for the Suite are coordinated, meaning that when a 
               non-patch release is done for one, a new version release is 
               done for all Suite projects.

The RAJA release process includes the following sequence of steps:

  #. Identify all work (features in development, outstanding PRs, etc.) to be 
     to be included in the release.
  #. Merge all PRs containing work to be included in the release into the 
     develop branch.
  #. Make a :ref:`rcbranch-label` from the develop branch. Finalize the 
     release by completing remaining release tasks on that branch.
  #. When the release candidate branch is ready, make a PR for it to be merged
     into the **main branch.** When it is approved and all CI checks pass,
     merge the release candidate branch into the RAJA main branch.
  #. On GitHub, make a new release with a tag for the release. Following our
     convention, the tag label should have the format ``YYYY.mm.pp``. See 
     :ref:`version-label` for a description of the version numbering scheme we 
     use.  In the GitHub release description, please note key features, 
     bugfixes, etc. in the release. These should be a high-level summary of the 
     contents of the ``RELEASE_NOTES.md`` file in the RAJA repo, which may 
     contain more detailed information. Also, add a note to the 
     release description to remind users to download the gzipped tarfile for 
     the release instead of the assets GitHub creates for the release.
     The GitHub-created assets do not contain the RAJA submodules and may
     cause issues for users as a result.

     .. important:: For consistency, please follow a similar release 
                    description pattern for all RAJA releases.

  #. Check out the main branch locally and make sure it is up-to-date.     
     Then, generate the release tarfile by running the script 
     ``./scripts/make_release_tarball.sh`` from the top-level RAJA directory. 
     If this is successful, a gzipped tarfile whose name includes the release 
     tag **with no extraneous SHA-1 hash information** will be in the top-level
     RAJA directory.
  #. Edit the release in GitHub and upload the tarfile to the release.
  #. Make a PR to merge the main branch into the develop branch. After it 
     passes all CI checks and is approved, merge the PR. This will ensure that
     all changes done to finalize the release will be included in the develop 
     branch and future work on that branch.

After a RAJA release is done, there a other tasks that typically need to be 
performed to update content in other projects. These task are described in
:ref:`post_release-label`.

.. _rcbranch-label:

===========================
Release Candidate Branch
===========================

A *release candidate* branch is a temporary branch used to finalize a release.
When the features, documentation, bug fixes, etc.  to include in a release are 
complete and merged into the develop branch, a release candidate branch is made
off of the develop branch. Typically, a release candidate branch is named 
**rc-<release #>.** Please see :ref:`release-label` for a description of how 
a release candidate branch is used in the release process. 

.. important:: Creating a release candidate branch starts the next release 
               cycle whereby new work being performed on feature branches can 
               be merged into the develop branch.

Finalizing a release on a release candidate branch involves the following steps:

  #. If not already done, create a section for the release in the RAJA
     ``RELEASE_NOTES.md`` file. Describe all API changes, notable new features,
     bug fixes, improvements, build changes, etc. included in the release in 
     appropriately labeled sections of the file. Please follow the pattern
     established in the file for previous releases. All changes that users 
     should be aware of should be documented in the release notes. Hopefully,
     the release notes file has been updated along with the corresponding
     changes in PRs that are merged into the develop branch. At any rate, it is
     good practice to look over the commit history since the last release 
     to ensure all important changes are captured in the release notes.
  #. Update the version number entries for the new release in the 
     ``CMakeLists.txt`` file in the top-level RAJA directory. These include
     entries for: ``RAJA_VERSION_MAJOR``, ``RAJA_VERSION_MINOR``, and 
     ``RAJA_VERSION_PATCHLEVEL``. These items are used to define corresponding
     macro values in the ``include/RAJA/config.hpp`` file when the code is
     built so that users can access and check the RAJA version in their code.
  #. Update the ``version`` and ``release`` fields in the ``docs/conf.py`` 
     file to the new release number. This information is used in the online
     RAJA documentation.

.. important:: **No feature development is done on a release branch. Only bug 
               fixes, release documentation, and other release-oriented changes
               are made on a release candidate branch.**

.. _hotfixbranch-label:

===========================
Hotfix Branch
===========================

*Hotfix* branches are used in the (hopefully!) rare event that a bug is found
shortly after a release that may negatively impact RAJA users. A hotfix branch 
will address the issue be merged into both develop and main branches.

A hotfix branch is *made from main* with the name **hotfix/<issue>**. The 
issue is fixed (hopefully quickly!) and the release notes file is updated on 
the hotfix branch for the pending bugfix release. The branch is tested, against 
user code if necessary, to make sure the issue is resolved. Then, a PR is made 
to merge the hotfix branch into main. When it is approved and passes CI checks,
it is merged into the main branch. Lastly, a new release is made in a fashion 
similar to the process described in :ref:`release-label`. For completeness, 
the key steps for performing a hotfix release are:

  #. Make a **hotfix** branch from main for a release (hotfix/<issue>), fix the
     issue on the branch and verify, testing against user code if necessary.
     Update the release notes and RAJA patch version number as described
     in :ref:`rcbranch-label`.
  #. When the hotfix branch is ready, make a PR for it to be merged
     into the **main branch.** When that is approved and all CI checks pass,
     merge it into the RAJA main branch.
  #. On GitHub, make a new release with a tag for the release. Following our
     convention, the tag label should have the format ``YYYY.mm.pp``, where
     only the **patch** portion of the release tag should differ from the
     last release. In the GitHub release description, note that the release 
     is a bugfix release and describe the issue that is resolved. Also, add 
     a note to the release description to download the gzipped tarfile for the 
     release rather than the assets GitHub creates as part of the release.
  #. Check out the main branch locally and make sure it is up-to-date.     
     Then, generate the tarfile for the release by running the script 
     ``./scripts/make_release_tarball.sh`` from the top-level RAJA directory. 
     If this is successful, a gzipped tarfile whose name includes the release 
     tag **with no extraneous SHA-1 hash information** will be in the top-level
     RAJA directory.
  #. Make a PR to merge the main branch into the develop branch. After it 
     passes all CI checks and is approved, merge the PR. This will ensure that
     changes for the bugfix will be included in future development.

.. _post_release-label:

=========================
Post-release Activities
=========================

After a RAJA release is complete, other tasks are performed to update content 
in other repositories, typically. These tasks include:

  * Update the `RAJAProxies <https://github.com/LLNL/RAJAProxies>`_ project
    to the newly RAJA Portability Suite projects. This typically consists of 
    updating the submodules to the new RAJA Portability Suite project 
    versions, making sure the proxy-apps build and run correctly. When this
    is done, tag a release for proxy-app project.
  * Update the RAJA Spack package in the 
    `Spack repository <https://github.com/spack/spack>`_. This requires some
    knowledge of Spack and attention to details and Spack conventions. Please
    see :ref:`spack_package-label` for details.

.. _spack_package-label:

=========================
Spack Package Update
=========================

Describe how to update the RAJA Spack package....


