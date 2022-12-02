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
Currently, the Suite includes `Umpire <https://github.com/LLNL/Umpire>`_, 
`CHAI <https://github.com/LLNL/CHAI>`_, and 
`camp <https://github.com/LLNL/camp>`_, in addition to RAJA. 

.. important:: The RAJA Portability Suite does coordinated releases, meaning
               that when a non-patch release is done for one, a new version 
               release is done for all Suite projects at the same time. When
               a coordinated release is done, common dependencies, such as
               BLT and camp, are set to the same versions in all Suite projects.

               Individual projects in the RAJA Portability Suite may do
               patch releases (to fix bugs, etc.) independently of other
               Suite projects.

The RAJA release process includes the following sequence of steps:

  #. **Identify all work to be to be included in the release**, such as 
     features to development and issues to be resolved.
  #. **Merge each PR** containing work to be included in the release into the 
     develop branch when it is ready.
  #. **Make a release candidate branch** from the develop branch. 

     .. note:: Creation of the release candidate branch begins the next 
               release cycle. While the release candidate branch is being
               finalized, work can continue on the develop branch.
 
  #. **Finalize the release on the release candidate branch**  by completing 
     remaining release tasks on it. See :ref:`rcbranch-label` for typical 
     tasks to complete.
  #. **Make a PR to merge the release candidate branch into the main branch** 
     when the release candidate branch is ready for the release.

     .. note:: Since the main branch only changes when a release is made, the
               release candidate PR will likely contain many modifications.
               Fortunately, the vast majority of those changes will have been
               reviewed, approved, and merged into the develop branch. 
   
               To make it easier for team members to review the release 
               candidate PR, it usually helps to make a companion draft PR 
               to merge the release candidate branch into develop. This PR
               will only show the changes to merge into main for the release.
               Cross-reference the two PRs in their descriptions, tell 
               reviewers to review the draft PR into develop, but approve 
               the real PR to merge into main.

  #. **Merge the release candidate branch into the RAJA main branch** when it 
     is approved and all CI checks pass.
  #. **Create the release on GitHub**.

     #. Choose the RAJA main branch and the option to create the release tag 
        when the release is published. The release tag is the the name of the
        release.

        .. important:: Set the release name (and associated git tag name) 
                       following the convention established for prior releases.
                       Specifically, the tag label should have the format 
                       ``vYYYY.mm.pp``. See :ref:`version-label` for details 
                       about the version labeling scheme we use. 

     #. Fill in the release description. Note key features, bugfixes, etc.
        included in the release. The description should summarize the relevant 
        section of the ``RELEASE_NOTES.md`` file in the release candidate 
        branch that was merged. Also, add a note to the release description to 
        remind users to download the gzipped tarfile named for the release 
        (see below) instead of the assets GitHub creates for a release. The 
        assets created by GitHub do not contain the RAJA submodules and may 
        cause issues for users as a result.

        .. important:: For consistency, please follow a similar release 
                       description pattern for all RAJA releases.

     #. Publish the release when it is ready by clicking the button.

     #. Generate a release tarfile. Check out the main branch locally and 
        make sure it is up-to-date. Then, run the script::
 
          ./scripts/make_release_tarball.sh 

        from the top-level RAJA directory. The script strips out the Git files
        from the code and generates a tarfile whose name contains the release
        tag name. If this is successful, a gzipped tarfile whose name **does 
        not contain extraneous SHA-1 hash information** will be in the 
        top-level RAJA directory of your local repository.

     #. Edit the release in GitHub and upload the tarfile to the release.

  #. Lastly, **make a PR to merge the main branch into the develop branch**. 
     After it passes all CI checks and is approved, merge the PR. This will 
     ensure that all changes done to finalize the release are included 
     in the develop branch and future work on that branch.

After a RAJA release is done, there a other tasks that typically need to be 
performed to update content in other projects. These task are described in
:ref:`post_release-label`.

.. _rcbranch-label:

===========================
Release Candidate Branch
===========================

A *release candidate* branch is a temporary branch used to finalize a release.
When the features, documentation, bug fixes, etc. to include in a release are 
complete and merged into the develop branch, a release candidate branch is made
from the develop branch. Typically, a release candidate branch is named 
**rc-<release #>**, or similar. Please see :ref:`release-label` for a 
description of how a release candidate branch is used in the release process. 

Finalizing a release on a release candidate branch involves the following steps:

  #. **Complete the release notes describing the release** in the 
     ``RELEASE_NOTES.md`` file. Describe all API changes, notable new features,
     bug fixes, improvements, build changes, etc. included in the release in 
     appropriately labeled sections of the file. 

     .. important:: Please follow the pattern established in the release notes
                    file used for previous releases. 

     All changes that users need to be aware of should be documented in the
     release notes. Hopefully, the release notes file has been updated along
     with the corresponding changes in PRs that are merged into the develop
     branch. Regardless, it is good practice to look over the commit history
     since the last release to ensure all important changes are documented
     in the release notes.
  #. **Update the version number entries in the code**. The top-level 
     ``CMakeLists.txt`` file must be changed, where the entries: 
     ``RAJA_VERSION_MAJOR``, ``RAJA_VERSION_MINOR``, and 
     ``RAJA_VERSION_PATCHLEVEL`` are defined. These items are used to define 
     corresponding macro values in the ``include/RAJA/config.hpp`` file when 
     the code is configured so that users can access and check the RAJA 
     version in their code by including that header file. The ``version`` and 
     ``release`` fields in the ``RAJA/docs/conf.py`` file must also be changed
     to the new release number. This information is used in the online
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
will address the issue in both the develop and main branches.

A hotfix branch treated like a release candidate branch and it is used to 
generate a *patch release* following the same basic process that is described 
in :ref:`_release-label`.

For completeness, the key steps for performing a hotfix (patch) release are:

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
  #. Make a PR to merge the main branch back into the develop branch. After it 
     passes all CI checks and is approved, merge the PR. This will ensure that
     changes for the bugfix will be included in future development.

.. _post_release-label:

=========================
Post-release Activities
=========================

After a RAJA release is complete, other tasks are performed to update content 
in other repositories, typically. These tasks include:

  * Update the `RAJAProxies <https://github.com/LLNL/RAJAProxies>`_ project
    to the new RAJA Portability Suite project release. This typically consists 
    of updating the submodules to the new RAJA Portability Suite project 
    versions, making sure the proxy-apps build and run correctly. When this
    is done, tag a release for proxy-app project.
  * Update the 
    `RAJA Template Project <https://github.com/LLNL/RAJA-project-template>`_ 
    project to the new RAJA release.
  * Update the RAJA Spack package in the 
    `Spack repository <https://github.com/spack/spack>`_. This requires some
    knowledge of Spack and attention to details and Spack conventions. Please
    see :ref:`spack_package-label` for details.

.. _spack_package-label:

=========================
Spack Package Update
=========================

Describe how to update the RAJA Spack package....


