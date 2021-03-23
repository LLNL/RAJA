#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2016-21, Lawrence Livermore National Security, LLC
# and RAJA project contributors. See the RAJA/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

###############################################################################
# Help                                                                         
###############################################################################
Help()
{
   # Display Help
   echo "Syntax: make_raja_branch_from_fork_pr [-h|num]"
   echo "options:"
   echo "-h     Print this help usage message."
   echo "num   PR number."
   echo
}

###############################################################################
# Process the input options. 
###############################################################################
# Get the options
while getopts ":h" option; do
   case $option in
      h) # display Help
         Help
         exit;;
     \?) # incorrect option
         echo "Error: Invalid option"
         exit;;
   esac
done

echo "Making RAJA branch from branch on fork for PR $1";

echo "When script exits, you will be on branch pr-from-fork/$1 in your local RAJA repo."

echo "Do 'git push -u' to push the new branch to the main RAJA repo."

echo "Then, make a PR for that branch and reference the original PR from the forked repo."

git fetch origin pull/$1/head:pr-from-fork/$1 && git checkout pr-from-fork/$1
