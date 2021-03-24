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
   echo "This script will make a branch in a local git repo for a PR from a "
   echo "branch in a forked repo. The script must be run inside the local repo."
   echo 
   echo "Syntax: make_local_branch_from_fork_pr [-h|num]"
   echo "options:"
   echo "-h    Print this help usage message."
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

echo "Making RAJA branch from branch on fork for PR $1"
echo 
echo "When script exits, you will be on local branch pr-from-fork/$1."
echo
echo "You can then push the new branch to the main repo; e.g.,"
echo "   git push <remote> <branch>"
echo
echo "If you make a PR for the new branch, it is a good idea to reference " 
echo "the original PR from the forked repo to track the original PR discussion."

git fetch origin pull/$1/head:pr-from-fork/$1 && git checkout pr-from-fork/$1
