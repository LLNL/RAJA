#!/bin/sh
set -e
set -x


apt-get -qq update
apt-get -qq install -y --no-install-recommends wget cmake python-dev python-pip build-essential sudo git vim dh-autoreconf ninja-build ca-certificates libtbb-dev

useradd -ms /bin/bash raja
printf "raja:raja" | chpasswd
adduser raja sudo
printf "raja ALL= NOPASSWD: ALL\\n" >> /etc/sudoers
