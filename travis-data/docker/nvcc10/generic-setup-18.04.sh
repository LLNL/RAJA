#!/bin/sh
set -e
set -x


apt-get -qq update
apt-get -qq install -y --no-install-recommends wget
wget "http://keyserver.ubuntu.com/pks/lookup?op=get&search=0x60C317803A41BA51845E371A1E9377A2BA9EF27F" -O out && apt-key add out && rm out
echo deb http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main >> /etc/apt/sources.list
echo deb-src http://ppa.launchpad.net/ubuntu-toolchain-r/test/ubuntu bionic main >> /etc/apt/sources.list
apt-get -qq update
apt-get -qq install -y --no-install-recommends cmake python-dev build-essential sudo git vim dh-autoreconf ninja-build ca-certificates libtbb-dev

wget -q --no-check-certificate https://bootstrap.pypa.io/get-pip.py
python get-pip.py
rm get-pip.py
pip install -q -U pip
useradd -ms /bin/bash raja
printf "raja:raja" | chpasswd
adduser raja sudo
printf "raja ALL= NOPASSWD: ALL\\n" >> /etc/sudoers
