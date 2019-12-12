FROM rocm/dev-ubuntu-16.04

LABEL maintainer="David Beckingsale <david@llnl.gov>"

ADD generic-setup.sh /root/generic-setup.sh

RUN sudo /root/generic-setup.sh

USER raja
WORKDIR /home/raja
