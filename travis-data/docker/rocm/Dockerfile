FROM rocm/rocm-terminal:1.7.1

LABEL maintainer="David Beckingsale <david@llnl.gov"

USER root
ADD generic-setup.sh /root/generic-setup.sh
RUN ls -l /
RUN /root/generic-setup.sh

USER raja
WORKDIR /home/raja
