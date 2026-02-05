ARG BASE_IMAGE=openvla-oft:dgx
FROM ${BASE_IMAGE}

WORKDIR /workspace/openvla-oft

RUN pip install trossen-arm
RUN apt-get update && apt-get install -y \
        libusb-dev \
        usbutils

# Teensy Loader CLIのインストール(https://docs.trossenrobotics.com/trossen_arm/main/getting_started/software_setup.html)
RUN cd /workspace \
    && git clone https://github.com/PaulStoffregen/teensy_loader_cli.git \
    && cd teensy_loader_cli \
    && make \
    && cp teensy_loader_cli /usr/local/bin
RUN mkdir -p /etc/udev/rules.d \
    && wget -O /etc/udev/rules.d/00-teensy.rules https://www.pjrc.com/teensy/00-teensy.rules

# Default to bash for interactive use.
CMD ["bash"]
