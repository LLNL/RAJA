
.PHONY: all clean

HIPCC:=/opt/rocm-4.5.0/hip/bin/hipcc

all: reproducer-hip-indirect-device-function-call-issue.exe

reproducer-hip-indirect-device-function-call-issue.exe: reproducer-hip-indirect-device-function-call-issue.cpp
	$(HIPCC) $< -o $@ -fgpu-rdc --offload-arch=gfx908 -O2

clean:
	rm -rf reproducer-hip-indirect-device-function-call-issue.exe
