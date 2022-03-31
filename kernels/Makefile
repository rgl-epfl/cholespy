NVCC=nvcc -m64 --ptx --expt-relaxed-constexpr

all: kernels.h

kernels.ptx: kernels.cu
	$(NVCC) --Wno-deprecated-gpu-targets -gencode arch=compute_30,code=compute_30 kernels.cu -o kernels.ptx

kernels.h: kernels.ptx
	bin2c kernels.ptx > kernels.h

clean:
	rm -f kernels.ptx kernels.h
