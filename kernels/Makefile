NVCC=nvcc -m64 --ptx --expt-relaxed-constexpr

all: kernels.h

kernels.ptx: kernels.cu
	$(NVCC) --Wno-deprecated-gpu-targets -gencode arch=compute_50,code=compute_50 kernels.cu -o kernels.ptx

kernels.h: kernels.ptx
	bin2c kernels.ptx > kernels.h

clean:
	rm -f kernels.ptx kernels.h
