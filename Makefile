ARCH = 61 # modify this. Ampere=86
NAME = convnext
OUT = OPTSTS64
#MODE = PROF
#LBR = OPENCNN

all:
	nvcc src/CN_MZ_mdfd.cu -I/opt/julia-1.6.0/include/julia/ -I /usr/local/cuda-11.7/include -I /usr/local/cuda-11.7/targets/x86_64-linux/include -L /usr/local/cuda-11.7/lib64 -lcublas -lcudnn -m64 -arch=compute_$(ARCH) -code=sm_$(ARCH) -o $(NAME) -D$(OUT)

#	nvcc src/lbr.cu  -I /usr/local/cuda-11.7/include -I /usr/local/cuda-11.7/targets/x86_64-linux/include -L /usr/local/cuda-11.7/lib64 -lcublas -lcudnn -m64 -arch=compute_$(ARCH) -code=sm_$(ARCH) -shared -o lbr.so -D$(OUT) 


clean:
	rm $(NAME)
