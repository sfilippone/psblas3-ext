TOPDIR=../..
include $(TOPDIR)/Make.inc
#
# Libraries used
#
PSBLIBDIR=$(PSBLASDIR)/lib/
PSBINCDIR=$(PSBLASDIR)/include
LIBDIR=$(TOPDIR)/lib
INCDIR=$(TOPDIR)/include
PSBLAS_LIB= -L$(PSBLIBDIR)  -lpsb_util  -lpsb_base
#-lpsb_util -lpsb_krylov -lpsb_prec -lpsb_base 
LDLIBS=$(PSBLDLIBS)
#
# Compilers and such
#
#CCOPT= -g
FINCLUDES=$(FMFLAG).   $(FMFLAG)$(INCDIR)  $(FMFLAG)$(PSBINCDIR) $(FIFLAG).
CINCLUDES=-I$(GPU_INCDIR) -I$(CUDA_INCDIR) -I..
LIBNAME=libpsb_gpu.a


.SUFFIXES: .cu

NVCC=nvcc
#CUDEFINES=-gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=sm_20\

#CUDEFINES=-gencode arch=compute_20,code=sm_20\
#	-gencode arch=compute_30,code=sm_30  -gencode arch=compute_35,code=sm_35
CUDEFINES=-gencode arch=compute_30,code=sm_30  -gencode arch=compute_35,code=sm_35 \
 -gencode arch=compute_50,code=sm_50 

CUDAOBJS=psi_cuda_c_CopyCooToElg.o psi_cuda_c_CopyCooToHlg.o \
psi_cuda_d_CopyCooToElg.o psi_cuda_d_CopyCooToHlg.o \
psi_cuda_s_CopyCooToElg.o psi_cuda_s_CopyCooToHlg.o \
psi_cuda_z_CopyCooToElg.o psi_cuda_z_CopyCooToHlg.o 


lib: $(CUDAOBJS)
	ar cur ../$(LIBNAME) $(CUDAOBJS)

.cu.o:
	$(NVCC)   $(CINCLUDES) $(CDEFINES) $(CUDEFINES) -c $<

clean:
	/bin/rm -f $(CUDAOBJS)
