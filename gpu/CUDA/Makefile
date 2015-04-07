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


.SUFFIXES: .cu

NVCC=nvcc
CUDEFINES=-gencode arch=compute_13,code=sm_13 -gencode arch=compute_20,code=sm_20\
	-gencode arch=compute_30,code=sm_30  -gencode arch=compute_35,code=sm_35

CUDAOBJS=w_cuda_CopyCooToElg.o w_cuda_CopyCooToHlg.o

all: $(CUDAOBJS)
	/bin/mv  $(CUDAOBJS)  ..

.cu.o:
	$(NVCC) -g -G  $(CINCLUDES) $(CDEFINES) $(CUDEFINES) -c $<

