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



CUDAOBJS=psi_cuda_c_CopyCooToElg.o psi_cuda_c_CopyCooToHlg.o \
psi_cuda_d_CopyCooToElg.o psi_cuda_d_CopyCooToHlg.o \
psi_cuda_s_CopyCooToElg.o psi_cuda_s_CopyCooToHlg.o \
psi_cuda_z_CopyCooToElg.o psi_cuda_z_CopyCooToHlg.o 


lib: $(CUDAOBJS)
	ar cur ../$(LIBNAME) $(CUDAOBJS)

clean:
	/bin/rm -f $(CUDAOBJS)
