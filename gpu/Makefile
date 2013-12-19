include ../Make.inc
#
# Libraries used
#
PSBLIBDIR=$(PSBLASDIR)/lib/
PSBINCDIR=$(PSBLASDIR)/include
LIBDIR=../lib
INCDIR=../include
PSBLAS_LIB= -L$(PSBLIBDIR)  -lpsb_util  -lpsb_base
#-lpsb_util -lpsb_krylov -lpsb_prec -lpsb_base 
LDLIBS=$(PSBLDLIBS)
#
# Compilers and such
#
#CCOPT= -g
FINCLUDES=$(FMFLAG).   $(FMFLAG)$(INCDIR)  $(FMFLAG)$(PSBINCDIR) $(FIFLAG).
CINCLUDES=-I$(GPU_INCDIR) -I$(CUDA_INCDIR)
LIBNAME=libpsb_gpu.a


FOBJS=cusparse_mod.o base_cusparse_mod.o \
 s_cusparse_mod.o  d_cusparse_mod.o  c_cusparse_mod.o  z_cusparse_mod.o \
	psb_vectordev_mod.o core_mod.o \
	psb_s_vectordev_mod.o psb_d_vectordev_mod.o \
	psb_c_vectordev_mod.o psb_z_vectordev_mod.o psb_base_vectordev_mod.o \
        elldev_mod.o hlldev_mod.o diagdev_mod.o hdiagdev_mod.o \
	psb_i_gpu_vect_mod.o \
	psb_d_gpu_vect_mod.o  psb_s_gpu_vect_mod.o\
	psb_z_gpu_vect_mod.o  psb_c_gpu_vect_mod.o\
        psb_d_elg_mat_mod.o   psb_d_hlg_mat_mod.o \
	psb_d_hybg_mat_mod.o  psb_d_csrg_mat_mod.o\
        psb_s_elg_mat_mod.o   psb_s_hlg_mat_mod.o \
	psb_s_hybg_mat_mod.o  psb_s_csrg_mat_mod.o\
        psb_c_elg_mat_mod.o   psb_c_hlg_mat_mod.o \
	psb_c_hybg_mat_mod.o  psb_c_csrg_mat_mod.o\
        psb_z_elg_mat_mod.o   psb_z_hlg_mat_mod.o \
	psb_z_hybg_mat_mod.o  psb_z_csrg_mat_mod.o\
	psb_gpu_penv_mod.o psb_gpu_mod.o \
	psb_d_diag_mat_mod.o\
	psb_d_hdiag_mat_mod.o

COBJS= elldev.o hlldev.o diagdev.o hdiagdev.o vectordev.o cuda_util.o fcusparse.o 

OBJS=$(COBJS) $(FOBJS)

lib: cmp ilib copy

cmp: $(OBJS) 
	ar cur $(LIBNAME) $(OBJS)

copy: cmp ilib
	/bin/cp -p $(LIBNAME) *$(.mod) $(LIBDIR)

hlldev_mod.o elldev_mod.o psb_base_vectordev_mod.o: core_mod.o
psb_d_gpu_vect_mod.o psb_s_gpu_vect_mod.o psb_z_gpu_vect_mod.o psb_c_gpu_vect_mod.o: psb_i_gpu_vect_mod.o
psb_i_gpu_vect_mod.o : psb_vectordev_mod.o
cusparse_mod.o:  s_cusparse_mod.o  d_cusparse_mod.o  c_cusparse_mod.o  z_cusparse_mod.o 
s_cusparse_mod.o  d_cusparse_mod.o  c_cusparse_mod.o  z_cusparse_mod.o : base_cusparse_mod.o  
psb_d_hlg_mat_mod.o: hlldev_mod.o psb_d_gpu_vect_mod.o 
psb_d_elg_mat_mod.o: elldev_mod.o psb_d_gpu_vect_mod.o
psb_d_diag_mat_mod.o: diagdev_mod.o psb_d_gpu_vect_mod.o
psb_d_hdiag_mat_mod.o: hdiagdev_mod.o psb_d_gpu_vect_mod.o 
psb_s_hlg_mat_mod.o: hlldev_mod.o psb_s_gpu_vect_mod.o 
psb_s_elg_mat_mod.o: elldev_mod.o psb_s_gpu_vect_mod.o 
psb_s_csrg_mat_mod.o psb_s_hybg_mat_mod.o: cusparse_mod.o psb_vectordev_mod.o
psb_d_csrg_mat_mod.o psb_d_hybg_mat_mod.o: cusparse_mod.o psb_vectordev_mod.o
psb_z_hlg_mat_mod.o: hlldev_mod.o psb_z_gpu_vect_mod.o 
psb_z_elg_mat_mod.o: elldev_mod.o psb_z_gpu_vect_mod.o 
psb_c_hlg_mat_mod.o: hlldev_mod.o psb_c_gpu_vect_mod.o 
psb_c_elg_mat_mod.o: elldev_mod.o psb_c_gpu_vect_mod.o 
psb_c_csrg_mat_mod.o psb_c_hybg_mat_mod.o: cusparse_mod.o psb_vectordev_mod.o
psb_z_csrg_mat_mod.o psb_z_hybg_mat_mod.o: cusparse_mod.o psb_vectordev_mod.o
psb_vectordev_mod.o: psb_s_vectordev_mod.o psb_d_vectordev_mod.o psb_c_vectordev_mod.o psb_z_vectordev_mod.o
psb_s_vectordev_mod.o psb_d_vectordev_mod.o psb_c_vectordev_mod.o psb_z_vectordev_mod.o: psb_base_vectordev_mod.o
vectordev.o: cuda_util.o  
elldev.o: elldev.c
fcusparse.h elldev.c: elldev.h vectordev.h
fcusparse.o: fcusparse.h

psb_gpu_mod.o:  psb_d_gpu_vect_mod.o  psb_s_gpu_vect_mod.o\
	psb_z_gpu_vect_mod.o  psb_c_gpu_vect_mod.o\
        psb_d_elg_mat_mod.o   psb_d_hlg_mat_mod.o \
	psb_d_hybg_mat_mod.o  psb_d_csrg_mat_mod.o\
        psb_s_elg_mat_mod.o   psb_s_hlg_mat_mod.o \
	psb_s_hybg_mat_mod.o  psb_s_csrg_mat_mod.o\
        psb_c_elg_mat_mod.o   psb_c_hlg_mat_mod.o \
	psb_c_hybg_mat_mod.o  psb_c_csrg_mat_mod.o\
        psb_z_elg_mat_mod.o   psb_z_hlg_mat_mod.o \
	psb_z_hybg_mat_mod.o  psb_z_csrg_mat_mod.o\
	psb_d_diag_mat_mod.o \
	psb_d_hdiag_mat_mod.o

ilib: cmp 
	cd impl &&  $(MAKE) LIBNAME=$(LIBNAME)
clean: cclean iclean
	/bin/rm -f  $(FOBJS) *$(.mod) *.a

cclean: 
	/bin/rm -f  $(COBJS) 
iclean:
	cd impl && $(MAKE) clean

verycleanlib: 
	(cd ../..; make veryclean)