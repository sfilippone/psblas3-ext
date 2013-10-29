include ../Make.inc
#
# Libraries used
#
PSBLIBDIR=$(PSBLASDIR)/lib/
PSBINCDIR=$(PSBLASDIR)/include
LIBDIR=../lib
PSBLAS_LIB= -L$(PSBLIBDIR)  -lpsb_util  -lpsb_base
#-lpsb_util -lpsb_krylov -lpsb_prec -lpsb_base 
LDLIBS=$(PSBLDLIBS)
#
# Compilers and such
#
#CCOPT= -g
FINCLUDES=$(FMFLAG).   $(FMFLAG)$(PSBINCDIR) $(FIFLAG).
CINCLUDES=-I$(GPU_INCDIR) -I$(CUDA_INCDIR)
LIBNAME=libpsb_ext.a


FOBJS=  psb_d_ell_mat_mod.o psb_d_hll_mat_mod.o \
	psb_s_hll_mat_mod.o psb_s_ell_mat_mod.o \
	psb_c_hll_mat_mod.o psb_c_ell_mat_mod.o \
	psb_z_hll_mat_mod.o psb_z_ell_mat_mod.o \
	psb_d_dia_mat_mod.o psb_d_hdia_mat_mod.o \
	psb_ext_mod.o

COBJS= 

OBJS=$(COBJS) $(FOBJS)

lib: cmp ilib copy

cmp: $(OBJS) 
	ar cur $(LIBNAME) $(OBJS)

copy: cmp ilib
	/bin/cp -p $(LIBNAME) *$(.mod) $(LIBDIR)

psb_ext_mod.o: psb_d_dia_mat_mod.o  \
        psb_d_ell_mat_mod.o psb_d_hll_mat_mod.o  \
	psb_s_hll_mat_mod.o psb_s_ell_mat_mod.o \
	psb_c_hll_mat_mod.o psb_c_ell_mat_mod.o \
	psb_z_hll_mat_mod.o psb_z_ell_mat_mod.o \
	psb_d_hdia_mat_mod.o

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



