include ../Make.inc
#
# Libraries used
#
PSBLIBDIR=$(PSBLASDIR)/lib/
PSBINCDIR=$(PSBLASDIR)/include
PSBMODDIR=$(PSBLASDIR)/modules
LIBDIR=../lib
INCDIR=../include
PSBLAS_LIB= -L$(PSBLIBDIR)  -lpsb_util  -lpsb_base
#-lpsb_util -lpsb_krylov -lpsb_prec -lpsb_base 
LDLIBS=$(PSBLDLIBS)
#
# Compilers and such
#
#CCOPT= -g
FINCLUDES=$(FMFLAG). $(FMFLAG)$(PSBMODDIR) $(FMFLAG)$(PSBINCDIR) $(FIFLAG) . $(LIBRSB_INCLUDES) $(LIBRSB_DEFINES)
CINCLUDES=-I$(GPU_INCDIR) -I$(CUDA_INCDIR)
LIBNAME=libpsb_rsb.a


FOBJS=  rsb_mod.o psb_d_rsb_mat_mod.o \
	psb_rsb_penv_mod.o psb_rsb_mod.o

COBJS= rsb_int.o

OBJS=$(COBJS) $(FOBJS)

lib: cmp mcopy ilib copy

cmp: $(OBJS) 
	ar cur $(LIBNAME) $(OBJS)

copy: cmp ilib
	/bin/cp -p $(LIBNAME) $(LIBDIR)

mcopy: cmp
	/bin/cp -p $(LIBNAME) *$(.mod) $(INCDIR)

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



