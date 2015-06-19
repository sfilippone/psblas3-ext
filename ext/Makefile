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
FINCLUDES=$(FMFLAG). $(FMFLAG)$(INCDIR)   $(FMFLAG)$(PSBINCDIR) $(FIFLAG). $(LIBRSB_DEFINES) 
CINCLUDES=
LIBNAME=libpsb_ext.a


FOBJS=  psb_d_ell_mat_mod.o psb_d_hll_mat_mod.o \
	psb_s_hll_mat_mod.o psb_s_ell_mat_mod.o \
	psb_c_hll_mat_mod.o psb_c_ell_mat_mod.o \
	psb_z_hll_mat_mod.o psb_z_ell_mat_mod.o \
	psb_d_dia_mat_mod.o psb_d_hdia_mat_mod.o \
	psb_s_dia_mat_mod.o psb_s_hdia_mat_mod.o \
	psb_c_dia_mat_mod.o psb_c_hdia_mat_mod.o \
	psb_z_dia_mat_mod.o psb_z_hdia_mat_mod.o \
	psb_d_dns_mat_mod.o \
	psi_ext_util_mod.o psi_i_ext_util_mod.o \
	psi_s_ext_util_mod.o psi_c_ext_util_mod.o \
	psi_d_ext_util_mod.o psi_z_ext_util_mod.o \
	psb_ext_mod.o

COBJS= 

OBJS=$(COBJS) $(FOBJS)

lib: cmp mcopy ilib copy

cmp: $(OBJS) 
	ar cur $(LIBNAME) $(OBJS)

copy: cmp ilib
	/bin/cp -p $(LIBNAME) $(LIBDIR)
mcopy: cmp 
	/bin/cp -p *$(.mod) $(INCDIR)

psb_ext_mod.o: psb_s_dia_mat_mod.o   psb_d_dia_mat_mod.o  \
	psb_c_dia_mat_mod.o   psb_z_dia_mat_mod.o  \
        psb_d_ell_mat_mod.o psb_d_hll_mat_mod.o  \
	psb_s_hll_mat_mod.o psb_s_ell_mat_mod.o \
	psb_c_hll_mat_mod.o psb_c_ell_mat_mod.o \
	psb_z_hll_mat_mod.o psb_z_ell_mat_mod.o \
	psb_s_hdia_mat_mod.o psb_d_hdia_mat_mod.o \
	psb_c_hdia_mat_mod.o psb_z_hdia_mat_mod.o \
	psb_d_dns_mat_mod.o
#	psb_d_rsb_mat_mod.o psb_d_hdia_mat_mod.o 
psi_ext_util_mod.o: psi_i_ext_util_mod.o \
	psi_s_ext_util_mod.o psi_c_ext_util_mod.o \
	psi_d_ext_util_mod.o psi_z_ext_util_mod.o 

psb_s_dia_mat_mod.o psb_c_dia_mat_mod.o psb_d_dia_mat_mod.o psb_z_dia_mat_mod.o: psi_ext_util_mod.o
psb_s_hdia_mat_mod.o psb_c_hdia_mat_mod.o psb_d_hdia_mat_mod.o psb_z_hdia_mat_mod.o: psi_ext_util_mod.o
psb_s_hll_mat_mod.o psb_c_hll_mat_mod.o psb_d_hll_mat_mod.o psb_z_hll_mat_mod.o: psi_ext_util_mod.o

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



