TOPDIR=../../..
include $(TOPDIR)/Make.inc
#
# Libraries used
#
LIBDIR=$(TOPDIR)/lib/
PSBLIBDIR=$(PSBLASDIR)/lib/
OPTDIR=$(LIBDIR)
PSBINCDIR=$(PSBLASDIR)/include
PSBMODDIR=$(PSBLASDIR)/modules
PSBLAS_LIB= -L$(LIBDIR) -L$(PSBLIBDIR) $(LGPU) -lpsb_ext $(LRSB) -lpsb_util -lpsb_base 
INCDIR=$(TOPDIR)/include
MODDIR=$(TOPDIR)/modules

LDLIBS=$(PSBGPULDLIBS) 

FINCLUDES=$(FMFLAG)$(MODDIR) $(FMFLAG)$(INCDIR) $(FMFLAG). $(FMFLAG)$(PSBMODDIR)  $(FMFLAG)$(PSBINCDIR) $(LIBRSB_DEFINES)

ZTOBJS=z_file_spmv.o data_input.o
CTOBJS=c_file_spmv.o data_input.o
DTOBJS=d_file_spmv.o data_input.o
STOBJS=s_file_spmv.o data_input.o
DPGOBJS=dpdegenmv.o
SPGOBJS=spdegenmv.o
EXEDIR=./runs

all: pgen file
pgen: dpdegenmv spdegenmv 
file: s_file_spmv c_file_spmv d_file_spmv z_file_spmv

dpdegenmv: $(DPGOBJS)
	$(FLINK) $(LOPT) $(DPGOBJS) -fopenmp -o dpdegenmv $(FINCLUDES) $(PSBLAS_LIB) $(LDLIBS) 
	/bin/mv dpdegenmv $(EXEDIR) 
spdegenmv: $(SPGOBJS)
	$(FLINK) $(LOPT) $(SPGOBJS) -o spdegenmv $(PSBLAS_LIB) $(LDLIBS) 
	/bin/mv spdegenmv $(EXEDIR) 

z_file_spmv: $(ZTOBJS)
	$(FLINK) $(LOPT) $(ZTOBJS) -o z_file_spmv $(PSBLAS_LIB) $(LDLIBS) 
	/bin/mv z_file_spmv $(EXEDIR) 
c_file_spmv: $(CTOBJS)
	$(FLINK) $(LOPT) $(CTOBJS) -o c_file_spmv $(PSBLAS_LIB) $(LDLIBS) 
	/bin/mv c_file_spmv $(EXEDIR) 
d_file_spmv: $(DTOBJS)
	$(FLINK) $(LOPT) $(DTOBJS) -o d_file_spmv $(PSBLAS_LIB) $(LDLIBS) 
	/bin/mv d_file_spmv $(EXEDIR) 
s_file_spmv: $(STOBJS)
	$(FLINK) $(LOPT) $(STOBJS) -o s_file_spmv $(PSBLAS_LIB) $(LDLIBS) 
	/bin/cp s_file_spmv $(EXEDIR) 

d_file_spmv.o s_file_spmv.o z_file_spmv.o c_file_spmv.o: data_input.o

clean: 
	/bin/rm -f $(DTOBJS) $(STOBJS) $(DPGOBJS) $(SPGOBJS) $(ZTOBJS) $(CTOBJS)  \
	$(EXEDIR)/dpdegenmv $(EXEDIR)/spdegenmv \
	$(EXEDIR)/d_file_spmv $(EXEDIR)/s_file_spmv \
	$(EXEDIR)/z_file_spmv $(EXEDIR)/c_file_spmv 

lib:
	(cd ../../; make library)
verycleanlib:
	(cd ../../; make veryclean)
