include ../../Make.inc
#
# Libraries used
#
LIBDIR=../../lib/
PSBLIBDIR=$(PSBLASDIR)/lib/
OPTDIR=$(LIBDIR)
PSBINCDIR=$(PSBLASDIR)/include
PSBLAS_LIB= -L$(LIBDIR) -L$(PSBLIBDIR) -lpsb_ext $(LGPU) $(LRSB) -lpsb_util -lpsb_base 
#PSBLAS_LIB= -L$(LIBDIR) -L$(PSBLIBDIR) -lpsb_ext -lpsb_gpu -lpsb_rsb -lpsb_util -lpsb_base 
#PSBLAS_LIB= -L$(LIBDIR) -L$(PSBLIBDIR) -lpsb_ext -lpsb_gpu $(LIBRSB_LIBS) -lpsb_util -lpsb_base 
INCDIR=../../include

LDLIBS=$(PSBLDLIBS)

FINCLUDES=$(FMFLAG)$(INCDIR) $(FMFLAG). $(FMFLAG)$(PSBINCDIR) $(LIBRSB_DEFINES)
#FINCLUDES=$(PSBLAS_LIB) $(FMFLAG). $(FMFLAG)$(PSBINCDIR)

ZTOBJS=z_file_spmv.o data_input.o
CTOBJS=c_file_spmv.o data_input.o
DTOBJS=d_file_spmv.o data_input.o
STOBJS=s_file_spmv.o data_input.o
DPGOBJS=dpdegenmv.o
SPGOBJS=spdegenmv.o
#SIMPLEOBJS=simple.o
COMPFORMOBJS=compform.o data_input.o
EXEDIR=./runs

all: dpdegenmv spdegenmv compform

compform: $(COMPFORMOBJS)
	$(F90LINK) $(LOPT) $(COMPFORMOBJS) -fopenmp -o compform $(FINCLUDES) $(PSBLAS_LIB) $(LDLIBS) $(LIBRSB_LIBS)
	/bin/mv compform $(EXEDIR)
# simple: $(SIMPLEOBJS)
# 	$(F90LINK) $(LOPT) $(SIMPLEOBJS) -fopenmp -o simple $(FINCLUDES) $(PSBLAS_LIB) $(LDLIBS) $(LIBRSB_LIBS)
# 	/bin/mv simple $(EXEDIR)
dpdegenmv: $(DPGOBJS)
	$(F90LINK) $(LOPT) $(DPGOBJS) -fopenmp -o dpdegenmv $(FINCLUDES) $(PSBLAS_LIB) $(LDLIBS) 
	/bin/mv dpdegenmv $(EXEDIR) 
spdegenmv: $(SPGOBJS)
	$(F90LINK) $(LOPT) $(SPGOBJS) -o spdegenmv $(PSBLAS_LIB) $(LDLIBS) 
	/bin/mv spdegenmv $(EXEDIR) 

z_file_spmv: $(ZTOBJS)
	$(F90LINK) $(LOPT) $(ZTOBJS) -o z_file_spmv $(PSBLAS_LIB) $(LDLIBS) 
	/bin/mv z_file_spmv $(EXEDIR) 
c_file_spmv: $(CTOBJS)
	$(F90LINK) $(LOPT) $(CTOBJS) -o c_file_spmv $(PSBLAS_LIB) $(LDLIBS) 
	/bin/mv c_file_spmv $(EXEDIR) 
d_file_spmv: $(DTOBJS)
	$(F90LINK) $(LOPT) $(DTOBJS) -o d_file_spmv $(PSBLAS_LIB) $(LDLIBS) 
	/bin/mv d_file_spmv $(EXEDIR) 
s_file_spmv: $(STOBJS)
	$(F90LINK) $(LOPT) $(STOBJS) -o s_file_spmv $(PSBLAS_LIB) $(LDLIBS) 
	/bin/cp s_file_spmv $(EXEDIR) 

d_file_spmv.o s_file_spmv.o compformats.o: data_input.o

clean: 
	/bin/rm -f $(DTOBJS) $(STOBJS) $(DPGOBJS) $(SPGOBJS) $(SIMPLEOBJS) $(ZTOBJS) $(CTOBJS) $(COMPFORMOBJS) \
	$(EXEDIR)/dpdegenmv $(EXEDIR)/spdegenmv \
	$(EXEDIR)/d_file_spmv $(EXEDIR)/s_file_spmv \
	$(EXEDIR)/z_file_spmv $(EXEDIR)/c_file_spmv \
	$(EXEDIR)/simple $(EXEDIR)/compform
	$(EXEDIR)/data

lib:
	(cd ../../; make library)
verycleanlib:
	(cd ../../; make veryclean)
