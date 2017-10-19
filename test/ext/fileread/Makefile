TOPDIR=../../..
include $(TOPDIR)/Make.inc
#
# Libraries used
#
LIBDIR=$(TOPDIR)/lib/
INCDIR=$(TOPDIR)/include
MODDIR=$(TOPDIR)/modules
PSBLIBDIR=$(PSBLASDIR)/lib/
PSBINCDIR=$(PSBLASDIR)/include
PSBMODDIR=$(PSBLASDIR)/modules
PSBLAS_LIB= -L$(LIBDIR) -L$(PSBLIBDIR) -lpsb_ext -lpsb_util -lpsb_krylov -lpsb_prec -lpsb_base 

LDLIBS=$(PSBEXTLDLIBS)
#
# Compilers and such
#
CCOPT= -g
FINCLUDES=$(FMFLAG)$(INCDIR) $(FMFLAG)$(MODDIR) $(FMFLAG). $(FMFLAG)$(PSBINCDIR)  $(FMFLAG)$(PSBMODDIR)

DFSOBJS=df_sample.o getp.o

EXEDIR=./runs

all:  df_sample

df_sample: $(DFSOBJS)
	$(FLINK) $(LINKOPT) $(DFSOBJS) -o df_sample \
	$(PSBLAS_LIB) $(LDLIBS)
	/bin/mv df_sample $(EXEDIR) 

df_sample.o: getp.o

clean: 
	/bin/rm -f $(DFSOBJS) \
        *$(.mod)  	$(EXEDIR)/df_sample 

lib:
	(cd ../../; make library)
verycleanlib:
	(cd ../../; make veryclean)

