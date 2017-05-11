TOPDIR=../../..
include $(TOPDIR)/Make.inc
LIBDIR=$(TOPDIR)lib/
PSBLIBDIR=$(PSBLASDIR)/lib/
PSBINCDIR=$(PSBLASDIR)/include
PSBLAS_LIB= -L$(PSBLIBDIR) -lpsb_util -lpsb_krylov -lpsb_prec -lpsb_base \
	 -lpsb_gpu -L$(LIBDIR)
MLDLIBDIR=$(MLDDIR)/lib
MLDINCDIR=$(MLDDIR)/include 
MLDLIBS=-L$(MLDLIBDIR) -lmld_prec
AINVDIR=/home/sfilippo/NUMERICAL/NewPSBLAS/mld-ainv
AINVINCDIR=$(AINVDIR)/lib
AINVLIBDIR=$(AINVDIR)/lib
AINVLIBS=-L$(AINVLIBDIR) -lainv_prec

LDLIBS=$(PSBGPULDLIBS)
#
# Compilers and such
#
CCOPT= -g
FINCLUDES=$(FMFLAG)$(LIBDIR) $(FMFLAG). $(FMFLAG)$(PSBINCDIR) \
          $(FMFLAG)$(MLDINCDIR) $(FMFLAG)$(AINVINCDIR) 

DFSOBJS=df_sample.o data_input.o
SFSOBJS=sf_sample.o data_input.o
CFSOBJS=cf_sample.o data_input.o
ZFSOBJS=zf_sample.o data_input.o

EXEDIR=./runs

all:  df_sample

df_sample: $(DFSOBJS)
	$(FLINK) $(LINKOPT) $(DFSOBJS) -o df_sample \
	$(AINVLIBS) $(MLDLIBS) $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv df_sample $(EXEDIR) 

df_sample.o: data_input.o

clean: 
	/bin/rm -f $(DFOBJS) $(ZFOBJS) $(SFSOBJS)  $(DFSOBJS) \
        *$(.mod)  $(EXEDIR)/df_bench $(EXEDIR)/zf_bench \
	$(EXEDIR)/sf_sample $(EXEDIR)/cf_sample  \
	$(EXEDIR)/df_sample $(EXEDIR)/zf_sample

lib:
	(cd ../../; make library)
verycleanlib:
	(cd ../../; make veryclean)

