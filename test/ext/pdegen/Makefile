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
FINCLUDES=$(FMFLAG)$(MODDIR) $(FMFLAG)$(INCDIR) $(FMFLAG). $(FMFLAG)$(PSBINCDIR)  $(FMFLAG)$(PSBMODDIR)


EXEDIR=./runs

all: ppde3d spde3d ppde2d spde2d


ppde3d: ppde3d.o 
	$(FLINK)  ppde3d.o -o ppde3d $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv ppde3d $(EXEDIR)


spde3d: spde3d.o 
	$(FLINK) spde3d.o  -o spde3d $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv spde3d $(EXEDIR)

ppde2d: ppde2d.o 
	$(FLINK)  ppde2d.o -o ppde2d $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv ppde2d $(EXEDIR)


spde2d: spde2d.o 
	$(FLINK) spde2d.o  -o spde2d $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv spde2d $(EXEDIR)


clean: 
	/bin/rm -f ppde3d.o spde3d.o  ppde2d.o spde2d.o *$(.mod) \
	$(EXEDIR)/ppde3d $(EXEDIR)/spde3d $(EXEDIR)/ppde2d $(EXEDIR)/spde2d 

verycleanlib: 
	(cd ../..; make veryclean)
lib:
	(cd ../../; make library)



