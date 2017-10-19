include ../../Make.inc
#
# Libraries used
#
TOPDIR=../..
LIBDIR=$(TOPDIR)/lib
MODDIR=$(TOPDIR)/modules
PSBLIBDIR=$(PSBLASDIR)/lib/
PSBINCDIR=$(PSBLASDIR)/include
PSBLAS_LIB= -L$(PSBLIBDIR) -lpsb_util -lpsb_krylov -lpsb_prec -lpsb_base \
	 -lpsb_gpu -L$(LIBDIR) -lamd
MLDLIBDIR=$(MLDDIR)/lib
MLDINCDIR=$(MLDDIR)/include 
MLDLIBS=-L$(MLDLIBDIR) -lmld_prec
AINVDIR=/home/sfilippo/NUMERICAL/NewPSBLAS/mld-ainv
AINVINCDIR=$(AINVDIR)/lib
AINVLIBDIR=$(AINVDIR)/lib
AINVLIBS=-L$(AINVLIBDIR) -lainv_prec

LDLIBS=$(PSBLDLIBS) 
#
# Compilers and such
#
CCOPT= -g
FINCLUDES=$(FMFLAG)$(MODDIR) $(FMFLAG)$(LIBDIR) $(FMFLAG). $(FMFLAG)$(PSBINCDIR) \
          $(FMFLAG)$(MLDINCDIR) $(FMFLAG)$(AINVINCDIR) 


EXEDIR=./runs

all: mldpde mldpde2d

ppde: ppde.o 
	$(FLINK)  ppde.o -o ppde $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv ppde $(EXEDIR)
mldpde.o: pde3d_exp_mod.o pde3d_gauss_mod.o pde3d_base_mod.o
mldpde: mldpde.o pde3d_exp_mod.o pde3d_gauss_mod.o  pde3d_base_mod.o
	$(FLINK)  mldpde.o pde3d_exp_mod.o pde3d_gauss_mod.o pde3d_base_mod.o -o mldpde $(AINVLIBS) $(MLDLIBS) $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv mldpde $(EXEDIR)
mldpde2d.o:  pde2d_exp_mod.o pde2d_gauss_mod.o pde2d_base_mod.o
mldpde2d: mldpde2d.o  pde2d_exp_mod.o pde2d_gauss_mod.o pde2d_base_mod.o
	$(FLINK)  mldpde2d.o  pde2d_exp_mod.o pde2d_gauss_mod.o pde2d_base_mod.o -o mldpde2d $(AINVLIBS) $(MLDLIBS) $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv mldpde2d $(EXEDIR)

mmldpde: mmldpde.o 
	$(FLINK)  mmldpde.o -o mmldpde $(AINVLIBS) $(MLDLIBS) $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv mmldpde $(EXEDIR)

ppeps2d: ppeps2d.o 
	$(FLINK)  ppeps2d.o -o ppeps2d  $(AINVLIBS) $(MLDLIBS)  $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv ppeps2d $(EXEDIR)
 

ppde2d: ppde2d.o 
	$(FLINK)  ppde2d.o -o ppde2d $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv ppde2d $(EXEDIR)

taxpby: taxpby.o 
	$(FLINK)  taxpby.o -o taxpby $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv taxpby $(EXEDIR)


spde: spde.o 
	$(FLINK) spde.o  -o spde $(PSBLAS_LIB) $(LDLIBS)
	/bin/mv spde $(EXEDIR)


clean: 
	/bin/rm -f *.o *$(.mod)\
	 mldpde.o mmldpde.o ppeps2d.o  $(EXEDIR)/ppde  $(EXEDIR)/mldpde2d  $(EXEDIR)/mldpde 
verycleanlib: 
	(cd ../..; make veryclean)
lib:
	(cd ../../; make library)



