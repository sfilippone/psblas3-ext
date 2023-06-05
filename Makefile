include Make.inc

all: dirs objs $(TARGETS)
	@echo "====================================="
	@echo "PSBLAS-GPU library Compilation Successful."

objs: extobj $(OBJST)
extd: dirs

dirs:
	(if test ! -d lib ; then mkdir lib; fi)
	(if test ! -d include ; then mkdir include; fi; $(INSTALL_DATA) Make.inc  include/Make.inc.ext)
	(if test ! -d modules ; then mkdir modules; fi;)	
extd: extobj
	$(MAKE) -C ext lib LIBNAME=$(PSB_EXTLIBNAME)
gpud: extd gpuobj
	$(MAKE) -C gpu lib LIBNAME=$(PSB_GPULIBNAME)
rsbd: rsbobj
	$(MAKE) -C rsb lib LIBNAME=$(PSB_RSBLIBNAME)
extobj: 
	$(MAKE) -C ext objs
gpuobj: extobj
	$(MAKE) -C gpu objs
rsbobj:
	$(MAKE) -C rsb objs


install: all
	(mkdir -p $(INSTALL_INCLUDEDIR) &&\
	   $(INSTALL_DATA) Make.inc  $(INSTALL_INCLUDEDIR)/Make.inc.ext)
	(mkdir -p $(INSTALL_LIBDIR) &&\
	   $(INSTALL_DATA) lib/*.a  $(INSTALL_LIBDIR))
	(mkdir -p $(INSTALL_MODULESDIR) && \
	   $(INSTALL_DATA) modules/*$(.mod) $(INSTALL_MODULESDIR))
	(mkdir -p $(INSTALL_INCLUDEDIR) && \
		   (if test -f include/cintrf.h ; then \
		$(INSTALL_DATA)  include/*.h $(INSTALL_INCLUDEDIR); fi) )

clean: 
	$(MAKE) -C ext clean
	$(MAKE) -C rsb clean
	$(MAKE) -C gpu clean

cleanlib:
	(cd lib; /bin/rm -f *.a *$(.mod) *$(.fh))
	(cd include; /bin/rm -f *.a *$(.mod) *$(.fh))
veryclean: cleanlib clean
