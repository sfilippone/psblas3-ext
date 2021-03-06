include Make.inc

all: libd $(TARGETS)
	@echo "====================================="
	@echo "PSBLAS-GPU library Compilation Successful."

extd: libd

libd:
	(if test ! -d lib ; then mkdir lib; fi)
	(if test ! -d include ; then mkdir include; fi; $(INSTALL_DATA) Make.inc  include/Make.inc.ext)
	(if test ! -d modules ; then mkdir modules; fi;)	
extd:
	$(MAKE) -C ext lib LIBNAME=$(PSB_EXTLIBNAME)
gpud: extd
	$(MAKE) -C gpu lib LIBNAME=$(PSB_GPULIBNAME)
rsbd:
	$(MAKE) -C rsb lib LIBNAME=$(PSB_RSBLIBNAME)

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
