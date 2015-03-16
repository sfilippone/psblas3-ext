include Make.inc

all: libd $(TARGETS)
	@echo "====================================="
	@echo "PSBLAS-GPU library Compilation Successful."

extd: libd

libd:
	(if test ! -d lib ; then mkdir lib; fi)
	(if test ! -d include ; then mkdir include; fi)
extd:
	cd ext && $(MAKE) lib LIBNAME=$(PSB_EXTLIBNAME)
gpud: extd
	cd gpu && $(MAKE) lib LIBNAME=$(PSB_GPULIBNAME)
rsbd:
	cd rsb && $(MAKE) lib LIBNAME=$(PSB_RSBLIBNAME)

install: all
	(./mkdir.sh  $(INSTALL_INCLUDEDIR) &&\
	   $(INSTALL_DATA) Make.inc  $(INSTALL_INCLUDEDIR)/Make.inc.ext)
	(./mkdir.sh  $(INSTALL_LIBDIR) &&\
	   $(INSTALL_DATA) lib/*.a  $(INSTALL_LIBDIR))
	(./mkdir.sh  $(INSTALL_INCLUDEDIR) && \
	   $(INSTALL_DATA) include/*$(.mod) $(INSTALL_INCLUDEDIR))

clean: 
	cd ext &&  $(MAKE) clean
	cd rsb &&  $(MAKE) clean
	cd gpu &&  $(MAKE) clean

cleanlib:
	(cd lib; /bin/rm -f *.a *$(.mod) *$(.fh))
	(cd include; /bin/rm -f *.a *$(.mod) *$(.fh))
veryclean: cleanlib clean
