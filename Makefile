include Make.inc

all: libd srcd
	@echo "====================================="
	@echo "PSBLAS-GPU library Compilation Successful."

srcd: libd

libd:
	(if test ! -d lib ; then mkdir lib; fi)
	(if test ! -d include ; then mkdir include; fi)
srcd:
	cd src && $(MAKE) lib LIBNAME=$(PSB_EXTLIBNAME)

install: all
	(./mkdir.sh  $(INSTALL_DIR) &&\
	   $(INSTALL_DATA) Make.inc  $(INSTALL_DIR))
	(./mkdir.sh  $(INSTALL_LIBDIR) &&\
	   $(INSTALL_DATA) lib/*.a  $(INSTALL_LIBDIR))
	(./mkdir.sh  $(INSTALL_INCLUDEDIR) && \
	   $(INSTALL_DATA) include/*$(.mod) $(INSTALL_INCLUDEDIR))

clean: 
	cd src &&  $(MAKE) clean

cclean: 
	cd src &&  $(MAKE) cclean

cleanlib:
	(cd lib; /bin/rm -f *.a *$(.mod) *$(.fh))
veryclean: cleanlib clean
