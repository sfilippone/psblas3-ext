# This is Make.inc file generated automatically by the PSBLAS configure script.
# It should be ready to use, included by Makefile.
# If it gives problems, consider editing it.
PSBLASDIR=/usr/local/psblas-3.0
include $(PSBLASDIR)/Make.inc


PSB_EXTLIBNAME=libpsb_ext.a

INSTALL=/usr/bin/install -c
INSTALL_DATA=${INSTALL} -m 644
INSTALL_DIR=/usr/local/psblasgpu
INSTALL_LIBDIR=/usr/local/psblasgpu/lib
INSTALL_INCLUDEDIR=/usr/local/psblasgpu/include
INSTALL_DOCSDIR=/usr/local/psblasgpu/docs

LIBRSB_LIBS=-lrsb -L/home/alex/Scrivania/PSBLAS/svn_librsb
LIBRSB_INCDIR=/home/alex/Scrivania/PSBLAS/svn_librsb
LIBRSB_INCLUDES=-I/home/alex/Scrivania/PSBLAS/svn_librsb
LIBRSB_DEFINES=-DHAVE_RSB

CDEFINES=-DHAVE_RSB  $(PSBCDEFINES) $(LIBRSB_INCLUDES)
FDEFINES=-DHAVE_RSB   $(LIBRSB_INCLUDES) $(LIBRSB_LIB) $(PSBFDEFINES)
