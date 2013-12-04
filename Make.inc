# This is Make.inc file generated automatically by the PSBLAS configure script.
# It should be ready to use, included by Makefile.
# If it gives problems, consider editing it.
PSBLASDIR=/server/fanfarillo/psblas
include $(PSBLASDIR)/Make.inc


PSB_EXTLIBNAME=libpsb_ext.a

INSTALL=/usr/bin/install -c
INSTALL_DATA=${INSTALL} -m 644
INSTALL_DIR=/usr/local/psblasgpu
INSTALL_LIBDIR=/usr/local/psblasgpu/lib
INSTALL_INCLUDEDIR=/usr/local/psblasgpu/include
INSTALL_DOCSDIR=/usr/local/psblasgpu/docs

LIBRSB_LIBS=
LIBRSB_INCDIR=
LIBRSB_INCLUDES=
LIBRSB_DEFINES=

CDEFINES= $(PSBCDEFINES) $(LIBRSB_INCLUDES)
FDEFINES= $(LIBRSB_INCLUDES) $(LIBRSB_LIB) $(PSBFDEFINES)
