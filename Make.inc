# This is Make.inc file generated automatically by the PSBLAS configure script.
# It should be ready to use, included by Makefile.
# If it gives problems, consider editing it.
PSBLASDIR=/usr/local/PSBLAS/V3/gnu49-k8

include $(PSBLASDIR)/Make.inc


PSB_EXTLIBNAME=libpsb_ext.a

INSTALL=/usr/bin/install -c
INSTALL_DATA=${INSTALL} -m 644
INSTALL_DIR=/usr/local/PSBLAS/EXT/gnu49-k8
INSTALL_LIBDIR=/usr/local/PSBLAS/EXT/gnu49-k8/lib
INSTALL_INCLUDEDIR=/usr/local/PSBLAS/EXT/gnu49-k8/include
INSTALL_DOCSDIR=/usr/local/PSBLAS/EXT/gnu49-k8/docs

CDEFINES= $(PSBCDEFINES)
FDEFINES= $(PSBFDEFINES)

