dnl
dnl $Id$
dnl
dnl 20080206
dnl M4 macros for the PSBLAS library and useful for packages using PSBLAS.
dnl

dnl @synopsis PAC_CHECK_LIBS
dnl
dnl Tries to detect the presence of a specific function among various libraries, using AC_CHECK_LIB
dnl repeatedly on the specified libraries.
dnl 
dnl Example use:
dnl
dnl PAC_CHECK_LIBS([atlas blas],
dnl		[dgemm],
dnl		[have_dgemm=yes],
dnl		[have_dgemm=no])
dnl 
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl
dnl 20080211 modified slighty from original.
AC_DEFUN([PAC_CHECK_LIBS],
[
 pac_check_libs_ok=no
 [for pac_check_libs_f in $2 
 do ]
 [for pac_check_libs_l in $1 
 do ]
    if test x"$pac_check_libs_ok" == xno ; then
     AC_CHECK_LIB([$pac_check_libs_l],[$pac_check_libs_f], [pac_check_libs_ok=yes; pac_check_libs_LIBS="-l$pac_check_libs_l"],[],[$5])
    fi
  done
  done
 # Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
 [ if test x"$pac_check_libs_ok" = xyes ; then
	$3
 else
        pac_check_libs_ok=no
        $4
 fi
 ]
])dnl 

dnl @synopsis PAC_FORTRAN_FUNC_MOVE_ALLOC( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program with move_alloc (a Fortran 2003 function).
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl
AC_DEFUN([PAC_FORTRAN_HAVE_MOVE_ALLOC],
ac_exeext=''
ac_ext='f'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([for Fortran MOVE_ALLOC intrinsic])
cat > conftest.$ac_ext <<EOF
           program test_move_alloc
               integer, allocatable :: a(:), b(:)
               allocate(a(3))
               call move_alloc(a, b)
               print *, allocated(a), allocated(b)
               print *, b
           end program test_move_alloc
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [rm -rf conftest*
  $1])
else
  AC_MSG_RESULT([no])	
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  rm -rf conftest*
  $2
])dnl
fi
rm -f conftest*])



dnl @synopsis PAC_CHECK_HAVE_GFORTRAN( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will check if MPIFC is $FC.
dnl The check will proceed by compiling a small Fortran program
dnl containing the __GNUC__ macro, which should be defined in the
dnl gfortran compiled programs.
dnl
dnl On pass, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl
AC_DEFUN(PAC_CHECK_HAVE_GFORTRAN,
ac_exeext=''
ac_ext='F'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[
cat > conftest.$ac_ext <<EOF
           program main
#ifdef __GNUC__ 
              print *, "GCC!"
#else
        this program will fail
#endif
           end

EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  ifelse([$1], , :, [rm -rf conftest*
  $1])
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  rm -rf conftest*
  $2
])dnl
fi
rm -f conftest*])



dnl @synopsis PAC_HAVE_MODERN_GFORTRAN( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will check if the GNU fortran version is suitable for PSBLAS.
dnl If yes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl
dnl Note : Will use MPIFC; if unset, will use '$FC'.
dnl 
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl
AC_DEFUN(PAC_HAVE_MODERN_GFORTRAN,
ac_exeext=''
ac_ext='F'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([GNU Fortran version at least 4.6])
cat > conftest.$ac_ext <<EOF
           program main
#if ( __GNUC__ >= 4 && __GNUC_MINOR__ >= 6 ) || ( __GNUC__ > 4 )
              print *, "ok"
#else
        this program will fail
#endif
           end

EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([ yes.])
  ifelse([$1], , :, [rm -rf conftest*
  $1])
else
 AC_MSG_RESULT([ no.])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  rm -rf conftest*
  $2
])dnl
fi
rm -f conftest*])


dnl @synopsis PAC_FORTRAN_CHECK_HAVE_MPI_MOD( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will determine if the fortran compiler MPIFC needs to include mpi.h or needs
dnl to use the mpi module.
dnl
dnl If yes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl 
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl Modified Salvatore Filippone <salvatore.filippone@uniroma2.it>
dnl
AC_DEFUN(PAC_FORTRAN_CHECK_HAVE_MPI_MOD,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([MPI Fortran interface])
cat > conftest.$ac_ext <<EOF
           program test
             use mpi
           end program test
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([ use mpi ])
  ifelse([$1], , :, [rm -rf conftest*
  $1])
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
  AC_MSG_RESULT([ include mpif.h ])
ifelse([$2], , , [  rm -rf conftest*
  $2
])dnl
fi
rm -f conftest*])



dnl @synopsis PAC_ARG_WITH_FLAGS(lcase_name, UCASE_NAME)
dnl
dnl Test for --with-lcase_name="compiler/loader flags".  if defined, prepend 
dnl flags to standard UCASE_NAME definition.
dnl
dnl Use this macro to facilitate additional special flags that should be
dnl passed on to the preprocessor/compilers/loader.
dnl
dnl NOTE : Renamed after TAC_ARG_WITH_FLAGS as in the Trilinos-8.0.4 package.
dnl 
dnl NOTE : This macro works in a way the user should invoke
dnl         --with-flags=...
dnl	   only once, otherwise the first one will take effect.
dnl
dnl Example use:
dnl 
dnl PAC_ARG_WITH_FLAGS(cxxflags, CXXFLAGS)
dnl 
dnl tests for --with-cxxflags and pre-pends to CXXFLAGS
dnl 
dnl @author Mike Heroux <mheroux@cs.sandia.gov>
dnl @notes  Michele Martone <michele.martone@uniroma2.it>
dnl
AC_DEFUN([PAC_ARG_WITH_FLAGS],
[
AC_MSG_CHECKING([whether additional [$2] flags should be added (should be invoked only once)])
dnl AC_MSG_CHECKING([whether additional [$2] flags should be added])
AC_ARG_WITH($1,
AC_HELP_STRING([--with-$1], 
[additional [$2] flags to be added: will prepend to [$2]]),
[
$2="${withval} ${$2}"
AC_MSG_RESULT([$2 = ${$2}])
],
AC_MSG_RESULT(no)
)
])


dnl @synopsis PAC_ARG_WITH_LIBS
dnl
dnl Test for --with-libs="name(s)".
dnl 
dnl Prepends the specified name(s) to the list of libraries to link 
dnl with.  
dnl
dnl note: Renamed after PAC_ARG_WITH_LIBS as in the Trilinos package.
dnl
dnl Example use:
dnl
dnl PAC_ARG_WITH_LIBS
dnl 
dnl tests for --with-libs and pre-pends to LIBS
dnl
dnl @author Jim Willenbring <jmwille@sandia.gov>
dnl
AC_DEFUN([PAC_ARG_WITH_LIBS],
[
AC_MSG_CHECKING([whether additional libraries are needed])
AC_ARG_WITH(libs,
AC_HELP_STRING([--with-libs], 
[List additional link flags  here.  For example, --with-libs=-lspecial_system_lib
or --with-libs=-L/path/to/libs]),
[
LIBS="${withval} ${LIBS}"
AC_MSG_RESULT([LIBS = ${LIBS}])
],
AC_MSG_RESULT(no)
)
]
)
dnl @synopsis PAC_ARG_WITH_PSBLAS
dnl
dnl Test for --with-psblas="pathname".
dnl 
dnl Defines the path to PSBLAS build dir.
dnl
dnl note: Renamed after PAC_ARG_WITH_LIBS as in the Trilinos package.
dnl
dnl Example use:
dnl
dnl PAC_ARG_WITH_PSBLAS
dnl 
dnl tests for --with-psblas and pre-pends to PSBLAS_PATH
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
dnl
AC_DEFUN([PAC_ARG_WITH_PSBLAS],
[
AC_ARG_WITH(psblas,
AC_HELP_STRING([--with-psblas], [The directory for PSBLAS, for example,
 --with-psblas=/opt/packages/psblas-3.0]),
[pac_cv_psblas_dir=$withval],
[pac_cv_psblas_dir=''])
]
)


dnl @synopsis PAC_FORTRAN_HAVE_PSB_LONG_INT( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile with psb_long_int_k_
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl
AC_DEFUN(PAC_FORTRAN_HAVE_PSB_LONG_INT,
ac_objext='.o'
ac_ext='f90'
ac_compile='${MPIFC-$FC} -c -o conftest${ac_objext} $FMFLAG$PSBLAS_DIR/include $FMFLAG$PSBLAS_DIR/lib  conftest.$ac_ext  1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([for version of PSBLAS supporting psb_long_int_k_])
cat > conftest.$ac_ext <<EOF
           program test
	       use psb_sparse_mod
               integer(psb_long_int_k_) :: val 
           end program test
EOF
if AC_TRY_EVAL(ac_compile) && test -s conftest${ac_objext}; then
  ifelse([$1], , :, [rm -rf conftest*
  $1])
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  rm -rf conftest*
  $2
])dnl
fi
rm -f conftest*])



dnl @synopsis PAC_ARG_SERIAL_MPI
dnl
dnl Test for --with-serial-mpi={yes|no}
dnl 
dnl 
dnl
dnl Example use:
dnl
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
dnl
AC_DEFUN([PAC_ARG_SERIAL_MPI],
[
AC_MSG_CHECKING([whether we want serial (fake) mpi])
AC_ARG_ENABLE(serial,
AC_HELP_STRING([--enable-serial], 
[Specify whether to enable a fake mpi library to run in serial mode. ]),
[
pac_cv_serial_mpi="yes";
]
dnl ,
dnl [pac_cv_serial_mpi="no";]
)
if test x"$pac_cv_serial_mpi" == x"yes" ; then
   AC_MSG_RESULT([yes.])
else
 pac_cv_serial_mpi="no";
 AC_MSG_RESULT([no.])
fi
]
)


dnl @synopsis PAC_FORTRAN_HAVE_PSBLAS( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program using the PSBLAS library
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl
AC_DEFUN(PAC_FORTRAN_HAVE_PSBLAS,
ac_objext='.o'
ac_ext='f90'
ac_compile='${MPIFC-$FC} -c -o conftest${ac_objext} $FMFLAG$PSBLAS_DIR/include $FMFLAG$PSBLAS_DIR/lib conftest.$ac_ext  1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([for working source dir of PSBLAS])
cat > conftest.$ac_ext <<EOF
           program test
	       use psb_base_mod
           end program test
EOF
if AC_TRY_EVAL(ac_compile) && test -s conftest${ac_objext}; then
  ifelse([$1], , :, [rm -rf conftest*
  $1])
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  rm -rf conftest*
  $2
])dnl
fi
rm -f conftest*])

dnl @synopsis PAC_FORTRAN_TEST_TR15581( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program checking the TR15581 Fortran extension support.
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
AC_DEFUN(PAC_FORTRAN_TEST_TR15581,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FCFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for Fortran allocatables TR15581])
i=0
while test \( -f tmpdir_$i \) -o \( -d tmpdir_$i \) ; do
  i=`expr $i + 1`
done
mkdir tmpdir_$i
cd tmpdir_$i
cat > conftest.$ac_ext <<EOF
module conftest
  type outer
    integer,  allocatable :: v(:)
  end type outer

  interface foo
    module procedure foov, food
  end interface
contains

  subroutine foov(a,b)

    implicit none
    integer, allocatable, intent(inout) :: a(:)
    integer, allocatable, intent(out) :: b(:)


    allocate(b(size(a)))

  end subroutine foov
  subroutine food(a,b)

    implicit none
    type(outer), intent(inout) :: a
    type(outer), intent(out) :: b


    allocate(b%v(size(a%v)))

  end subroutine food

end module conftest



program testtr15581
  use conftest
  type(outer) :: da, db
  integer, allocatable :: a(:), b(:)

  allocate(a(10),da%v(10))
  a = (/ (i,i=1,10) /)
  da%v = (/ (i,i=1,10) /)
  call foo(a,b)
  call foo(da,db)
  write(*,*) b
  write(*,*) db%v

end program testtr15581
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [
  $1])
else
  AC_MSG_RESULT([no])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  
  $2
])dnl
fi
cd ..
rm -fr tmpdir_$i])

dnl @synopsis PAC_FORTRAN_TEST_VOLATILE( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program checking the VOLATILE Fortran support.
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
AC_DEFUN(PAC_FORTRAN_TEST_VOLATILE,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FCFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for Fortran VOLATILE])
i=0
while test \( -f tmpdir_$i \) -o \( -d tmpdir_$i \) ; do
  i=`expr $i + 1`
done
mkdir tmpdir_$i
cd tmpdir_$i
cat > conftest.$ac_ext <<EOF
program conftest
  integer, volatile :: i, j
end program conftest
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [
  $1])
else
  AC_MSG_RESULT([no])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  
  $2
])dnl
fi
cd ..
rm -fr tmpdir_$i])



dnl @synopsis PAC_FORTRAN_TEST_EXTENDS( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program checking the EXTENDS Fortran support.
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
AC_DEFUN(PAC_FORTRAN_TEST_EXTENDS,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FCFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for Fortran EXTENDS])
i=0
while test \( -f tmpdir_$i \) -o \( -d tmpdir_$i \) ; do
  i=`expr $i + 1`
done
mkdir tmpdir_$i
cd tmpdir_$i
cat > conftest.$ac_ext <<EOF
program conftest
  type foo
    integer :: i
  end type foo
  type, extends(foo) :: bar
    integer j
  end type bar 
  type(bar) :: barvar
end program conftest
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [
  $1])
else
  AC_MSG_RESULT([no])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  
  $2
])dnl
fi
cd ..
rm -fr tmpdir_$i])

dnl @synopsis PAC_FORTRAN_TEST_CLASS_TBP( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program checking the TBP Fortran support.
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
AC_DEFUN(PAC_FORTRAN_TEST_CLASS_TBP,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FCFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for Fortran CLASS TBP])
i=0
while test \( -f tmpdir_$i \) -o \( -d tmpdir_$i \) ; do
  i=`expr $i + 1`
done
mkdir tmpdir_$i
cd tmpdir_$i
cat > conftest.$ac_ext <<EOF
module foo_mod
  type foo
    integer :: i 
  contains
    procedure, pass(a) :: doit
    procedure, pass(a) :: getit
  end type foo

  private doit,getit
contains
  subroutine  doit(a) 
    class(foo) :: a
    
    a%i = 1
    write(*,*) 'FOO%DOIT base version'
  end subroutine doit
  function getit(a) result(res)
    class(foo) :: a
    integer :: res

    res = a%i
  end function getit

end module foo_mod
program conftest
  use foo_mod
  type(foo) :: foovar
end program conftest
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [
  $1])
else
  AC_MSG_RESULT([no])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  
  $2
])dnl
fi
cd ..
rm -fr tmpdir_$i])


dnl @synopsis PAC_FORTRAN_TEST_FINAL( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program checking the FINAL Fortran support.
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
AC_DEFUN(PAC_FORTRAN_TEST_FINAL,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FCFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for Fortran FINAL])
i=0
while test \( -f tmpdir_$i \) -o \( -d tmpdir_$i \) ; do
  i=`expr $i + 1`
done
mkdir tmpdir_$i
cd tmpdir_$i
cat > conftest.$ac_ext <<EOF
module foo_mod
  type foo
    integer :: i 
  contains
    final  :: destroy_foo
  end type foo

  private destroy_foo
contains
  subroutine destroy_foo(a)
    type(foo) :: a
     ! Just a test
  end subroutine destroy_foo
end module foo_mod
program conftest
  use foo_mod
  type(foo) :: foovar
end program conftest
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [
  $1])
else
  AC_MSG_RESULT([no])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  
  $2
])dnl
fi
cd ..
rm -fr tmpdir_$i])

dnl @synopsis PAC_FORTRAN_TEST_SAME_TYPE( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program checking the SAME_TYPE_AS Fortran support.
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
AC_DEFUN(PAC_FORTRAN_TEST_SAME_TYPE,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FCFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for Fortran SAME_TYPE_AS])
i=0
while test \( -f tmpdir_$i \) -o \( -d tmpdir_$i \) ; do
  i=`expr $i + 1`
done
mkdir tmpdir_$i
cd tmpdir_$i
cat > conftest.$ac_ext <<EOF
program stt
  type foo
    integer :: i
  end type foo
  type, extends(foo) :: new_foo
    integer :: j
  end type new_foo
  type(foo) :: foov
  type(new_foo) :: nfv1, nfv2

    
  write(*,*) 'foov == nfv1? ', same_type_as(foov,nfv1)
  write(*,*) 'nfv2 == nfv1? ', same_type_as(nfv2,nfv1)
end program stt
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [
  $1])
else
  AC_MSG_RESULT([no])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  
  $2
])dnl
fi
cd ..
rm -fr tmpdir_$i])

dnl @synopsis PAC_FORTRAN_TEST_EXTENDS_TYPE( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program checking the EXTENDS_TYPE_OF Fortran support.
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
AC_DEFUN(PAC_FORTRAN_TEST_EXTENDS_TYPE,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FCFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for Fortran EXTENDS_TYPE_OF])
i=0
while test \( -f tmpdir_$i \) -o \( -d tmpdir_$i \) ; do
  i=`expr $i + 1`
done
mkdir tmpdir_$i
cd tmpdir_$i
cat > conftest.$ac_ext <<EOF
program xtt
  type foo
    integer :: i
  end type foo
  type, extends(foo) :: new_foo
    integer :: j
  end type new_foo
  type(foo) :: foov
  type(new_foo) :: nfv1, nfv2

  write(*,*) 'nfv1 extends foov? ', extends_type_of(nfv1,foov)
end program xtt
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [
  $1])
else
  AC_MSG_RESULT([no])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  
  $2
])dnl
fi
cd ..
rm -fr tmpdir_$i])


dnl @synopsis PAC_CHECK_BLACS
dnl
dnl Will try to find the BLACS
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
dnl
AC_DEFUN(PAC_CHECK_BLACS,
[AC_ARG_WITH(blacs, AC_HELP_STRING([--with-blacs=LIB], [Specify BLACSLIBNAME or -lBLACSLIBNAME or the absolute library filename.]),
        [psblas_cv_blacs=$withval],
        [psblas_cv_blacs=''])

case $psblas_cv_blacs in
	yes | "") ;;
	-* | */* | *.a | *.so | *.so.* | *.o) 
	     BLACS_LIBS="$psblas_cv_blacs" ;;
	*) BLACS_LIBS="-l$psblas_cv_blacs" ;;
esac

#
# Test user-defined BLACS
#
if test x"$psblas_cv_blacs" != "x" ; then
      save_LIBS="$LIBS";
      AC_LANG([Fortran])
      LIBS="$BLACS_LIBS $LIBS"
      AC_MSG_CHECKING([for dgesd2d in $BLACS_LIBS])
      AC_TRY_LINK_FUNC(dgesd2d, [psblas_cv_blacs_ok=yes], [psblas_cv_blacs_ok=no;BLACS_LIBS=""])
      AC_MSG_RESULT($psblas_cv_blacs_ok)

     if test x"$psblas_cv_blacs_ok" == x"yes";  then 
     AC_MSG_CHECKING([for blacs_pinfo in $BLACS_LIBS])
     AC_TRY_LINK_FUNC(blacs_pinfo, [psblas_cv_blacs_ok=yes], [psblas_cv_blacs_ok=no;BLACS_LIBS=""])
     AC_MSG_RESULT($psblas_cv_blacs_ok)
     fi 
     LIBS="$save_LIBS";
fi
AC_LANG([C])	

######################################
# System BLACS with PESSL default names. 
######################################
if test x"$BLACS_LIBS" == "x" ; then
   AC_LANG([Fortran])
   PAC_CHECK_LIBS([blacssmp blacsp2 blacs], 
	[dgesd2d],
	[psblas_cv_blacs_ok=yes; LIBS="$LIBS $pac_check_libs_LIBS "  ]
	[BLACS_LIBS="$pac_check_libs_LIBS" ]
	AC_MSG_NOTICE([BLACS libraries detected.]),[]
    )
    if test x"$BLACS_LIBS" != "x"; then 
          save_LIBS="$LIBS";
          LIBS="$BLACS_LIBS $LIBS"
          AC_MSG_CHECKING([for blacs_pinfo in $BLACS_LIBS])
          AC_LANG([Fortran])
	  AC_TRY_LINK_FUNC(blacs_pinfo, [psblas_cv_blacs_ok=yes], [psblas_cv_blacs_ok=no;BLACS_LIBS=""])
          AC_MSG_RESULT($psblas_cv_blacs_ok)
          LIBS="$save_LIBS";	
    fi 
fi
######################################
# Maybe we're looking at PESSL BLACS?#
######################################
if  test x"$BLACS_LIBS" != "x" ; then
    save_LIBS="$LIBS";
    LIBS="$BLACS_LIBS $LIBS"
    AC_MSG_CHECKING([for PESSL BLACS])
    AC_LANG([Fortran])
    AC_TRY_LINK_FUNC(esvemonp, [psblas_cv_pessl_blacs=yes], [psblas_cv_pessl_blacs=no])
    AC_MSG_RESULT($psblas_cv_pessl_blacs)
    LIBS="$save_LIBS";
fi    
if test "x$psblas_cv_pessl_blacs" == "xyes";  then
   FDEFINES="$psblas_cv_define_prepend-DHAVE_ESSL_BLACS $FDEFINES"
fi 
    

##############################################################################
#	Netlib BLACS library with default names
##############################################################################

if test x"$BLACS_LIBS" == "x" ; then
   save_LIBS="$LIBS";
   AC_LANG([Fortran])
   PAC_CHECK_LIBS([ blacs_MPI-LINUX-0 blacs_MPI-SP5-0 blacs_MPI-SP4-0 blacs_MPI-SP3-0 blacs_MPI-SP2-0 blacsCinit_MPI-ALPHA-0 blacsCinit_MPI-IRIX64-0 blacsCinit_MPI-RS6K-0 blacsCinit_MPI-SPP-0 blacsCinit_MPI-SUN4-0 blacsCinit_MPI-SUN4SOL2-0 blacsCinit_MPI-T3D-0 blacsCinit_MPI-T3E-0 
	], 
	[dgesd2d],
	[psblas_cv_blacs_ok=yes; LIBS="$LIBS $pac_check_libs_LIBS " 
	psblas_have_netlib_blacs=yes;  ]
	[BLACS_LIBS="$pac_check_libs_LIBS" ]
	AC_MSG_NOTICE([BLACS libraries detected.]),[]
    )
    
    if test x"$BLACS_LIBS" != "x" ; then	
      AC_LANG([Fortran])	   
      PAC_CHECK_LIBS([ blacsF77init_MPI-LINUX-0 blacsF77init_MPI-SP5-0 blacsF77init_MPI-SP4-0 blacsF77init_MPI-SP3-0 blacsF77init_MPI-SP2-0 blacsF77init_MPI-ALPHA-0 blacsF77init_MPI-IRIX64-0 blacsF77init_MPI-RS6K-0 blacsF77init_MPI-SPP-0 blacsF77init_MPI-SUN4-0 blacsF77init_MPI-SUN4SOL2-0 blacsF77init_MPI-T3D-0 blacsF77init_MPI-T3E-0 
 	], 
	[blacs_pinfo],
	[psblas_cv_blacs_ok=yes; LIBS="$pac_check_libs_LIBS $LIBS" ]
	[BLACS_LIBS="$pac_check_libs_LIBS $BLACS_LIBS" ]
	AC_MSG_NOTICE([Netlib BLACS Fortran initialization libraries detected.]),[]
       )
    fi

    if test x"$BLACS_LIBS" != "x" ; then	
    
      AC_LANG([C])
      PAC_CHECK_LIBS([ blacsCinit_MPI-LINUX-0 blacsCinit_MPI-SP5-0 blacsCinit_MPI-SP4-0 blacsCinit_MPI-SP3-0 blacsCinit_MPI-SP2-0 blacsCinit_MPI-ALPHA-0 blacsCinit_MPI-IRIX64-0 blacsCinit_MPI-RS6K-0 blacsCinit_MPI-SPP-0 blacsCinit_MPI-SUN4-0 blacsCinit_MPI-SUN4SOL2-0 blacsCinit_MPI-T3D-0 blacsCinit_MPI-T3E-0 
	], 
	[Cblacs_pinfo],
	[psblas_cv_blacs_ok=yes; LIBS="$pac_check_libs_LIBS $LIBS" ]
	[BLACS_LIBS="$BLACS_LIBS $pac_check_libs_LIBS" ]
	AC_MSG_NOTICE([Netlib BLACS C initialization libraries detected.]),[]
       )
    fi
    LIBS="$save_LIBS";	
fi

if test x"$BLACS_LIBS" == "x" ; then
	AC_MSG_ERROR([
	No BLACS library detected! $PACKAGE_NAME will be unusable.
	Please make sure a BLACS implementation is accessible (ex.: --with-blacs="-lblacsname -L/blacs/dir" )
	])
else 
      save_LIBS="$LIBS";
      LIBS="$BLACS_LIBS $LIBS"
      AC_MSG_CHECKING([for ksendid in $BLACS_LIBS])
      AC_LANG([Fortran])
      AC_TRY_LINK_FUNC(ksendid, [psblas_cv_have_sendid=yes],[psblas_cv_have_sendid=no])
      AC_MSG_RESULT($psblas_cv_have_sendid)
      LIBS="$save_LIBS"
      AC_LANG([C])
      if test "x$psblas_cv_have_sendid" == "xyes";  then
        FDEFINES="$psblas_cv_define_prepend-DHAVE_KSENDID $FDEFINES"
      fi 
fi
])dnl


dnl @synopsis PAC_MAKE_IS_GNUMAKE
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
dnl
define(PAC_MAKE_IS_GNUMAKE,[
AC_MSG_CHECKING(for gnumake)
MAKE=${MAKE:-make}

if $MAKE --version 2>&1 | grep -e"GNU Make" >/dev/null; then 
    AC_MSG_RESULT(yes)
    psblas_make_gnumake='yes'
else
    AC_MSG_RESULT(no)
    psblas_make_gnumake='no'
fi
])dnl


dnl @synopsis PAC_BLAS([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl modified from ACX_BLAS([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl
dnl This macro looks for a library that implements the BLAS
dnl linear-algebra interface (see http://www.netlib.org/blas/). On
dnl success, it sets the BLAS_LIBS output variable to hold the
dnl requisite library linkages.
dnl
dnl To link with BLAS, you should link with:
dnl
dnl 	$BLAS_LIBS $LIBS $FLIBS
dnl
dnl in that order. FLIBS is the output variable of the
dnl AC_F77_LIBRARY_LDFLAGS macro (called if necessary by ACX_BLAS), and
dnl is sometimes necessary in order to link with F77 libraries. Users
dnl will also need to use AC_F77_DUMMY_MAIN (see the autoconf manual),
dnl for the same reason.
dnl
dnl Many libraries are searched for, from ATLAS to CXML to ESSL. The
dnl user may also use --with-blas=<lib> in order to use some specific
dnl BLAS library <lib>. In order to link successfully, however, be
dnl aware that you will probably need to use the same Fortran compiler
dnl (which can be set via the F77 env. var.) as was used to compile the
dnl BLAS library.
dnl
dnl ACTION-IF-FOUND is a list of shell commands to run if a BLAS
dnl library is found, and ACTION-IF-NOT-FOUND is a list of commands to
dnl run it if it is not found. If ACTION-IF-FOUND is not specified, the
dnl default action will define HAVE_BLAS.
dnl
dnl This macro requires autoconf 2.50 or later.
dnl
dnl @category InstalledPackages
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>
dnl @version 2001-12-13
dnl @license GPLWithACException
dnl modified by salvatore.filippone@uniroma2.it
dnl shifted check for ESSL as it was generating erroneous results on
dnl AIX SP5. 
dnl Modified with new name to handle Fortran compilers (such as NAG) 
dnl for which the linking MUST be done with the compiler (i.e.: 
dnl trying to link the Fortran version of the BLAS with the C compiler 
dnl would fail even when linking in the compiler's library)

AC_DEFUN([PAC_BLAS], [
AC_PREREQ(2.50)
AC_REQUIRE([AC_F77_LIBRARY_LDFLAGS])
pac_blas_ok=no

AC_ARG_WITH(blas,
	[AC_HELP_STRING([--with-blas=<lib>], [use BLAS library <lib>])])
case $with_blas in
	yes | "") ;;
	no) pac_blas_ok=disable ;;
	-* | */* | *.a | *.so | *.so.* | *.o) BLAS_LIBS="$with_blas" ;;
	*) BLAS_LIBS="-l$with_blas" ;;
esac

# Get fortran linker names of BLAS functions to check for.
AC_F77_FUNC(sgemm)
AC_F77_FUNC(dgemm)

pac_blas_save_LIBS="$LIBS"
LIBS="$LIBS $FLIBS"

# First, check BLAS_LIBS environment variable
if test $pac_blas_ok = no; then
if test "x$BLAS_LIBS" != x; then
	save_LIBS="$LIBS"; LIBS="$BLAS_LIBS $LIBS"
	AC_LANG([Fortran])
	AC_MSG_CHECKING([for sgemm in $BLAS_LIBS])
	AC_TRY_LINK_FUNC(sgemm, [pac_blas_ok=yes], [BLAS_LIBS=""])
	AC_MSG_RESULT($pac_blas_ok)
	AC_LANG([C])
	LIBS="$save_LIBS"
fi
fi


# BLAS in ATLAS library? (http://math-atlas.sourceforge.net/)
if test $pac_blas_ok = no; then
	AC_CHECK_LIB(atlas, ATL_xerbla,
		[AC_LANG([Fortran])
		 AC_CHECK_LIB(f77blas, sgemm,
		[AC_LANG([C])
		 AC_CHECK_LIB(cblas, cblas_dgemm,
			[pac_blas_ok=yes
			 BLAS_LIBS="-lcblas -lf77blas -latlas"],
			[], [-lf77blas -latlas])],
			[], [-latlas])])
	AC_LANG([C])

fi

# BLAS in PhiPACK libraries? (requires generic BLAS lib, too)
if test $pac_blas_ok = no; then
        AC_LANG([Fortran])
	AC_CHECK_LIB(blas, sgemm,
		[AC_CHECK_LIB(dgemm, dgemm,
		[AC_CHECK_LIB(sgemm, sgemm,
			[pac_blas_ok=yes; BLAS_LIBS="-lsgemm -ldgemm -lblas"],
			[], [-lblas])],
			[], [-lblas])])
        AC_LANG([C])
fi

# BLAS in Alpha CXML library? 
if test $pac_blas_ok = no; then
	AC_CHECK_LIB(cxml, $sgemm, [pac_blas_ok=yes;BLAS_LIBS="-lcxml"])
fi

# BLAS in Alpha DXML library? (now called CXML, see above)
if test $pac_blas_ok = no; then
	AC_CHECK_LIB(dxml, $sgemm, [pac_blas_ok=yes;BLAS_LIBS="-ldxml"])

fi

# BLAS in Sun Performance library?
if test $pac_blas_ok = no; then
	if test "x$GCC" != xyes; then # only works with Sun CC
		AC_CHECK_LIB(sunmath, acosp,
			[AC_CHECK_LIB(sunperf, $sgemm,
        			[BLAS_LIBS="-xlic_lib=sunperf -lsunmath"
                                 pac_blas_ok=yes],[],[-lsunmath])])

	fi
fi

# BLAS in SCSL library?  (SGI/Cray Scientific Library)
if test $pac_blas_ok = no; then
	AC_CHECK_LIB(scs, $sgemm, [pac_blas_ok=yes; BLAS_LIBS="-lscs"])
fi

# BLAS in SGIMATH library?
if test $pac_blas_ok = no; then
	AC_CHECK_LIB(complib.sgimath, $sgemm,
		     [pac_blas_ok=yes; BLAS_LIBS="-lcomplib.sgimath"])
fi

# BLAS in IBM ESSL library? (requires generic BLAS lib, too)
if test $pac_blas_ok = no; then
	AC_CHECK_LIB(blas, $sgemm,
		[AC_CHECK_LIB(essl, $sgemm,
			[pac_blas_ok=yes; BLAS_LIBS="-lessl -lblas"],
			[], [-lblas $FLIBS])])
fi
# BLAS linked to by default?  (happens on some supercomputers)
if test $pac_blas_ok = no; then
	save_LIBS="$LIBS"; LIBS="$LIBS"
	AC_TRY_LINK_FUNC($sgemm, [pac_blas_ok=yes], [BLAS_LIBS=""])
dnl	AC_CHECK_FUNC($sgemm, [pac_blas_ok=yes])
	LIBS="$save_LIBS"
fi

# Generic BLAS library?
if test $pac_blas_ok = no; then
  AC_LANG([Fortran])
  AC_CHECK_LIB(blas, sgemm, [pac_blas_ok=yes; BLAS_LIBS="-lblas"])
  AC_LANG([C])
  if test $pac_blas_ok = no; then
    AC_CHECK_LIB(blas, $sgemm, [pac_blas_ok=yes; BLAS_LIBS="-lblas"])
  fi
fi

AC_SUBST(BLAS_LIBS)

LIBS="$pac_blas_save_LIBS"

# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
if test x"$pac_blas_ok" = xyes; then
        ifelse([$1],,AC_DEFINE(HAVE_BLAS,1,[Define if you have a BLAS library.]),[$1])
        :
else
        pac_blas_ok=no
        $2
fi
])dnl PAC_BLAS


dnl @synopsis PAC_LAPACK([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl @synopsis ACX_LAPACK([ACTION-IF-FOUND[, ACTION-IF-NOT-FOUND]])
dnl
dnl This macro looks for a library that implements the LAPACK
dnl linear-algebra interface (see http://www.netlib.org/lapack/). On
dnl success, it sets the LAPACK_LIBS output variable to hold the
dnl requisite library linkages.
dnl
dnl To link with LAPACK, you should link with:
dnl
dnl     $LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS
dnl
dnl in that order. BLAS_LIBS is the output variable of the ACX_BLAS
dnl macro, called automatically. FLIBS is the output variable of the
dnl AC_F77_LIBRARY_LDFLAGS macro (called if necessary by ACX_BLAS), and
dnl is sometimes necessary in order to link with F77 libraries. Users
dnl will also need to use AC_F77_DUMMY_MAIN (see the autoconf manual),
dnl for the same reason.
dnl
dnl The user may also use --with-lapack=<lib> in order to use some
dnl specific LAPACK library <lib>. In order to link successfully,
dnl however, be aware that you will probably need to use the same
dnl Fortran compiler (which can be set via the F77 env. var.) as was
dnl used to compile the LAPACK and BLAS libraries.
dnl
dnl ACTION-IF-FOUND is a list of shell commands to run if a LAPACK
dnl library is found, and ACTION-IF-NOT-FOUND is a list of commands to
dnl run it if it is not found. If ACTION-IF-FOUND is not specified, the
dnl default action will define HAVE_LAPACK.
dnl
dnl @category InstalledPackages
dnl @author Steven G. Johnson <stevenj@alum.mit.edu>
dnl @version 2002-03-12
dnl @license GPLWithACException
dnl modified by salvatore.filippone@uniroma2.it
dnl shifted check for ESSL as it was generating erroneous results on
dnl AIX SP5. 
dnl Modified with new name to handle Fortran compilers (such as NAG) 
dnl for which the linking MUST be done with the compiler (i.e.: 
dnl trying to link the Fortran version of the BLAS with the C compiler 
dnl would fail even when linking in the compiler's library)

AC_DEFUN([PAC_LAPACK], [
AC_REQUIRE([PAC_BLAS])
pac_lapack_ok=no

AC_ARG_WITH(lapack,
        [AC_HELP_STRING([--with-lapack=<lib>], [use LAPACK library <lib>])])
case $with_lapack in
        yes | "") ;;
        no) pac_lapack_ok=disable ;;
        -* | */* | *.a | *.so | *.so.* | *.o) LAPACK_LIBS="$with_lapack" ;;
        *) LAPACK_LIBS="-l$with_lapack" ;;
esac

# Get fortran linker name of LAPACK function to check for.
AC_F77_FUNC(cheev)

# We cannot use LAPACK if BLAS is not found
if test "x$pac_blas_ok" != xyes; then
        pac_lapack_ok=noblas
fi

# First, check LAPACK_LIBS environment variable
if test "x$LAPACK_LIBS" != x; then
        save_LIBS="$LIBS"; LIBS="$LAPACK_LIBS $BLAS_LIBS $LIBS $FLIBS"
        AC_MSG_CHECKING([for cheev in $LAPACK_LIBS])
	AC_LANG([Fortran])
	dnl Warning : square brackets are EVIL!
	cat > conftest.$ac_ext <<EOF
        program test_cheev 
          call cheev
        end 
EOF
	if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
	  pac_lapack_ok=yes
	  AC_MSG_RESULT([yes])	
	else
	  AC_MSG_RESULT([no])	
	  echo "configure: failed program was:" >&AC_FD_CC
	  cat conftest.$ac_ext >&AC_FD_CC
	fi 
	rm -f conftest*
        LIBS="$save_LIBS"
        if test pac_lapack_ok = no; then
                LAPACK_LIBS=""
        fi
        AC_LANG([C])
fi

# LAPACK linked to by default?  (is sometimes included in BLAS lib)
if test $pac_lapack_ok = no; then
        save_LIBS="$LIBS"; LIBS="$LIBS $BLAS_LIBS $FLIBS"
        AC_MSG_CHECKING([for cheev in default libs])
	AC_LANG([Fortran])
	dnl Warning : square brackets are EVIL!
	cat > conftest.$ac_ext <<EOF
        program test_cheev 
          call cheev
        end 
EOF
	if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
	  pac_lapack_ok=yes
	  AC_MSG_RESULT([yes])	
	else
	  AC_MSG_RESULT([no])	
	  echo "configure: failed program was:" >&AC_FD_CC
	  cat conftest.$ac_ext >&AC_FD_CC
	fi 
	rm -f conftest*
        LIBS="$save_LIBS"
        AC_LANG([C])
fi

# Generic LAPACK library?
for lapack in lapack lapack_rs6k; do
        if test $pac_lapack_ok = no; then
                save_LIBS="$LIBS"; LIBS="$BLAS_LIBS $LIBS"
		AC_LANG([Fortran])
		AC_CHECK_LIB($lapack, cheev,
                    [pac_lapack_ok=yes; LAPACK_LIBS="-l$lapack"], [], [$FLIBS])
		AC_LANG([C])
                LIBS="$save_LIBS"
        fi
done

AC_SUBST(LAPACK_LIBS)

# Finally, execute ACTION-IF-FOUND/ACTION-IF-NOT-FOUND:
if test x"$pac_lapack_ok" = xyes; then
        ifelse([$1],,AC_DEFINE(HAVE_LAPACK,1,[Define if you have LAPACK library.]),[$1])
        :
else
        pac_lapack_ok=no
        $2
fi
])dnl PAC_LAPACK

dnl @synopsis PAC_FORTRAN_TEST_FLUSH( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program checking the FLUSH Fortran support.
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Michele Martone <michele.martone@uniroma2.it>
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
AC_DEFUN(PAC_FORTRAN_TEST_FLUSH,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FCFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for Fortran FLUSH statement])
i=0
while test \( -f tmpdir_$i \) -o \( -d tmpdir_$i \) ; do
  i=`expr $i + 1`
done
mkdir tmpdir_$i
cd tmpdir_$i
cat > conftest.$ac_ext <<EOF
program conftest
   integer :: iunit=10
   open(10)
   write(10,*) 'Test '
   flush(10)
   close(10)
end program conftest
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [
  $1])
else
  AC_MSG_RESULT([no])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  
  $2
])dnl
fi
cd ..
rm -fr tmpdir_$i])

dnl @synopsis PAC_FORTRAN_TEST_ISO_FORTRAN_ENV( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will determine if the fortran compiler MPIFC supports ISO_FORTRAN_ENV
dnl
dnl If yes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl 
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
dnl
AC_DEFUN(PAC_FORTRAN_TEST_ISO_FORTRAN_ENV,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for ISO_FORTRAN_ENV])
cat > conftest.$ac_ext <<EOF
           program test
             use iso_fortran_env
           end program test
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [rm -rf conftest*
  $1])
else
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
  AC_MSG_RESULT([no])
ifelse([$2], , , [  rm -rf conftest*
  $2
])dnl
fi
rm -f conftest*])

dnl @synopsis PAC_FORTRAN_TEST_MOLD( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program checking the MOLD=  Fortran support.
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
AC_DEFUN(PAC_FORTRAN_TEST_MOLD,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FCFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for Fortran MOLD= allocation])
i=0
while test \( -f tmpdir_$i \) -o \( -d tmpdir_$i \) ; do
  i=`expr $i + 1`
done
mkdir tmpdir_$i
cd tmpdir_$i
cat > conftest.$ac_ext <<EOF
program xtt
  type foo
    integer :: i
  end type foo
  type, extends(foo) :: new_foo
    integer :: j
  end type new_foo
  class(foo), allocatable  :: fooab
  type(new_foo) :: nfv 
  integer :: info

  allocate(fooab, mold=nfv, stat=info)

end program xtt
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [
  $1])
else
  AC_MSG_RESULT([no])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  
  $2
])dnl
fi
cd ..
rm -fr tmpdir_$i])


dnl @synopsis PAC_FORTRAN_TEST_SOURCE( [ACTION-IF-FOUND [, ACTION-IF-NOT-FOUND]])
dnl
dnl Will try to compile and link a program checking the SOURCE=  Fortran support.
dnl
dnl Will use MPIFC, otherwise '$FC'.
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
AC_DEFUN(PAC_FORTRAN_TEST_SOURCE,
ac_exeext=''
ac_ext='f90'
ac_link='${MPIFC-$FC} -o conftest${ac_exeext} $FCFLAGS $LDFLAGS conftest.$ac_ext $LIBS 1>&5'
dnl Warning : square brackets are EVIL!
[AC_MSG_CHECKING([support for Fortran SOURCE= allocation])
i=0
while test \( -f tmpdir_$i \) -o \( -d tmpdir_$i \) ; do
  i=`expr $i + 1`
done
mkdir tmpdir_$i
cd tmpdir_$i
cat > conftest.$ac_ext <<EOF
program xtt
  type foo
    integer :: i
  end type foo
  type, extends(foo) :: new_foo
    integer :: j
  end type new_foo
  class(foo), allocatable  :: fooab
  type(new_foo) :: nfv 
  integer :: info

  allocate(fooab, source=nfv, stat=info)

end program xtt
EOF
if AC_TRY_EVAL(ac_link) && test -s conftest${ac_exeext}; then
  AC_MSG_RESULT([yes])
  ifelse([$1], , :, [
  $1])
else
  AC_MSG_RESULT([no])
  echo "configure: failed program was:" >&AC_FD_CC
  cat conftest.$ac_ext >&AC_FD_CC
ifelse([$2], , , [  
  $2
])dnl
fi
cd ..
rm -fr tmpdir_$i])


dnl @synopsis PAC_CHECK_SPGPU
dnl
dnl Will try to find the spgpu library and headers.
dnl
dnl Will use $CC
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
dnl
AC_DEFUN(PAC_CHECK_SPGPU,
	 [SAVE_LIBS="$LIBS"
	  SAVE_CPPFLAGS="$CPPFLAGS"
	  PAC_CHECK_CUDA
	  AC_MSG_NOTICE([From CUDA: $pac_cv_have_cuda ])
	  if test "x$pac_cv_have_cuda" == "xyes"; then  
	  AC_ARG_WITH(spgpu, AC_HELP_STRING([--with-spgpu=DIR], [Specify the directory for SPGPU library and includes.]),
		      [psb_cv_spgpudir=$withval],
		      [psb_cv_spgpudir=''])
	  
	  AC_LANG([C])
	  if test "x$psb_cv_spgpudir" != "x"; then 
	  LIBS="-L$psb_cv_spgpudir/lib $LIBS"
	  GPU_INCLUDES="-I$psb_cv_spgpudir/include"
	  CPPFLAGS="$GPU_INCLUDES $CUDA_INCLUDES $CPPFLAGS"
	  GPU_LIBDIR="-L$psb_cv_spgpudir/lib"
	  fi
	  AC_MSG_CHECKING([spgpu dir $psb_cv_spgpudir])
	  AC_CHECK_HEADER([core.h],
			  [pac_gpu_header_ok=yes],
			  [pac_gpu_header_ok=no; GPU_INCLUDES=""])
	  
	  if test "x$pac_gpu_header_ok" == "xyes" ; then 
	  GPU_LIBS="-lspgpu $GPU_LIBDIR"
	  LIBS="$GPU_LIBS $CUDA_LIBS -lm $LIBS";
	  AC_MSG_CHECKING([for spgpuCreate in $GPU_LIBS])
	  AC_TRY_LINK_FUNC(spgpuCreate, 
			   [psb_cv_have_spgpu=yes;pac_gpu_lib_ok=yes; ],
			   [psb_cv_have_spgpu=no;pac_gpu_lib_ok=no; GPU_LIBS=""])
	  AC_MSG_RESULT($pac_gpu_lib_ok)
	  if test "x$psb_cv_have_spgpu" == "xyes" ; then 
	  AC_MSG_NOTICE([Have found SPGPU])
	  SPGPULIBNAME="libpsbgpu.a";
	  SPGPU_DIR="$psb_cv_spgpudir";
	  SPGPU_DEFINES="-DHAVE_SPGPU";
	  SPGPU_INCDIR="$SPGPU_DIR/include";
	  SPGPU_INCLUDES="-I$SPGPU_INCDIR";
	  SPGPU_LIBS="-lspgpu -L$SPGPU_DIR/lib";
	  CUDA_DIR="$psb_cv_cuda_dir";
	  CUDA_DEFINES="-DHAVE_CUDA";
	  CUDA_INCLUDES="-I$psb_cv_cuda_dir/include"
	  CUDA_LIBDIR="-L$psb_cv_cuda_dir/lib64 -L$psb_cv_cuda_dir/lib"
	  FDEFINES="$psblas_cv_define_prepend-DHAVE_SPGPU $psblas_cv_define_prepend-DHAVE_CUDA $FDEFINES";
	  CDEFINES="-DHAVE_SPGPU -DHAVE_CUDA $CDEFINES" ;
	  fi
  fi
fi
LIBS="$SAVE_LIBS"
CPPFLAGS="$SAVE_CPPFLAGS"
])dnl 




dnl @synopsis PAC_CHECK_CUDA
dnl
dnl Will try to find the cuda library and headers.
dnl
dnl Will use $CC
dnl
dnl If the test passes, will execute ACTION-IF-FOUND. Otherwise, ACTION-IF-NOT-FOUND.
dnl Note : This file will be likely to induce the compiler to create a module file
dnl (for a module called conftest).
dnl Depending on the compiler flags, this could cause a conftest.mod file to appear
dnl in the present directory, or in another, or with another name. So be warned!
dnl
dnl @author Salvatore Filippone <salvatore.filippone@uniroma2.it>
dnl
AC_DEFUN(PAC_CHECK_CUDA,
[AC_ARG_WITH(cuda, AC_HELP_STRING([--with-cuda=DIR], [Specify the directory for CUDA library and includes.]),
        [psb_cv_cuda_dir=$withval],
        [psb_cv_cuda_dir=''])

AC_LANG([C])
SAVE_LIBS="$LIBS"
SAVE_CPPFLAGS="$CPPFLAGS"
if test "x$psb_cv_cuda_dir" != "x"; then 
   CUDA_DIR="$psb_cv_cuda_dir"
   LIBS="-L$psb_cv_cuda_dir/lib $LIBS"
   CUDA_INCLUDES="-I$psb_cv_cuda_dir/include"
   CUDA_DEFINES="-DHAVE_CUDA"
   CPPFLAGS="$CUDA_INCLUDES $CPPFLAGS"
   CUDA_LIBDIR="-L$psb_cv_cuda_dir/lib64 -L$psb_cv_cuda_dir/lib"
fi
AC_MSG_CHECKING([cuda dir $psb_cv_cuda_dir])
AC_CHECK_HEADER([cuda_runtime.h],
 [pac_cuda_header_ok=yes],
 [pac_cuda_header_ok=no; CUDA_INCLUDES=""])

if test "x$pac_cuda_header_ok" == "xyes" ; then 
 CUDA_LIBS="-lcusparse -lcublas -lcudart $CUDA_LIBDIR"
 LIBS="$CUDA_LIBS -lm $LIBS";
 AC_MSG_CHECKING([for cudaMemcpy in $CUDA_LIBS])
 AC_TRY_LINK_FUNC(cudaMemcpy, 
		  [pac_cv_have_cuda=yes;pac_cuda_lib_ok=yes; ],
		  [pac_cv_have_cuda=no;pac_cuda_lib_ok=no; CUDA_LIBS=""])
 AC_MSG_RESULT($pac_cuda_lib_ok)
fi
LIBS="$SAVE_LIBS"
CPPFLAGS="$SAVE_CPPFLAGS"
])dnl 
