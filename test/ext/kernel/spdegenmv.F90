!    
!                Parallel Sparse BLAS  GPU plugin
!      (C) Copyright 2013
!                         Salvatore Filippone
!                         Alessandro Fanfarillo
!   
!    Redistribution and use in source and binary forms, with or without
!    modification, are permitted provided that the following conditions
!    are met:
!      1. Redistributions of source code must retain the above copyright
!         notice, this list of conditions and the following disclaimer.
!      2. Redistributions in binary form must reproduce the above copyright
!         notice, this list of conditions, and the following disclaimer in the
!         documentation and/or other materials provided with the distribution.
!      3. The name of the PSBLAS group or the names of its contributors may
!         not be used to endorse or promote products derived from this
!         software without specific written permission.
!   
!    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
!    ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
!    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
!    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE PSBLAS GROUP OR ITS CONTRIBUTORS
!    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
!    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
!    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
!    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
!    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
!    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
!    POSSIBILITY OF SUCH DAMAGE.
!   
!    
! File: spdegenmv.f90
!
! Program: pdegenmv
! This sample program measures the performance of the matrix-vector product.
! The matrix is generated in the same way as for the pdegen test case of
! the main PSBLAS library.
!
!
module psb_s_pde3d_mod
  use psb_base_mod, only : psb_spk_, psb_ipk_, psb_desc_type,&
       &  psb_sspmat_type, psb_s_vect_type, szero,&
       &  psb_s_base_sparse_mat, psb_s_base_vect_type, psb_i_base_vect_type

  interface 
    function s_func_3d(x,y,z) result(val)
      import :: psb_spk_
      real(psb_spk_), intent(in) :: x,y,z
      real(psb_spk_) :: val
    end function s_func_3d
  end interface 

  interface psb_gen_pde3d
    module procedure  psb_s_gen_pde3d
  end interface psb_gen_pde3d
  

contains

  function s_null_func_3d(x,y,z) result(val)

    real(psb_spk_), intent(in) :: x,y,z
    real(psb_spk_) :: val
    
    val = szero

  end function s_null_func_3d

!    
!
!  subroutine to allocate and fill in the coefficient matrix and
!  the rhs. 
!
  subroutine psb_s_gen_pde3d(ctxt,idim,a,bv,xv,desc_a,afmt,&
       & a1,a2,a3,b1,b2,b3,c,g,info,f,amold,vmold,imold,nrl)
    use psb_base_mod
    !
    !   Discretizes the partial differential equation
    ! 
    !   a1 dd(u)  a2 dd(u)    a3 dd(u)    b1 d(u)   b2 d(u)  b3 d(u)  
    ! -   ------ -  ------ -  ------ +  -----  +  ------  +  ------ + c u = f
    !      dxdx     dydy       dzdz        dx       dy         dz   
    !
    ! with Dirichlet boundary conditions
    !   u = g 
    !
    !  on the unit cube  0<=x,y,z<=1.
    !
    !
    ! Note that if b1=b2=b3=c=0., the PDE is the  Laplace equation.
    !
    implicit none
    procedure(s_func_3d)  :: b1,b2,b3,c,a1,a2,a3,g
    integer(psb_ipk_)     :: idim
    type(psb_sspmat_type) :: a
    type(psb_s_vect_type) :: xv,bv
    type(psb_desc_type)   :: desc_a
    type(psb_ctxt_type) :: ctxt
    integer(psb_ipk_)     :: info
    character(len=*)      :: afmt
    procedure(s_func_3d), optional :: f
    class(psb_s_base_sparse_mat), optional :: amold
    class(psb_s_base_vect_type), optional :: vmold
    class(psb_i_base_vect_type), optional :: imold
    integer(psb_ipk_), optional :: nrl

    ! Local variables.

    integer(psb_ipk_), parameter :: nb=20
    type(psb_s_csc_sparse_mat)  :: acsc
    type(psb_s_coo_sparse_mat)  :: acoo
    type(psb_s_csr_sparse_mat)  :: acsr
    real(psb_spk_)           :: zt(nb),x,y,z
    integer(psb_ipk_) :: m,n,nnz,glob_row,nlr,i,ii,ib,k
    integer(psb_ipk_) :: ix,iy,iz,ia,indx_owner
    integer(psb_ipk_) :: np, iam, nr, nt
    integer(psb_ipk_) :: icoeff
    integer(psb_ipk_), allocatable     :: irow(:),icol(:),myidx(:)
    real(psb_spk_), allocatable :: val(:)
    ! deltah dimension of each grid cell
    ! deltat discretization time
    real(psb_spk_)            :: deltah, sqdeltah, deltah2
    real(psb_spk_), parameter :: rhs=0.e0,one=1.e0,zero=0.e0
    real(psb_dpk_)    :: t0, t1, t2, t3, tasb, talc, ttot, tgen, tcdasb
    integer(psb_ipk_) :: err_act
    procedure(s_func_3d), pointer :: f_
    character(len=20)  :: name, ch_err,tmpfmt

    info = psb_success_
    name = 'create_matrix'
    call psb_erractionsave(err_act)

    call psb_info(ctxt, iam, np)


    if (present(f)) then 
      f_ => f
    else
      f_ => s_null_func_3d
    end if

    deltah   = 1.e0/(idim+2)
    sqdeltah = deltah*deltah
    deltah2  = 2.e0* deltah

    ! initialize array descriptor and sparse matrix storage. provide an
    ! estimate of the number of non zeroes 

    m   = idim*idim*idim
    n   = m
    nnz = ((n*9)/(np))
    if(iam == psb_root_) write(psb_out_unit,'("Generating Matrix (size=",i0,")...")')n

    if (present(nrl)) then 
      nr = nrl
    else
      !
      ! Using a simple BLOCK distribution.
      !
      nt = (m+np-1)/np
      nr = max(0,min(nt,m-(iam*nt)))
    end if

    nt = nr
    call psb_sum(ctxt,nt) 
    if (nt /= m) then 
      write(psb_err_unit,*) iam, 'Initialization error ',nr,nt,m
      info = -1
      call psb_barrier(ctxt)
      call psb_abort(ctxt)
      return    
    end if

    call psb_barrier(ctxt)
    t0 = psb_wtime()
    call psb_cdall(ctxt,desc_a,info,nl=nr)
    if (info == psb_success_) call psb_spall(a,desc_a,info,nnz=nnz)
    ! define  rhs from boundary conditions; also build initial guess 
    if (info == psb_success_) call psb_geall(xv,desc_a,info)
    if (info == psb_success_) call psb_geall(bv,desc_a,info)

    call psb_barrier(ctxt)
    talc = psb_wtime()-t0

    if (info /= psb_success_) then
      info=psb_err_from_subroutine_
      ch_err='allocation rout.'
      call psb_errpush(info,name,a_err=ch_err)
      goto 9999
    end if

    ! we build an auxiliary matrix consisting of one row at a
    ! time; just a small matrix. might be extended to generate 
    ! a bunch of rows per call. 
    ! 
    allocate(val(20*nb),irow(20*nb),&
         &icol(20*nb),stat=info)
    if (info /= psb_success_ ) then 
      info=psb_err_alloc_dealloc_
      call psb_errpush(info,name)
      goto 9999
    endif

    myidx = desc_a%get_global_indices()
    nlr = size(myidx)

    ! loop over rows belonging to current process in a block
    ! distribution.

    call psb_barrier(ctxt)
    t1 = psb_wtime()
    do ii=1, nlr,nb
      ib = min(nb,nlr-ii+1) 
      icoeff = 1
      do k=1,ib
        i=ii+k-1
        ! local matrix pointer 
        glob_row=myidx(i)
        ! compute gridpoint coordinates
        if (mod(glob_row,(idim*idim)) == 0) then
          ix = glob_row/(idim*idim)
        else
          ix = glob_row/(idim*idim)+1
        endif
        if (mod((glob_row-(ix-1)*idim*idim),idim) == 0) then
          iy = (glob_row-(ix-1)*idim*idim)/idim
        else
          iy = (glob_row-(ix-1)*idim*idim)/idim+1
        endif
        iz = glob_row-(ix-1)*idim*idim-(iy-1)*idim
        ! x, y, x coordinates
        x = (ix-1)*deltah
        y = (iy-1)*deltah
        z = (iz-1)*deltah
        zt(k) = f_(x,y,z)
        ! internal point: build discretization
        !   
        !  term depending on   (x-1,y,z)
        !
        val(icoeff) = -a1(x,y,z)/sqdeltah-b1(x,y,z)/deltah2
        if (ix == 1) then 
          zt(k) = g(szero,y,z)*(-val(icoeff)) + zt(k)
        else
          icol(icoeff) = (ix-2)*idim*idim+(iy-1)*idim+(iz)
          irow(icoeff) = glob_row
          icoeff       = icoeff+1
        endif
        !  term depending on     (x,y-1,z)
        val(icoeff)  = -a2(x,y,z)/sqdeltah-b2(x,y,z)/deltah2
        if (iy == 1) then 
          zt(k) = g(x,szero,z)*(-val(icoeff))   + zt(k)
        else
          icol(icoeff) = (ix-1)*idim*idim+(iy-2)*idim+(iz)
          irow(icoeff) = glob_row
          icoeff       = icoeff+1
        endif
        !  term depending on     (x,y,z-1)
        val(icoeff)=-a3(x,y,z)/sqdeltah-b3(x,y,z)/deltah2
        if (iz == 1) then 
          zt(k) = g(x,y,szero)*(-val(icoeff))   + zt(k)
        else
          icol(icoeff) = (ix-1)*idim*idim+(iy-1)*idim+(iz-1)
          irow(icoeff) = glob_row
          icoeff       = icoeff+1
        endif

        !  term depending on     (x,y,z)
        val(icoeff)=2.e0*(a1(x,y,z)+a2(x,y,z)+a3(x,y,z))/sqdeltah &
             & + c(x,y,z)
        icol(icoeff) = (ix-1)*idim*idim+(iy-1)*idim+(iz)
        irow(icoeff) = glob_row
        icoeff       = icoeff+1                  
        !  term depending on     (x,y,z+1)
        val(icoeff)=-a3(x,y,z)/sqdeltah+b3(x,y,z)/deltah2
        if (iz == idim) then 
          zt(k) = g(x,y,sone)*(-val(icoeff))   + zt(k)
        else
          icol(icoeff) = (ix-1)*idim*idim+(iy-1)*idim+(iz+1)
          irow(icoeff) = glob_row
          icoeff       = icoeff+1
        endif
        !  term depending on     (x,y+1,z)
        val(icoeff)=-a2(x,y,z)/sqdeltah+b2(x,y,z)/deltah2
        if (iy == idim) then 
          zt(k) = g(x,sone,z)*(-val(icoeff))   + zt(k)
        else
          icol(icoeff) = (ix-1)*idim*idim+(iy)*idim+(iz)
          irow(icoeff) = glob_row
          icoeff       = icoeff+1
        endif
        !  term depending on     (x+1,y,z)
        val(icoeff)=-a1(x,y,z)/sqdeltah+b1(x,y,z)/deltah2
        if (ix==idim) then 
          zt(k) = g(sone,y,z)*(-val(icoeff))   + zt(k)
        else
          icol(icoeff) = (ix)*idim*idim+(iy-1)*idim+(iz)
          irow(icoeff) = glob_row
          icoeff       = icoeff+1
        endif

      end do
      call psb_spins(icoeff-1,irow,icol,val,a,desc_a,info)
      if(info /= psb_success_) exit
      call psb_geins(ib,myidx(ii:ii+ib-1),zt(1:ib),bv,desc_a,info)
      if(info /= psb_success_) exit
      zt(:)=0.e0
      call psb_geins(ib,myidx(ii:ii+ib-1),zt(1:ib),xv,desc_a,info)
      if(info /= psb_success_) exit
    end do

    tgen = psb_wtime()-t1
    if(info /= psb_success_) then
      info=psb_err_from_subroutine_
      ch_err='insert rout.'
      call psb_errpush(info,name,a_err=ch_err)
      goto 9999
    end if

    deallocate(val,irow,icol)

    call psb_barrier(ctxt)
    t1 = psb_wtime()
    call psb_cdasb(desc_a,info,mold=imold)
    tcdasb = psb_wtime()-t1
    call psb_barrier(ctxt)
    t1 = psb_wtime()
    if (info == psb_success_) then 
      if (present(amold)) then 
        call psb_spasb(a,desc_a,info,dupl=psb_dupl_err_,mold=amold)
      else
        call psb_spasb(a,desc_a,info,dupl=psb_dupl_err_,afmt=afmt)
      end if
    end if
    call psb_barrier(ctxt)
    if(info /= psb_success_) then
      info=psb_err_from_subroutine_
      ch_err='asb rout.'
      call psb_errpush(info,name,a_err=ch_err)
      goto 9999
    end if
    if (info == psb_success_) call psb_geasb(xv,desc_a,info,mold=vmold)
    if (info == psb_success_) call psb_geasb(bv,desc_a,info,mold=vmold)
    if(info /= psb_success_) then
      info=psb_err_from_subroutine_
      ch_err='asb rout.'
      call psb_errpush(info,name,a_err=ch_err)
      goto 9999
    end if
    tasb = psb_wtime()-t1
    call psb_barrier(ctxt)
    ttot = psb_wtime() - t0 

    call psb_amx(ctxt,talc)
    call psb_amx(ctxt,tgen)
    call psb_amx(ctxt,tasb)
    call psb_amx(ctxt,ttot)
    if(iam == psb_root_) then
      tmpfmt = a%get_fmt()
      write(psb_out_unit,'("The matrix has been generated and assembled in ",a3," format.")')&
           &   tmpfmt
      write(psb_out_unit,'("-allocation  time : ",es12.5)') talc
      write(psb_out_unit,'("-coeff. gen. time : ",es12.5)') tgen
      write(psb_out_unit,'("-desc asbly  time : ",es12.5)') tcdasb
      write(psb_out_unit,'("- mat asbly  time : ",es12.5)') tasb
      write(psb_out_unit,'("-total       time : ",es12.5)') ttot

    end if
    call psb_erractionrestore(err_act)
    return

9999 call psb_error_handler(ctxt,err_act)

    return
  end subroutine psb_s_gen_pde3d


end module psb_s_pde3d_mod

program pdgenmv
  use psb_base_mod
  use psb_util_mod 
  use psb_ext_mod
  use psb_s_pde3d_mod
  implicit none

  ! input parameters
  character(len=5)  :: acfmt
  integer   :: idim

  ! miscellaneous 
  real(psb_spk_), parameter :: one = 1.e0
  real(psb_dpk_) :: t1, t2, tprec, tcnv, flops, tflops, tt1, tt2,  bdwdth

  ! sparse matrix and preconditioner
  type(psb_sspmat_type) :: a
  ! descriptor
  type(psb_desc_type)   :: desc_a
  ! dense matrices
  type(psb_s_vect_type)  :: xv,bv 

  real(psb_spk_), allocatable :: xc1(:),xc2(:)
  ! blacs parameters
  type(psb_ctxt_type) :: ctxt
  integer(psb_ipk_)   :: iam, np

  ! solver parameters
  integer(psb_epk_) :: amatsize, precsize, descsize, annz, nbytes
  real(psb_spk_)   :: err, eps
  integer, parameter :: ntests=200
  type(psb_s_coo_sparse_mat), target   :: acoo
  type(psb_s_csr_sparse_mat), target   :: acsr
  type(psb_s_csc_sparse_mat), target   :: acsc
  type(psb_s_ell_sparse_mat), target   :: aell
  type(psb_s_hll_sparse_mat), target   :: ahll
  type(psb_s_dia_sparse_mat), target   :: adia
  type(psb_s_dns_sparse_mat), target   :: adns
  type(psb_s_hdia_sparse_mat), target   :: ahdia

  class(psb_s_base_sparse_mat), pointer :: acmold
  ! other variables
  logical, parameter :: dump=.false.
  integer            :: info, i, nr
  character(len=20)  :: name,ch_err
  character(len=40)  :: fname

  info=psb_success_

  
  call psb_init(ctxt)
  call psb_info(ctxt,iam,np)

  if (iam < 0) then 
    ! This should not happen, but just in case
    call psb_exit(ctxt)
    stop
  endif
  if(psb_get_errstatus() /= 0) goto 9999
  name='pde90'
  call psb_set_errverbosity(2)
  !
  ! Hello world
  !
  if (iam == psb_root_) then 
    write(*,*) 'Welcome to PSBLAS version: ',psb_version_string_
    write(*,*) 'This is the ',trim(name),' sample program'
  end if
  !
  !  get parameters
  !
  call get_parms(ctxt,acfmt,idim)

  !
  !  allocate and fill in the coefficient matrix and initial vectors
  !
  call psb_barrier(ctxt)
  t1 = psb_wtime()
  call psb_gen_pde3d(ctxt,idim,a,bv,xv,desc_a,'CSR  ',&
       & a1,a2,a3,b1,b2,b3,c,g,info)  
  call psb_barrier(ctxt)
  t2 = psb_wtime() - t1
  if(info /= psb_success_) then
    info=psb_err_from_subroutine_
    ch_err='create_matrix'
    call psb_errpush(info,name,a_err=ch_err)
    goto 9999
  end if
  if (iam == psb_root_) write(psb_out_unit,'("Overall matrix creation time : ",es12.5)')t2
  if (iam == psb_root_) write(psb_out_unit,'(" ")')

  if (dump) then 
    write(fname,'(a,i3.3,a)') 'pde',idim,'.mtx'
    call a%print(fname,head='PDEGEN test matrix')
  end if

  select case(psb_toupper(acfmt))
  case('ELL')
    acmold => aell
  case('HLL')
    acmold => ahll
  case('DIA')
    acmold => adia
  case('DNS')
    acmold => adns
  case('HDIA')
    acmold => ahdia
  case('CSR')
    acmold => acsr
  case('CSC')
    acmold => acsc
  case('COO')
    acmold => acoo
  case default
    write(*,*) 'Unknown format defaulting to HLL'
    acmold => ahll
  end select
  call a%cscnv(info,mold=acoo)
  
  call psb_barrier(ctxt)
  t1 = psb_wtime()
  call a%cscnv(info,mold=acmold)
  tcnv = psb_wtime()-t1
  call psb_amx(ctxt,tcnv)

  if ((info /= 0).or.(psb_get_errstatus()/=0)) then 
    write(0,*) 'From cscnv ',info
    call psb_error()
    stop
  end if

  call xv%set(sone)
  nr       = desc_a%get_local_rows() 

  call psb_barrier(ctxt)
  t1 = psb_wtime()
  do i=1,ntests 
    call psb_spmm(sone,a,xv,szero,bv,desc_a,info)
  end do
  call psb_barrier(ctxt)
  t2 = psb_wtime() - t1
  call psb_amx(ctxt,t2)

  annz     = a%get_nzeros()
  amatsize = a%sizeof()
  descsize = psb_sizeof(desc_a)
  call psb_sum(ctxt,annz)
  call psb_sum(ctxt,amatsize)
  call psb_sum(ctxt,descsize)

  if (iam == psb_root_) then
    write(psb_out_unit,&
         & '("Matrix: ell1 ",i0)') idim
    write(psb_out_unit,&
         &'("Test on                          : ",i20," processors")') np
    write(psb_out_unit,&
         &'("Size of matrix                   : ",i20,"           ")') nr
    write(psb_out_unit,&
         &'("Number of nonzeros               : ",i20,"           ")') annz
    write(psb_out_unit,&
         &'("Memory occupation                : ",i20,"           ")') amatsize
    flops  = ntests*(2.d0*annz)
    tflops = flops
    flops  = flops / (t2)
    tflops = tflops / (tt2)
    write(psb_out_unit,'("Storage type for    A: ",a)') a%get_fmt()
    write(psb_out_unit,'("Conversion time              (CPU)   : ",F20.3)')&
         & tcnv
    write(psb_out_unit,&
         & '("Number of flops (",i6," prod)        : ",F20.0,"           ")') &
         &  ntests,flops
    write(psb_out_unit,'("Time for ",i6," products (s) (CPU)   : ",F20.3)')&
         &  ntests,t2
    write(psb_out_unit,'("Time per product    (ms)     (CPU)   : ",F20.3)')&
         & t2*1.d3/(1.d0*ntests)
    write(psb_out_unit,'("MFLOPS                       (CPU)   : ",F20.3)')&
         & flops/1.d6
    !
    ! This computation assumes the data movement associated with CSR:
    ! it is minimal in terms of coefficients. Other formats may either move
    ! more data (padding etc.) or less data (if they can save on the indices). 
    !
    nbytes = nr*(2*psb_sizeof_sp + psb_sizeof_ip)+&
         & annz*(psb_sizeof_sp + psb_sizeof_ip)
    bdwdth = ntests*nbytes/(t2*1.d6)
    write(psb_out_unit,*)
    write(psb_out_unit,'("MBYTES/S                  (CPU)  : ",F20.3)') bdwdth
    write(psb_out_unit,'("Storage type for DESC_A: ",a)') desc_a%indxmap%get_fmt()
    write(psb_out_unit,'("Total memory occupation for DESC_A: ",i12)')descsize

  end if

  !  
  !  cleanup storage and exit
  !
  call psb_gefree(bv,desc_a,info)
  call psb_gefree(xv,desc_a,info)
  call psb_spfree(a,desc_a,info)
  call psb_cdfree(desc_a,info)
  if(info /= psb_success_) then
    info=psb_err_from_subroutine_
    ch_err='free routine'
    call psb_errpush(info,name,a_err=ch_err)
    goto 9999
  end if

  call psb_exit(ctxt)
  stop

9999 continue

  call psb_error(ctxt)

contains
  !
  ! get iteration parameters from standard input
  !
  subroutine  get_parms(ctxt,acfmt,idim)
    type(psb_ctxt_type) :: ctxt
    character(len=*)    :: acfmt
    integer      :: idim
    integer      :: np, iam
    integer      :: intbuf(10), ip

    call psb_info(ctxt, iam, np)

    if (iam == 0) then
      read(psb_inp_unit,*) acfmt
      read(psb_inp_unit,*) idim
    endif
    call psb_bcast(ctxt,acfmt)
    call psb_bcast(ctxt,idim)

    if (iam == 0) then
      write(psb_out_unit,'("Testing matrix       : ell1")')      
      write(psb_out_unit,'("Grid dimensions      : ",i4,"x",i4,"x",i4)')idim,idim,idim
      write(psb_out_unit,'("Number of processors : ",i0)')np
      write(psb_out_unit,'("Data distribution    : BLOCK")')
      write(psb_out_unit,'(" ")')
    end if
    return

  end subroutine get_parms

  !
  ! functions parametrizing the differential equation 
  !  
  function a1(x,y,z)
    use psb_base_mod, only : psb_spk_
    real(psb_spk_) :: a1
    real(psb_spk_), intent(in) :: x,y,z
    a1=1.e0
  end function a1
  function a2(x,y,z)
    use psb_base_mod, only : psb_spk_
    real(psb_spk_) ::  a2
    real(psb_spk_), intent(in) :: x,y,z
    a2=2.e1*y
  end function a2
  function a3(x,y,z)
    use psb_base_mod, only : psb_spk_
    real(psb_spk_) ::  a3
    real(psb_spk_), intent(in) :: x,y,z      
    a3=1.e0
  end function a3
  function c(x,y,z)
    use psb_base_mod, only : psb_spk_
    real(psb_spk_) :: c
    real(psb_spk_), intent(in) :: x,y,z      
    c=1.e0
  end function c
  function b1(x,y,z)
    use psb_base_mod, only : psb_spk_
    real(psb_spk_) ::  b1   
    real(psb_spk_), intent(in) :: x,y,z
    b1=1.e0
  end function b1
  function b2(x,y,z)
    use psb_base_mod, only : psb_spk_
    real(psb_spk_) ::  b2
    real(psb_spk_), intent(in) :: x,y,z
    b2=1.e0
  end function b2
  function b3(x,y,z)
    use psb_base_mod, only : psb_spk_
    real(psb_spk_) ::  b3
    real(psb_spk_), intent(in) :: x,y,z
    b3=1.e0
  end function b3
  function g(x,y,z)
    use psb_base_mod, only : psb_spk_, sone, szero
    real(psb_spk_) ::  g
    real(psb_spk_), intent(in) :: x,y,z
    g = szero
    if (x == sone) then
      g = sone
    else if (x == szero) then 
      g = exp(y**2-z**2)
    end if
  end function g

end program pdgenmv
