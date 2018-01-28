!    
!                Parallel Sparse BLAS  version 2.3.1
!      (C) Copyright 2006, 2007, 2008, 2009, 2010
!                         Salvatore Filippone
!                         Alfredo Buttari
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
! File: ppde.f90
!
! Program: ppde
! This sample program solves a linear system obtained by discretizing a
! PDE with Dirichlet BCs. 
! 
!
! The PDE is a general second order equation in 3d
!
!   a1 dd(u)  a2 dd(u)    a3 dd(u)    b1 d(u)   b2 d(u)  b3 d(u)  
! -   ------ -  ------ -  ------  +  -----  +  ------  +  ------ + c u  = f
!      dxdx     dydy       dzdz        dx       dy         dz   
!
! with Dirichlet boundary conditions
!   u = g 
!
!  on the unit cube  0<=x,y,z<=1.
!
! In this sample program the index space of the discretized
! computational domain is first numbered sequentially in a standard way, 
! then the corresponding vector is distributed according to a BLOCK
! data distribution.
!
! Boundary conditions are set in a very simple way, by adding 
! equations of the form
!
!   u(x,y) = exp(-x^2-y^2-z^2)
!
! Note that if a1=a2=a3=a4=0., the PDE is the well-known Laplace equation.
!
program mldpde2d
  use psb_base_mod
  use mld_prec_mod
  use psb_krylov_mod
  use psb_util_mod
  use psb_d_gpu_vect_mod
  use psb_d_elg_mat_mod
  use psb_d_hll_mat_mod
  use mld_d_invt_solver
  use mld_d_invk_solver
  use mld_d_ainv_solver
  use pde2d_exp_mod
!!$  use pde2d_base_mod
  implicit none

  ! input parameters
  character(len=20) :: kmethd, ptype, renum
  character(len=5)  :: afmt
  integer   :: idim

  ! miscellaneous 
  real(psb_dpk_), parameter :: one = 1.d0
  real(psb_dpk_) :: t1, t2, tprec, gt1,gt2 


  ! sparse matrix and preconditioner
  type(psb_dspmat_type) :: a,agpu
  type(mld_dprec_type)  :: prec, precg
  type(mld_d_invt_solver_type) :: invtsv
  type(mld_d_invk_solver_type) :: invksv
  type(mld_d_ainv_solver_type) :: ainvsv
  type ainvparms 
    character(len=12) :: alg
    integer           :: fill, inv_fill, orth_alg
    real(psb_dpk_)    :: thresh, inv_thresh
  end type ainvparms
  type(ainvparms)     :: parms
  ! descriptor
  type(psb_desc_type)   :: desc_a, desc_b
  ! dense matrices
  real(psb_dpk_), allocatable :: b(:), x(:), d(:)
  type(psb_d_vect_type)      :: xv,bv, vtst, xg, bg 
  type(psb_d_vect_gpu)       :: vgpumold
  type(psb_d_elg_sparse_mat) :: aelg
  type(psb_d_ell_sparse_mat) :: aell
  type(psb_d_hll_sparse_mat) :: ahll
  type(psb_d_csr_sparse_mat) :: acsr
  ! blacs parameters
  integer            :: ictxt, iam, np

  ! solver parameters
  integer            :: iter, itmax,itrace, istopc, irst,giter, nr
  integer(psb_long_int_k_) :: amatsize, precsize, descsize, d2size, precnz,amatnz
  real(psb_dpk_)   :: err, eps, gerr, amxval
  integer, allocatable :: perm(:)

  ! other variables
  integer            :: info, i
  character(len=20)  :: name,ch_err
  character(len=40)  :: fname

  info=psb_success_

  
  call psb_init(ictxt)
  call psb_info(ictxt,iam,np)

  if (iam < 0) then 
    ! This should not happen, but just in case
    call psb_exit(ictxt)
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
  call set_parm(5.d-3)
  !
  !  get parameters
  !
  call get_parms(ictxt,kmethd,ptype,afmt,renum,idim,istopc,itmax,itrace,irst,parms)

  !
  !  allocate and fill in the coefficient matrix, rhs and initial guess 
  !
  call psb_barrier(ictxt)
  t1 = psb_wtime()
  call psb_gen_pde2d(ictxt,idim,a,bv,xv,desc_a,afmt,&
       & a1,a2,b1,b2,c,g,info)  
  call psb_barrier(ictxt)
  t2 = psb_wtime() - t1
  if(info /= psb_success_) then
    info=psb_err_from_subroutine_
    ch_err='create_matrix'
    call psb_errpush(info,name,a_err=ch_err)
    goto 9999
  end if
  b=bv%get_vect()
  x=xv%get_vect()
  info = psb_get_errstatus()
  if (info /= 0) call psb_error(ictxt)
  if (np == 1) then 
    nr = a%get_nrows()
    call psb_mat_renum(renum,a,info,perm=perm)
    if (info /= 0) then 
      write(0,*) 'Error from RENUM',info
      goto 9999
    end if
    if (allocated(perm)) then
!!$    write(0,*) size(perm),':',nr,':',perm(:),b(:)
    else
      write(0,*) allocated(perm)
      goto 9999
    end if
    call psb_gelp('N',perm(1:nr),b(1:nr),info)
  end if
  do i=1,size(x)
    x(i) = i
  end do
  call xv%bld(x)
  call psb_spmm(done,a,xv,dzero,bv,desc_a,info)
  call xv%set(dzero)
  x=dzero
  b=bv%get_vect()
  
!!$  write(0,*) size(perm),':',nr,':',perm(:),b(:)
  info = psb_get_errstatus()
  if (info /= 0) call psb_error(ictxt)
  call bg%bld(b,mold=vgpumold)
  call xg%bld(x,mold=vgpumold)
  if (info == 0) call a%cscnv(agpu,info,mold=aelg)
  if (info == 0) call a%cscnv(info,mold=ahll)
  if (info /= 0) write(0,*) 'From CLONE: ',info
!!$  call a%print('acsr.mtx')
!!$  call agpu%print('agpu.mtx')
  info = psb_get_errstatus()
  if (info /= 0) call psb_error(ictxt)
  if (iam == psb_root_) write(psb_out_unit,'("Overall matrix creation time : ",es12.5)')t2
  if (iam == psb_root_) write(psb_out_unit,'(" ")')
!!$  write(fname,'(a,i2.2,a,i2.2,a)') 'amat-',iam,'-',np,'.mtx'
!!$  call a%print(fname)
!!$  call psb_cdprt(20+iam,desc_a,short=.false.)
!!$  call psb_cdcpy(desc_a,desc_b,info)
!!$  call psb_set_debug_level(9999)
!!$
!!$  call psb_cdbldext(a,desc_a,2,desc_b,info,extype=psb_ovt_asov_)
!!$  if (info /= 0) then 
!!$    write(0,*) 'Error from bldext'
!!$    call psb_abort(ictxt)
!!$  end if
  !
  !  prepare the preconditioner.
  !  
  if(iam == psb_root_) write(psb_out_unit,'("Setting preconditioner to : ",a)')ptype
  if (psb_toupper(trim(ptype)) == "AINV") then 
    call mld_precinit(prec,"BJAC",info)
  else
    call mld_precinit(prec,ptype,info)
  end if

  info = psb_get_errstatus()
  if (info /= 0) call psb_error(ictxt)

  if (psb_toupper(ptype) == 'DIAG') then 
    call mld_precset(prec,mld_smoother_sweeps_,1,info)
  end if
  if (psb_toupper(ptype) == 'AINV') then 
    select case (psb_toupper(parms%alg)) 
    case ('INVK') 
      call mld_inner_precset(prec,invksv,info) 
    case ('INVT') 
      call mld_inner_precset(prec,invtsv,info) 
    case ('AINV') 
      call mld_inner_precset(prec,ainvsv,info) 
    end select
    call mld_precset(prec,mld_ainv_alg_,  parms%orth_alg,  info)
    call mld_precset(prec,mld_sub_fillin_,  parms%fill,    info)
    call mld_precset(prec,mld_sub_iluthrs_, parms%thresh,  info)
    call mld_precset(prec,mld_inv_fillin_, parms%inv_fill, info)
    call mld_precset(prec,mld_inv_thresh_, parms%inv_thresh, info)
  end if
  info = psb_get_errstatus()
  if (info /= 0) call psb_error(ictxt)

  call psb_barrier(ictxt)
  t1 = psb_wtime()
  call mld_precbld(a,desc_a,prec,info,amold=ahll)
  if(info /= psb_success_) then
    info=psb_err_from_subroutine_
    ch_err='psb_precbld'
    call psb_errpush(info,name,a_err=ch_err)
    goto 9999
  end if

  if (psb_toupper(trim(ptype)) == "AINV") then 
    call mld_precinit(precg,"BJAC",info)
  else
    call mld_precinit(precg,ptype,info)
  end if

  info = psb_get_errstatus()
  if (info /= 0) call psb_error(ictxt)

  if (psb_toupper(ptype) == 'DIAG') then 
    call mld_precset(precg,mld_smoother_sweeps_,1,info)
  end if
  if (psb_toupper(ptype) == 'AINV') then 
    select case (psb_toupper(parms%alg)) 
    case ('INVK') 
      call mld_inner_precset(precg,invksv,info) 
    case ('INVT') 
      call mld_inner_precset(precg,invtsv,info) 
    case ('AINV') 
      call mld_inner_precset(precg,ainvsv,info) 
    end select
    call mld_precset(precg,mld_ainv_alg_,  parms%orth_alg,  info)
    call mld_precset(precg,mld_sub_fillin_,  parms%fill,    info)
    call mld_precset(precg,mld_sub_iluthrs_, parms%thresh,  info)
    call mld_precset(precg,mld_inv_fillin_, parms%inv_fill, info)
    call mld_precset(precg,mld_inv_thresh_, parms%inv_thresh, info)
  end if
  info = psb_get_errstatus()
  if (info /= 0) call psb_error(ictxt)

  call psb_barrier(ictxt)
  t1 = psb_wtime()
  call mld_precbld(a,desc_a,precg,info,amold=aelg,vmold=vgpumold)
  if(info /= psb_success_) then
    info=psb_err_from_subroutine_
    ch_err='psb_precbld'
    call psb_errpush(info,name,a_err=ch_err)
    goto 9999
  end if

  tprec = psb_wtime()-t1

  call psb_amx(ictxt,tprec)

  if (iam == psb_root_) write(psb_out_unit,'("Preconditioner time : ",es12.5)')tprec
  if (iam == psb_root_) write(psb_out_unit,'(" ")')
  amatsize = psb_sizeof(a)
  descsize = psb_sizeof(desc_a)
  precsize = mld_sizeof(prec)
  amatnz   = a%get_nzeros()
  precnz   = prec%get_nzeros()
  call psb_sum(ictxt,amatsize)
  call psb_sum(ictxt,descsize)
  call psb_sum(ictxt,precsize)
  call psb_sum(ictxt,amatnz)
  call psb_sum(ictxt,precnz)
  if (iam == psb_root_) then
    write(psb_out_unit,'(" ")')
    write(psb_out_unit,'("Renumbering algorithm  :      ",a)') trim(renum)
    write(psb_out_unit,'("Total nonzeros          for A:      ",i12)')amatnz
    write(psb_out_unit,'("Total nonzeros          for PREC:   ",i12)')precnz      

    write(psb_out_unit,'("Total memory occupation for A:      ",i12)')amatsize
    write(psb_out_unit,'("Total memory occupation for PREC:   ",i12)')precsize    

    write(psb_out_unit,'("Total memory occupation for DESC_A: ",i12)')descsize
    write(psb_out_unit,'("Storage type for DESC_A: ",a)') desc_a%indxmap%get_fmt()
!!$    write(psb_out_unit,'("Storage type for DESC_B: ",a)') desc_b%indxmap%get_fmt()
    write(psb_out_unit,'(" ")')
    call mld_precdescr(prec,info) 
  end if


  !
  ! iterative method parameters 
  !
  if(iam == psb_root_) write(psb_out_unit,'("Calling GPU iterative method ",a)')kmethd
  call psb_barrier(ictxt)
  gt1 = psb_wtime()  
  eps   = 1.d-7
  !call psb_set_debug_level(psb_debug_ext_)
  call psb_krylov(kmethd,agpu,precg,bg,xg,eps,desc_a,info,& 
       & itmax=itmax,iter=giter,err=gerr,itrace=itrace,istop=istopc,irst=irst)     
  call psb_set_debug_level(0)

  if(info /= psb_success_) then
    info=psb_err_from_subroutine_
    ch_err='solver routine'
    call psb_errpush(info,name,a_err=ch_err)
    goto 9999
  end if

  if(iam == psb_root_) write(psb_out_unit,'("Calling CPU iterative method ",a)')kmethd
  call psb_barrier(ictxt)
  gt2 = psb_wtime() - gt1
  call psb_amx(ictxt,t2)
  call psb_barrier(ictxt)
  !call psb_set_debug_level(psb_debug_ext_)
  t1 = psb_wtime()  
  call psb_krylov(kmethd,a,prec,bv,xv,eps,desc_a,info,& 
       & itmax=itmax,iter=iter,err=err,itrace=itrace,istop=istopc,irst=irst)     
  call psb_barrier(ictxt)
  t2 = psb_wtime() - t1
  call psb_set_debug_level(0)
  call psb_amx(ictxt,t2)
  call psb_barrier(ictxt)

  if (iam == psb_root_) then
    write(psb_out_unit,'(" ")')
    write(psb_out_unit,'("Time to solve matrix  GPU     : ",es12.5)')gt2
    write(psb_out_unit,'("Time per iteration    GPU     : ",es12.5)')gt2/giter
    write(psb_out_unit,'("Number of iterations  GPU     : ",i0)')giter
    write(psb_out_unit,'("Convergence indicator on exit : ",es12.5)')gerr
    write(psb_out_unit,'("Time to solve matrix  CPU     : ",es12.5)')t2
    write(psb_out_unit,'("Time per iteration    CPU     : ",es12.5)')t2/iter
    write(psb_out_unit,'("Number of iterations          : ",i0)')iter
    write(psb_out_unit,'("Convergence indicator on exit : ",es12.5)')err
    write(psb_out_unit,'("Info  on exit                 : ",i0)')info
    write(psb_out_unit,'("Speedup of time per iteration : ",es12.5)')(t2/iter)/(gt2/giter)
  end if

  if (.false.) then 
    write(fname,'(a,i0,a)'),'pde-',idim,'.mtx'
    call a%print(fname,head="PDE 3D")
    ! call hb_write(a,info,filename=fname,key='PSBPDE',rhs=b,mtitle="PDEGEN test matrix")
    call prec%dump(info,prefix='test-c',istart=1,&
         & ac=.false.,solver=.true.,smoother=.true.)
    call precg%dump(info,prefix='test-g',istart=1,&
         & ac=.false.,solver=.true.,smoother=.true.)
    if (info /= 0) then 
      write(0,*) 'Error from prec%dump ', info
      goto  9999
    end if
  end if

  !  
  !  cleanup storage and exit
  !
  call psb_gefree(bv,desc_a,info)
  call psb_gefree(xv,desc_a,info)
  call psb_spfree(a,desc_a,info)
  call mld_precfree(prec,info)
  call psb_cdfree(desc_a,info)
  if(info /= psb_success_) then
    info=psb_err_from_subroutine_
    ch_err='free routine'
    call psb_errpush(info,name,a_err=ch_err)
    goto 9999
  end if

9999 continue
  if(info /= psb_success_) then
    call psb_error(ictxt)
  end if
  call psb_exit(ictxt)
  stop

contains
  !
  ! get iteration parameters from standard input
  !
  subroutine  get_parms(ictxt,kmethd,ptype,afmt,renum,idim,istopc,itmax,itrace,irst,parms)
    integer      :: ictxt
    character(len=*) :: kmethd, ptype, afmt,renum
    integer      :: idim, istopc,itmax,itrace,irst
    type(ainvparms) :: parms
    integer      :: np, iam
    integer      :: intbuf(10), ip

    call psb_info(ictxt, iam, np)

    if (iam == 0) then
      read(psb_inp_unit,*) kmethd
      read(psb_inp_unit,*) ptype
      read(psb_inp_unit,*) afmt
      read(psb_inp_unit,*) renum
      read(psb_inp_unit,*) idim
      read(psb_inp_unit,*) istopc
      read(psb_inp_unit,*) itmax
      read(psb_inp_unit,*) itrace
      read(psb_inp_unit,*) irst
      read(psb_inp_unit,*) parms%alg
      read(psb_inp_unit,*) parms%fill
      read(psb_inp_unit,*) parms%inv_fill
      read(psb_inp_unit,*) parms%thresh
      read(psb_inp_unit,*) parms%inv_thresh
      read(psb_inp_unit,*) parms%orth_alg
      
      write(psb_out_unit,'("Solving matrix       : ell1")')      
      write(psb_out_unit,'("Grid dimensions      : ",i8,"x",i8)')idim,idim
      write(psb_out_unit,'("Number of processors : ",i0)')np
      write(psb_out_unit,'("Data distribution    : BLOCK")')
      write(psb_out_unit,'("Preconditioner       : ",a)') ptype
      write(psb_out_unit,'("Iterative method     : ",a)') kmethd
      write(psb_out_unit,'(" ")')
    end if
    call psb_bcast(ictxt,kmethd)
    call psb_bcast(ictxt,afmt)
    call psb_bcast(ictxt,renum)
    call psb_bcast(ictxt,ptype)
    call psb_bcast(ictxt,idim)
    call psb_bcast(ictxt,istopc)
    call psb_bcast(ictxt,itmax)
    call psb_bcast(ictxt,itrace)
    call psb_bcast(ictxt,irst)
    call psb_bcast(ictxt,parms%alg)
    call psb_bcast(ictxt,parms%fill)
    call psb_bcast(ictxt,parms%inv_fill)
    call psb_bcast(ictxt,parms%thresh)
    call psb_bcast(ictxt,parms%inv_thresh)
    call psb_bcast(ictxt,parms%orth_alg)

    return

  end subroutine get_parms
  !
  !  print an error message 
  !  
  subroutine pr_usage(iout)
    integer :: iout
    write(iout,*)'incorrect parameter(s) found'
    write(iout,*)' usage:  pde90 methd prec dim &
         &[istop itmax itrace]'  
    write(iout,*)' where:'
    write(iout,*)'     methd:    cgstab cgs rgmres bicgstabl' 
    write(iout,*)'     prec :    bjac diag none'
    write(iout,*)'     dim       number of points along each axis'
    write(iout,*)'               the size of the resulting linear '
    write(iout,*)'               system is dim**3'
    write(iout,*)'     istop     stopping criterion  1, 2  '
    write(iout,*)'     itmax     maximum number of iterations [500] '
    write(iout,*)'     itrace    <=0  (no tracing, default) or '  
    write(iout,*)'               >= 1 do tracing every itrace'
    write(iout,*)'               iterations ' 
  end subroutine pr_usage

  !
  !  subroutine to allocate and fill in the coefficient matrix and
  !  the rhs. 
  !

  !
  !  subroutine to allocate and fill in the coefficient matrix and
  !  the rhs. 
  !
!!$  subroutine create_matrix(idim,a,bv,xv,desc_a,ictxt,afmt,info,coord)
!!$    !
!!$    !   discretize the partial diferential equation
!!$    ! 
!!$    !   b1 dd(u)  b2 dd(u)    b3 dd(u)    a1 d(u)   a2 d(u)  a3 d(u)  
!!$    ! -   ------ -  ------ -  ------ +  -----  +  ------  +  ------ + a4 u 
!!$    !      dxdx     dydy       dzdz        dx       dy         dz   
!!$    !
!!$    ! with Dirichlet boundary conditions, on the unit cube  0<=x,y,z<=1.
!!$    !
!!$    ! Note that if a1=a2=a3=a4=0., the PDE is the well-known Laplace equation.
!!$    !
!!$    use psb_base_mod
!!$    use psb_mat_mod
!!$    use pde2d_mod
!!$    implicit none
!!$    integer                      :: idim
!!$    integer, parameter           :: nb=20
!!$    real(psb_dpk_), allocatable, optional :: coord(:,:)
!!$    type(psb_d_vect_type)        :: xv,bv
!!$    type(psb_desc_type)          :: desc_a
!!$    integer                      :: ictxt, info
!!$    character                    :: afmt*5
!!$    type(psb_dspmat_type)       :: a
!!$    type(psb_d_csc_sparse_mat)       :: acsc
!!$    type(psb_d_coo_sparse_mat)       :: acoo
!!$    type(psb_d_csr_sparse_mat)       :: acsr
!!$    real(psb_dpk_)           :: zt(nb),x,y,z
!!$    integer                  :: m,n,nnz,glob_row,nlr,i,ii,ib,k
!!$    integer                  :: ix,iy,iz,ia,indx_owner
!!$    integer                  :: np, iam, nr, nt
!!$    integer                  :: element
!!$    integer, allocatable     :: irow(:),icol(:),myidx(:)
!!$    real(psb_dpk_), allocatable :: val(:),xy(:,:)
!!$    ! deltah dimension of each grid cell
!!$
!!$    ! deltat discretization time
!!$    real(psb_dpk_)         :: deltah, deltah2, sqdeltah
!!$    real(psb_dpk_),parameter   :: rhs=0.d0,one=1.d0,zero=0.d0
!!$    real(psb_dpk_)   :: t0, t1, t2, t3, tasb, talc, ttot, tgen 
!!$    integer            :: err_act
!!$
!!$    character(len=20)  :: name, ch_err,tmpfmt
!!$
!!$    info = psb_success_
!!$    name = 'create_matrix'
!!$    call psb_erractionsave(err_act)
!!$
!!$    call psb_info(ictxt, iam, np)
!!$
!!$    deltah   = 1.d0/(idim-1)
!!$    sqdeltah = deltah**2
!!$    deltah2  = 2*deltah
!!$    ! initialize array descriptor and sparse matrix storage. provide an
!!$    ! estimate of the number of non zeroes 
!!$
!!$    m   = idim*idim
!!$    n   = m
!!$    nnz = ((n*6)/(np))
!!$    if(iam == psb_root_) write(psb_out_unit,'("Generating Matrix (size=",i0,")...")')n
!!$
!!$    !
!!$    ! Using a simple BLOCK distribution.
!!$    !
!!$    nt = (m+np-1)/np
!!$    nr = max(0,min(nt,m-(iam*nt)))
!!$
!!$    nt = nr
!!$    call psb_sum(ictxt,nt) 
!!$    if (nt /= m) write(psb_err_unit,*) iam, 'Initialization error ',nr,nt,m
!!$    call psb_barrier(ictxt)
!!$    t0 = psb_wtime()
!!$    call psb_cdall(ictxt,desc_a,info,nl=nr)
!!$    if (info == psb_success_) call psb_spall(a,desc_a,info,nnz=nnz)
!!$    ! define  rhs from boundary conditions; also build initial guess 
!!$    if (info == psb_success_) call psb_geall(bv,desc_a,info)
!!$    if (info == psb_success_) call psb_geall(xv,desc_a,info)
!!$
!!$    if (present(coord)) then 
!!$      call psb_geall(coord,desc_a,info,n=2)
!!$      if (info /= 0) write(0,*) 'Error from geall coord',info
!!$    end if
!!$    nlr = psb_cd_get_local_rows(desc_a)
!!$    call psb_barrier(ictxt)
!!$    talc = psb_wtime()-t0
!!$
!!$    if (info /= psb_success_) then
!!$      info=psb_err_from_subroutine_
!!$      ch_err='allocation rout.'
!!$      call psb_errpush(info,name,a_err=ch_err)
!!$      goto 9999
!!$    end if
!!$
!!$    ! we build an auxiliary matrix consisting of one row at a
!!$    ! time; just a small matrix. might be extended to generate 
!!$    ! a bunch of rows per call. 
!!$    ! 
!!$    allocate(xy(nb,2),val(20*nb),irow(20*nb),&
!!$         &icol(20*nb),myidx(nlr),stat=info)
!!$    if (info /= psb_success_ ) then 
!!$      info=psb_err_alloc_dealloc_
!!$      call psb_errpush(info,name)
!!$      goto 9999
!!$    endif
!!$
!!$    do i=1,nlr
!!$      myidx(i) = i
!!$    end do
!!$
!!$
!!$    call psb_loc_to_glob(myidx,desc_a,info)
!!$
!!$    ! loop over rows belonging to current process in a block
!!$    ! distribution.
!!$
!!$    call psb_barrier(ictxt)
!!$    t1 = psb_wtime()
!!$    do ii=1, nlr,nb
!!$      
!!$      ib = min(nb,nlr-ii+1) 
!!$      element = 1
!!$      do k=1,ib
!!$        i=ii+k-1
!!$        ! local matrix pointer 
!!$        glob_row=myidx(i)
!!$        ! compute gridpoint coordinates
!!$        if (mod(glob_row,(idim)) == 0) then
!!$          ix = glob_row/(idim)
!!$        else
!!$          ix = glob_row/(idim)+1
!!$        endif
!!$        iy = glob_row-(ix-1)*idim
!!$        ! x, y
!!$        x=(ix-1)*deltah
!!$        y=(iy-1)*deltah
!!$        xy(k,1) = x
!!$        xy(k,2) = y
!!$        z = 0.d0
!!$        ! check on boundary points 
!!$        zt(k) = 0.d0
!!$        ! internal point: build discretization
!!$        !   
!!$        !  term depending on   (x-1,y,z)
!!$        !
!!$        if (ix==1) then 
!!$          val(element) = -b1(x,y,z)/sqdeltah -a1(x,y,z)/deltah2
!!$          !zt(k) = exp(-x**2-y**2)*(-val(element))
!!$          zt(k) = (-val(element))
!!$        else
!!$          val(element) = -b1(x,y,z)/sqdeltah -a1(x,y,z)/deltah2
!!$          icol(element)=(ix-2)*idim+(iy)
!!$          irow(element) = glob_row
!!$          element=element+1
!!$        endif
!!$        !  term depending on     (x,y-1,z)
!!$        if (iy==1) then 
!!$          val(element)=-b2(x,y,z)/sqdeltah -a2(x,y,z)/deltah2
!!$          val(element) = val(element)/sqdeltah
!!$          zt(k) = exp(-y**2-x**2)*exp(-x)*(-val(element))  
!!$          zt(k) = 0*(-val(element))  
!!$        else
!!$          val(element)  = -b2(x,y,z)/sqdeltah -a2(x,y,z)/deltah2
!!$          icol(element) = (ix-1)*idim+(iy-1)
!!$          irow(element) = glob_row
!!$          element=element+1
!!$        endif
!!$        !  term depending on     (x,y,z)
!!$        val(element)=2*b1(x,y,z)/sqdeltah&
!!$             & +2*b2(x,y,z)/sqdeltah&
!!$             & +a4(x,y,z)
!!$        icol(element) = (ix-1)*idim+(iy)
!!$        irow(element) = glob_row
!!$        element=element+1                  
!!$        !  term depending on     (x,y+1,z)
!!$        if (iy==idim) then 
!!$          val(element)= -b2(x,y,z)/sqdeltah +a2(x,y,z)/deltah2
!!$
!!$          zt(k) = exp(-y**2-x**2)*exp(-x)*(-val(element))  
!!$          zt(k) = 0*(-val(element))  
!!$        else
!!$          val(element)=-b2(x,y,z)/sqdeltah +a2(x,y,z)/deltah2
!!$          icol(element)=(ix-1)*idim+(iy+1)
!!$          irow(element) = glob_row
!!$          element=element+1
!!$        endif
!!$        !  term depending on     (x+1,y,z)
!!$        if (ix==idim) then 
!!$          val(element)= -b1(x,y,z)/sqdeltah+a1(x,y,z)/deltah2
!!$          zt(k) = exp(-y**2-x**2)*exp(-x)*(-val(element))  
!!$          zt(k) = 1.d0*(-val(element))  
!!$        else
!!$          val(element)=-b1(x,y,z)/sqdeltah+a1(x,y,z)/deltah2
!!$          icol(element)=(ix)*idim+(iy)
!!$          irow(element) = glob_row
!!$          element=element+1
!!$        endif
!!$
!!$      end do
!!$      call psb_spins(element-1,irow,icol,val,a,desc_a,info)
!!$      if(info /= psb_success_) exit
!!$      call psb_geins(ib,myidx(ii:ii+ib-1),zt(1:ib),bv,desc_a,info)
!!$      if(info /= psb_success_) exit
!!$      zt(1:ib)=myidx(ii:ii+ib-1)
!!$      zt(1:ib)=0.0d0
!!$      call psb_geins(ib,myidx(ii:ii+ib-1),zt(1:ib),xv,desc_a,info)
!!$      if(info /= psb_success_) exit
!!$      if (present(coord)) then 
!!$        call psb_geins(ib,myidx(ii:ii+ib-1),xy(1:ib,1:2),coord,desc_a,info)
!!$        if (info /= 0) write(0,*) 'Error from geins coord',info
!!$      end if
!!$    end do
!!$    
!!$    tgen = psb_wtime()-t1
!!$    if(info /= psb_success_) then
!!$      info=psb_err_from_subroutine_
!!$      ch_err='insert rout.'
!!$      call psb_errpush(info,name,a_err=ch_err)
!!$      goto 9999
!!$    end if
!!$
!!$    deallocate(val,irow,icol,stat=info)
!!$    call psb_barrier(ictxt)
!!$    t1 = psb_wtime()
!!$    call psb_cdasb(desc_a,info)
!!$    if (info == psb_success_) &
!!$         & call psb_spasb(a,desc_a,info,dupl=psb_dupl_err_,afmt=afmt)
!!$    call psb_barrier(ictxt)
!!$    if(info /= psb_success_) then
!!$      info=psb_err_from_subroutine_
!!$      ch_err='asb rout.'
!!$      call psb_errpush(info,name,a_err=ch_err)
!!$      goto 9999
!!$    end if
!!$    if (present(coord)) then 
!!$      call psb_geasb(coord,desc_a,info)
!!$      if (info /= 0) write(0,*) 'Error from geasb coord',info
!!$    end if
!!$    call psb_geasb(b,desc_a,info)
!!$    if (info == psb_success_) call psb_geasb(xv,desc_a,info)
!!$    if (info == psb_success_) call psb_geasb(bv,desc_a,info)
!!$    if(info /= psb_success_) then
!!$      info=psb_err_from_subroutine_
!!$      ch_err='asb rout.'
!!$      call psb_errpush(info,name,a_err=ch_err)
!!$      goto 9999
!!$    end if
!!$    tasb = psb_wtime()-t1
!!$    call psb_barrier(ictxt)
!!$    ttot = psb_wtime() - t0 
!!$
!!$    call psb_amx(ictxt,talc)
!!$    call psb_amx(ictxt,tgen)
!!$    call psb_amx(ictxt,tasb)
!!$    call psb_amx(ictxt,ttot)
!!$    if(iam == psb_root_) then
!!$      tmpfmt = a%get_fmt()
!!$      write(psb_out_unit,&
!!$           & '("The matrix has been generated and assembled in ",a3," format.")')&
!!$           &   tmpfmt
!!$      write(psb_out_unit,'("-allocation  time : ",es12.5)') talc
!!$      write(psb_out_unit,'("-coeff. gen. time : ",es12.5)') tgen
!!$      write(psb_out_unit,'("-assembly    time : ",es12.5)') tasb
!!$      write(psb_out_unit,'("-total       time : ",es12.5)') ttot
!!$
!!$    end if
!!$    call psb_erractionrestore(err_act)
!!$    return
!!$
!!$9999 continue
!!$    call psb_erractionrestore(err_act)
!!$    if (err_act == psb_act_abort_) then
!!$      call psb_error(ictxt)
!!$      return
!!$    end if
!!$    return
!!$  end subroutine create_matrix
end program mldpde2d

