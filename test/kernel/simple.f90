!!$  
!!$              Parallel Sparse BLAS  GPU plugin
!!$    (C) Copyright 2013
!!$                       Salvatore Filippone    University of Rome Tor Vergata
!!$                       Alessandro Fanfarillo  University of Rome Tor Vergata
!!$ 
!!$  Redistribution and use in source and binary forms, with or without
!!$  modification, are permitted provided that the following conditions
!!$  are met:
!!$    1. Redistributions of source code must retain the above copyright
!!$       notice, this list of conditions and the following disclaimer.
!!$    2. Redistributions in binary form must reproduce the above copyright
!!$       notice, this list of conditions, and the following disclaimer in the
!!$       documentation and/or other materials provided with the distribution.
!!$    3. The name of the PSBLAS group or the names of its contributors may
!!$       not be used to endorse or promote products derived from this
!!$       software without specific written permission.
!!$ 
!!$  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
!!$  ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
!!$  TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
!!$  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE PSBLAS GROUP OR ITS CONTRIBUTORS
!!$  BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
!!$  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
!!$  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
!!$  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
!!$  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
!!$  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
!!$  POSSIBILITY OF SUCH DAMAGE.
!!$ 
!!$  
! File: simple.f90
!
! Program: pdegenmv
! This sample program measures the performance of the matrix-vector product.
! The matrix is generated in the same way as for the pdegen test case of
! the main PSBLAS library.
!
!
program pdgenmv
  use psb_base_mod
  use psb_util_mod
  use psb_gpu_mod
  implicit none

  ! input parameters
  character(len=5)  :: acfmt, agfmt
  integer   :: idim

  ! miscellaneous 
  real(psb_dpk_), parameter :: one = 1.d0
  real(psb_dpk_) :: t1, t2, tprec, flops, tflops, tt1, tt2, gt1, gt2, gflops, bdwdth

  ! sparse matrix and preconditioner
  type(psb_dspmat_type) :: a, agpu
  ! descriptor
  type(psb_desc_type)   :: desc_a
  ! dense matrices
  type(psb_d_vect_type) :: xv, bv, xg, bg 
  type(psb_d_vect_gpu)  :: vmold
  real(psb_dpk_), allocatable :: xc1(:),xc2(:)
  ! blacs parameters
  integer            :: ictxt, iam, np

  ! solver parameters
  integer(psb_long_int_k_) :: amatsize, precsize, descsize, annz, nbytes
  real(psb_dpk_)   :: err, eps
  integer, parameter :: ntests=200, ngpu=50 
  type(psb_d_csr_sparse_mat), target   :: acsr
  type(psb_d_dia_sparse_mat), target   :: adia
  type(psb_d_ell_sparse_mat), target   :: aell
  type(psb_d_elg_sparse_mat), target   :: aelg
  type(psb_d_csrg_sparse_mat), target  :: acsrg
  type(psb_d_hybg_sparse_mat), target  :: ahybg
  type(psb_d_hll_sparse_mat), target   :: ahll
  type(psb_d_hlg_sparse_mat), target   :: ahlg
  class(psb_d_base_sparse_mat), pointer :: agmold, acmold
  ! other variables
  logical, parameter :: dump=.true.
  integer            :: info, i, nr
  character(len=20)  :: name,ch_err
  character(len=40)  :: fname

  info=psb_success_


  call psb_init(ictxt)
  call psb_info(ictxt,iam,np)

  !call psb_gpu_init(ictxt)

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
  !
  !  get parameters
  !
  call get_parms(ictxt,acfmt,agfmt,idim)

  !
  !  allocate and fill in the coefficient matrix and initial vectors
  !
  call psb_barrier(ictxt)
  t1 = psb_wtime()
  call psb_gen_pde3d(ictxt,idim,a,bv,xv,desc_a,'CSR  ',&
       & a1,a2,a3,b1,b2,b3,c,g,info)  
  call psb_barrier(ictxt)
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
  case('CSR')
    acmold => acsr
  case('DIA')
    acmold => adia
  case default
    write(*,*) 'Unknown format defaulting to HLL'
    acmold => ahll
  end select
  call a%cscnv(info,mold=acmold)
  if ((info /= 0).or.(psb_get_errstatus()/=0)) then 
    write(0,*) 'From cscnv ',info
    call psb_error()
    stop
  end if

  call xv%set(done)

  write(*,*) 'Prima spmm'

  call psb_barrier(ictxt)
  t1 = psb_wtime()
  do i=1,ntests 
    call psb_spmm(done,a,xv,dzero,bv,desc_a,info)
  end do
  call psb_barrier(ictxt)
  t2 = psb_wtime() - t1
  call psb_amx(ictxt,t2)

  write(*,*) 'Dopo spmm', iam,bv%get_vect()

  !call a%print(1)

  call psb_barrier(ictxt)

  ! ! FIXME: cache flush needed here
  ! call psb_barrier(ictxt)
  ! tt1 = psb_wtime()
  ! do i=1,ntests 
  !   call psb_spmm(done,agpu,xv,dzero,bg,desc_a,info)
  !   call psb_gpu_DeviceSync()
  !   if ((info /= 0).or.(psb_get_errstatus()/=0)) then 
  !     write(0,*) 'From 1 spmm',info,i,ntests
  !     call psb_error()
  !     stop
  !   end if

  ! end do
  ! call psb_barrier(ictxt)
  ! tt2 = psb_wtime() - tt1
  ! call psb_amx(ictxt,tt2)
  ! xc1 = bv%get_vect()
  ! xc2 = bg%get_vect()
  ! nr       = desc_a%get_local_rows() 
  ! eps = maxval(abs(xc1(1:nr)-xc2(1:nr)))
  ! call psb_amx(ictxt,eps)
  ! if (iam==0) write(*,*) 'Max diff on xGPU',eps


  ! call xg%sync()
  ! ! FIXME: cache flush needed here

  ! call psb_barrier(ictxt)
  ! gt1 = psb_wtime()
  ! do i=1,ntests*ngpu
  !   ! Make sure the X vector is on the GPU side of things.
  !   select type (v => xg%v)
  !   type is (psb_d_vect_gpu) 
  !     call v%set_dev()
  !   end select
  !   call psb_spmm(done,agpu,xg,dzero,bg,desc_a,info)
  !   ! For timing purposes we need to make sure all threads
  !   ! in the device are done. 
  !   call psb_gpu_DeviceSync()
  !   if ((info /= 0).or.(psb_get_errstatus()/=0)) then 
  !     write(0,*) 'From 2 spmm',info,i,ntests
  !     call psb_error()
  !     stop
  !   end if

  ! end do
  ! call psb_barrier(ictxt)
  ! gt2 = psb_wtime() - gt1
  ! call psb_amx(ictxt,gt2)
  ! call bg%sync()
  ! xc1 = bv%get_vect()
  ! xc2 = bg%get_vect()
  ! call psb_geaxpby(-done,bg,+done,bv,desc_a,info)
  ! eps = psb_geamax(bv,desc_a,info)

  ! call psb_amx(ictxt,t2)
  ! nr       = desc_a%get_local_rows() 
  ! eps = maxval(abs(xc1(1:nr)-xc2(1:nr)))
  ! call psb_amx(ictxt,eps)
  ! if (iam==0) write(*,*) 'Max diff on GPU',eps

  ! annz     = a%get_nzeros()
  ! amatsize = a%sizeof()
  ! descsize = psb_sizeof(desc_a)
  ! call psb_sum(ictxt,annz)
  ! call psb_sum(ictxt,amatsize)
  ! call psb_sum(ictxt,descsize)

  ! if (iam == psb_root_) then
  !   write(psb_out_unit,&
  !        & '("Matrix: ell1 ",i0)') idim
  !   write(psb_out_unit,&
  !        &'("Test on                          : ",i20," processors")') np
  !   write(psb_out_unit,&
  !        &'("Size of matrix                   : ",i20,"           ")') nr
  !   write(psb_out_unit,&
  !        &'("Number of nonzeros               : ",i20,"           ")') annz
  !   write(psb_out_unit,&
  !        &'("Memory occupation                : ",i20,"           ")') amatsize
  !   flops  = ntests*(2*1.d0*annz-1.d0*nr)
  !   tflops = flops
  !   gflops = flops * ngpu
  !   flops  = flops / (t2)
  !   tflops = tflops / (tt2)
  !   gflops = gflops / (gt2)
  !   write(psb_out_unit,'("Storage type for    A: ",a)') a%get_fmt()
  !   write(psb_out_unit,'("Storage type for AGPU: ",a)') agpu%get_fmt()
  !   write(psb_out_unit,&
  !        & '("Number of flops (",i0," prod)        : ",F20.0,"           ")') &
  !        &  ntests,flops
  !   write(psb_out_unit,'("Time for ",i6," products (s) (CPU)   : ",F20.3)')&
  !        &  ntests,t2
  !   write(psb_out_unit,'("Time per product    (ms)     (CPU)   : ",F20.3)')&
  !        & t2*1.d3/(1.d0*ntests)
  !   write(psb_out_unit,'("MFLOPS                       (CPU)   : ",F20.3)')&
  !        & flops/1.d6

  !   write(psb_out_unit,'("Time for ",i6," products (s) (xGPU)  : ",F20.3)')&
  !        & ntests, tt2
  !   write(psb_out_unit,'("Time per product    (ms)     (xGPU)  : ",F20.3)')&
  !        & tt2*1.d3/(1.d0*ntests)
  !   write(psb_out_unit,'("MFLOPS                       (xGPU)  : ",F20.3)')&
  !        & tflops/1.d6

  !   write(psb_out_unit,'("Time for ",i6," products (s) (GPU.)  : ",F20.3)')&
  !        & ngpu*ntests,gt2
  !   write(psb_out_unit,'("Time per product    (ms)     (GPU.)  : ",F20.3)')&
  !        & gt2*1.d3/(1.d0*ntests*ngpu)
  !   write(psb_out_unit,'("MFLOPS                       (GPU.)  : ",F20.3)')&
  !        & gflops/1.d6
  !   !
  !   ! This computation assumes the data movement associated with CSR:
  !   ! it is minimal in terms of coefficients. Other formats may either move
  !   ! more data (padding etc.) or less data (if they can save on the indices). 
  !   !
  !   nbytes = nr*(2*psb_sizeof_dp + psb_sizeof_int)+&
  !        & annz*(psb_sizeof_dp + psb_sizeof_int)
  !   bdwdth = ntests*nbytes/(t2*1.d6)
  !   write(psb_out_unit,*)
  !   write(psb_out_unit,'("MBYTES/S                  (CPU)  : ",F20.3)') bdwdth
  !   bdwdth = ngpu*ntests*nbytes/(gt2*1.d6)
  !   write(psb_out_unit,'("MBYTES/S                  (GPU)  : ",F20.3)') bdwdth
  !   write(psb_out_unit,'("Storage type for DESC_A: ",a)') desc_a%indxmap%get_fmt()
  !   write(psb_out_unit,'("Total memory occupation for DESC_A: ",i12)')descsize

  ! end if

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
  subroutine  get_parms(ictxt,acfmt,agfmt,idim)
    integer      :: ictxt
    character(len=*) :: agfmt, acfmt
    integer      :: idim
    integer      :: np, iam
    integer      :: intbuf(10), ip

    call psb_info(ictxt, iam, np)

    if (iam == 0) then
      read(psb_inp_unit,*) acfmt
      read(psb_inp_unit,*) agfmt
      read(psb_inp_unit,*) idim
    endif
    call psb_bcast(ictxt,acfmt)
    call psb_bcast(ictxt,agfmt)
    call psb_bcast(ictxt,idim)

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
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) :: a1
    real(psb_dpk_), intent(in) :: x,y,z
    a1=1.d0
  end function a1
  function a2(x,y,z)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) ::  a2
    real(psb_dpk_), intent(in) :: x,y,z
    a2=2.d1*y
  end function a2
  function a3(x,y,z)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) ::  a3
    real(psb_dpk_), intent(in) :: x,y,z      
    a3=1.d0
  end function a3
  function c(x,y,z)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) :: c
    real(psb_dpk_), intent(in) :: x,y,z      
    c=1.d0
  end function c
  function b1(x,y,z)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) ::  b1   
    real(psb_dpk_), intent(in) :: x,y,z
    b1=1.d0
  end function b1
  function b2(x,y,z)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) ::  b2
    real(psb_dpk_), intent(in) :: x,y,z
    b2=1.d0
  end function b2
  function b3(x,y,z)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) ::  b3
    real(psb_dpk_), intent(in) :: x,y,z
    b3=1.d0
  end function b3
  function g(x,y,z)
    use psb_base_mod, only : psb_dpk_, done, dzero
    real(psb_dpk_) ::  g
    real(psb_dpk_), intent(in) :: x,y,z
    g = dzero
    if (x == done) then
      g = done
    else if (x == dzero) then 
      g = exp(y**2-z**2)
    end if
  end function g

end program pdgenmv
