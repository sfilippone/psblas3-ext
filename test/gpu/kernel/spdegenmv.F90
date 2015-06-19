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
! File: spdegenmv.f90
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
  use psb_ext_mod
#ifdef HAVE_GPU
  use psb_gpu_mod
#endif
  implicit none

  ! input parameters
  character(len=5)  :: acfmt, agfmt
  integer   :: idim

  ! miscellaneous 
  real(psb_spk_), parameter :: one = 1.e0
  real(psb_dpk_) :: t1, t2, tprec, flops, tflops,&
       & tt1, tt2, gt1, gt2, gflops, bdwdth,&
       & tcnvcsr, tcnvc1, tcnvgpu, tcnvg1

  ! sparse matrix and preconditioner
  type(psb_sspmat_type) :: a, agpu, aux_a
  ! descriptor
  type(psb_desc_type)   :: desc_a
  ! dense matrices
  type(psb_s_vect_type), target :: xv,bv, xg, bg 
#ifdef HAVE_GPU
  type(psb_s_vect_gpu)  :: vmold
  type(psb_i_vect_gpu)  :: imold
#endif
  real(psb_spk_), allocatable :: x1(:), x2(:), x0(:)
  ! blacs parameters
  integer            :: ictxt, iam, np

  ! solver parameters
  integer(psb_long_int_k_) :: amatsize, precsize, descsize, annz, nbytes
  real(psb_spk_)   :: err, eps
  integer, parameter :: ntests=200, ngpu=50, ncnv=20
  type(psb_s_coo_sparse_mat), target   :: acoo
  type(psb_s_csr_sparse_mat), target   :: acsr
  type(psb_s_ell_sparse_mat), target   :: aell
  type(psb_s_hll_sparse_mat), target   :: ahll
  type(psb_s_dia_sparse_mat), target   :: adia
  type(psb_s_hdia_sparse_mat), target   :: ahdia
#ifdef HAVE_GPU
  type(psb_s_elg_sparse_mat), target   :: aelg
  type(psb_s_csrg_sparse_mat), target  :: acsrg
  type(psb_s_hybg_sparse_mat), target  :: ahybg
  type(psb_s_hlg_sparse_mat), target   :: ahlg
  type(psb_s_hdiag_sparse_mat), target   :: ahdiag
#endif
  class(psb_s_base_sparse_mat), pointer :: agmold, acmold
  ! other variables
  logical, parameter :: dump=.false.
  integer            :: info, i, j, nr, ig, nrg
  character(len=20)  :: name,ch_err
  character(len=40)  :: fname

  info=psb_success_

  
  call psb_init(ictxt)
  call psb_info(ictxt,iam,np)

#ifdef HAVE_GPU
  call psb_gpu_init(ictxt, iam)
#endif

  if (iam < 0) then 
    ! This should not happen, but just in case
    call psb_exit(ictxt)
    stop
  endif
  if(psb_get_errstatus() /= 0) goto 9999
  name='pdegenmv-gpu'
  call psb_set_errverbosity(2)
  !
  ! Hello world
  !
  if (iam == psb_root_) then 
    write(*,*) 'Welcome to PSBLAS version: ',psb_version_string_
    write(*,*) 'This is the ',trim(name),' sample program'
  end if
#ifdef HAVE_GPU
  write(*,*) 'Process ',iam,' running on device: ', psb_cuda_getDevice(),' out of', psb_cuda_getDeviceCount()
  write(*,*) 'Process ',iam,' device ', psb_cuda_getDevice(),' is a: ', trim(psb_gpu_DeviceName())  
#endif
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
    write(fname,'(a,i3.3,a,i3.3,a,i3.3,a)')  'pde',idim,'-',iam,'-',np,'.mtx'
    call a%print(fname,head='PDEGEN test matrix')
  end if

  select case(psb_toupper(acfmt))
  case('ELL')
    acmold => aell
  case('HLL')
    acmold => ahll
  case('DIA')
    acmold => adia
  case('HDIA')
    acmold => ahdia
  case('CSR')
    acmold => acsr
  case('COO')
    acmold => acoo
#ifdef HAVE_RSB
  case('RSB')
    acmold => arsb
#endif
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

#ifdef HAVE_GPU
  select case(psb_toupper(agfmt))
  case('ELG')
    agmold => aelg
  case('HLG')
    agmold => ahlg
  case('HDIAG')
    agmold => ahdiag
  case('CSRG')
    agmold => acsrg
  case('HYBG')
    agmold => ahybg
  case default
    write(*,*) 'Unknown format defaulting to HLG'
    agmold => ahlg
  end select
  call a%cscnv(agpu,info,mold=agmold)
  if ((info /= 0).or.(psb_get_errstatus()/=0)) then 
    write(0,*) 'From cscnv ',info
    call psb_error()
    stop
  end if
  call desc_a%cnv(mold=imold)

  call psb_geasb(bg,desc_a,info,scratch=.true.,mold=vmold)
  call psb_geasb(xg,desc_a,info,scratch=.true.,mold=vmold)
#endif
  nr       = desc_a%get_local_rows()
  nrg      = desc_a%get_global_rows() 
  call psb_geall(x0,desc_a,info)
  do i=1, nr
    call desc_a%l2g(i,ig,info)
    x0(i) = 1.0 + (1.0*ig)/nrg
  end do
  call a%cscnv(aux_a,info,mold=acoo)
  tcnvcsr = 0
  tcnvgpu = 0
  call psb_geall(x1,desc_a,info)
  do j=1, ncnv
    call aux_a%cscnv(a,info,mold=acoo)
    call psb_barrier(ictxt)
    t1 = psb_wtime()
    call a%cscnv(info,mold=acmold)
    t2 = psb_Wtime() -t1
    call psb_amx(ictxt,t2)
    tcnvcsr = tcnvcsr + t2
    if (j==1) tcnvc1 = t2
    call psb_geasb(x1,desc_a,info)
    call xv%bld(x0)
    call psb_geasb(bv,desc_a,info,scratch=.true.)
    
#ifdef HAVE_GPU
    
    call aux_a%cscnv(agpu,info,mold=acoo)
    call xg%bld(x0,mold=vmold)
    call psb_geasb(bg,desc_a,info,scratch=.true.,mold=vmold)
    call psb_barrier(ictxt)
    t1 = psb_wtime()
    call agpu%cscnv(info,mold=agmold)
    call psb_gpu_DeviceSync()
    t2 = psb_Wtime() -t1
    call psb_amx(ictxt,t2)
    if (j==1) tcnvg1 = t2
    tcnvgpu = tcnvgpu + t2
#endif
  end do


  call xv%set(x0)
  call psb_barrier(ictxt)
  t1 = psb_wtime()
  do i=1,ntests 
    call psb_spmm(sone,a,xv,szero,bv,desc_a,info)
  end do
  call psb_barrier(ictxt)
  t2 = psb_wtime() - t1
  call psb_amx(ictxt,t2)

#ifdef HAVE_GPU
  call xg%set(x0)

  ! FIXME: cache flush needed here
  x1 = bv%get_vect()
  x2 = bg%get_vect()
  
  call psb_barrier(ictxt)
  tt1 = psb_wtime()
  do i=1,ntests 
    call psb_spmm(sone,agpu,xv,szero,bg,desc_a,info)
    if ((info /= 0).or.(psb_get_errstatus()/=0)) then 
      write(0,*) 'From 1 spmm',info,i,ntests
      call psb_error()
      stop
    end if

  end do
  call psb_gpu_DeviceSync()
  call psb_barrier(ictxt)
  tt2 = psb_wtime() - tt1
  call psb_amx(ictxt,tt2)
  x1 = bv%get_vect()
  x2 = bg%get_vect()
  nr       = desc_a%get_local_rows() 
  eps = maxval(abs(x1(1:nr)-x2(1:nr)))
  call psb_amx(ictxt,eps)
  if (iam==0) write(*,*) 'Max diff on xGPU',eps


  ! FIXME: cache flush needed here
  call xg%set(x0)
  call xg%sync()
  call psb_barrier(ictxt)
  gt1 = psb_wtime()
  do i=1,ntests*ngpu
    call psb_spmm(sone,agpu,xg,szero,bg,desc_a,info)
    ! For timing purposes we need to make sure all threads
    ! in the device are done. 
    if ((info /= 0).or.(psb_get_errstatus()/=0)) then 
      write(0,*) 'From 2 spmm',info,i,ntests
      call psb_error()
      stop
    end if
    
  end do
  call psb_gpu_DeviceSync()
  call psb_barrier(ictxt)
  gt2 = psb_wtime() - gt1
  call psb_amx(ictxt,gt2)
  call bg%sync()
  x1 = bv%get_vect()
  x2 = bg%get_vect()
  call psb_geaxpby(-sone,bg,+sone,bv,desc_a,info)
  eps = psb_geamax(bv,desc_a,info)

  call psb_amx(ictxt,t2)
  eps = maxval(abs(x1(1:nr)-x2(1:nr)))
  call psb_amx(ictxt,eps)
  if (iam==0) write(*,*) 'Max diff on GPU',eps
  if (dump) then 
    write(fname,'(a,i3.3,a,i3.3,a)')'XCPU-out-',iam,'-',np,'.mtx'
    call mm_array_write(x1(1:nr),'Local part CPU',info,filename=fname)
    write(fname,'(a,i3.3,a,i3.3,a)')'XGPU-out-',iam,'-',np,'.mtx'
    call mm_array_write(x2(1:nr),'Local part GPU',info,filename=fname)
  end if
#endif
  annz     = a%get_nzeros()
  amatsize = a%sizeof()
  descsize = psb_sizeof(desc_a)
  call psb_sum(ictxt,nr)
  call psb_sum(ictxt,annz)
  call psb_sum(ictxt,amatsize)
  call psb_sum(ictxt,descsize)
  
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
    gflops = flops * ngpu
    write(psb_out_unit,'("Storage type for    A: ",a)') a%get_fmt()
#ifdef HAVE_GPU
    write(psb_out_unit,'("Storage type for AGPU: ",a)') agpu%get_fmt()
    write(psb_out_unit,'("Time to convert A from COO to CPU (1): ",F20.9)')&
         & tcnvc1
    write(psb_out_unit,'("Time to convert A from COO to CPU (t): ",F20.9)')&
         & tcnvcsr
    write(psb_out_unit,'("Time to convert A from COO to CPU (a): ",F20.9)')&
         & tcnvcsr/ncnv
    write(psb_out_unit,'("Time to convert A from COO to GPU (1): ",F20.9)')&
         & tcnvg1
    write(psb_out_unit,'("Time to convert A from COO to GPU (t): ",F20.9)')&
         & tcnvgpu
    write(psb_out_unit,'("Time to convert A from COO to GPU (a): ",F20.9)')&
         & tcnvgpu/ncnv

#endif
    write(psb_out_unit,&
         & '("Number of flops (",i0," prod)        : ",F20.0,"           ")') &
         &  ntests,flops

    flops  = flops / (t2)
    tflops = tflops / (tt2)
    gflops = gflops / (gt2)

    write(psb_out_unit,'("Time for ",i6," products (s) (CPU)   : ",F20.3)')&
         &  ntests,t2
    write(psb_out_unit,'("Time per product    (ms)     (CPU)   : ",F20.3)')&
         & t2*1.d3/(1.d0*ntests)
    write(psb_out_unit,'("MFLOPS                       (CPU)   : ",F20.3)')&
         & flops/1.d6
#ifdef HAVE_GPU
    write(psb_out_unit,'("Time for ",i6," products (s) (xGPU)  : ",F20.3)')&
         & ntests, tt2
    write(psb_out_unit,'("Time per product    (ms)     (xGPU)  : ",F20.3)')&
         & tt2*1.d3/(1.d0*ntests)
    write(psb_out_unit,'("MFLOPS                       (xGPU)  : ",F20.3)')&
         & tflops/1.d6

    write(psb_out_unit,'("Time for ",i6," products (s) (GPU.)  : ",F20.3)')&
         & ngpu*ntests,gt2
    write(psb_out_unit,'("Time per product    (ms)     (GPU.)  : ",F20.3)')&
         & gt2*1.d3/(1.d0*ntests*ngpu)
    write(psb_out_unit,'("MFLOPS                       (GPU.)  : ",F20.3)')&
         & gflops/1.d6
#endif
    !
    ! This computation assumes the data movement associated with CSR:
    ! it is minimal in terms of coefficients. Other formats may either move
    ! more data (padding etc.) or less data (if they can save on the indices). 
    !
    nbytes = nr*(2*psb_sizeof_sp + psb_sizeof_int)+&
         & annz*(psb_sizeof_sp + psb_sizeof_int)
    bdwdth = ntests*nbytes/(t2*1.d6)
    write(psb_out_unit,*)
    write(psb_out_unit,'("MBYTES/S sust. effective bandwidth  (CPU)  : ",F20.3)') bdwdth
#ifdef HAVE_GPU
    bdwdth = ngpu*ntests*nbytes/(gt2*1.d6)
    write(psb_out_unit,'("MBYTES/S sust. effective bandwidth  (GPU)  : ",F20.3)') bdwdth
    bdwdth = psb_gpu_MemoryPeakBandwidth()
    write(psb_out_unit,'("MBYTES/S peak bandwidth             (GPU)  : ",F20.3)') bdwdth
#endif
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
#ifdef HAVE_GPU
  call psb_gpu_exit()
#endif
  call psb_exit(ictxt)
  stop

9999 continue
  call psb_error(ictxt)

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
      write(*,*) 'CPU side format?'
      read(psb_inp_unit,*) acfmt
      write(*,*) 'GPU side format?'
      read(psb_inp_unit,*) agfmt
      write(*,*) 'Size of discretization cube?'
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
