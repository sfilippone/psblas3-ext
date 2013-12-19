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
program compformat
  use psb_base_mod
  use psb_util_mod
  use data_input
  use psb_ext_mod
  implicit none

  ! input parameters
  character(len=255) :: mtrx_file

  ! sparse matrices
  type(psb_dspmat_type) :: a, aux_a, agpu

  ! dense matrices
  real(psb_dpk_), allocatable, target :: aux_b(:,:), d(:)
  real(psb_dpk_), allocatable , save  :: b_col(:), x_col(:), r_col(:), &
       & x_col_glob(:), r_col_glob(:), bres(:)
  real(psb_dpk_), pointer  :: b_col_glob(:)
  type(psb_d_vect_type) :: xg, bg, xv, bv
!  type(psb_d_vect_gpu)  :: vmold
  real(psb_dpk_), allocatable :: xc1(:),xc2(:)
  ! communications data structure
  type(psb_desc_type):: desc_a

  integer            :: ictxt, iam, np
  integer(psb_long_int_k_) :: amatsize, precsize, descsize, annz, nbytes
  real(psb_dpk_)   :: err, eps 

  character(len=5)   :: afmt
  character(len=20)  :: name
  character(len=2)   :: filefmt
  integer, parameter :: iunit=12
  integer, parameter :: times=2000 
  integer, parameter :: ntests=200, ngpu=50 

  type(psb_d_csr_sparse_mat), target   :: acsr
  type(psb_d_ell_sparse_mat), target   :: aell
  type(psb_d_rsb_sparse_mat), target   :: arsb
  type(psb_d_hdia_sparse_mat), target   :: ahdia
  type(psb_d_dia_sparse_mat), target   :: adia
  ! type(psb_d_elg_sparse_mat), target   :: aelg
  ! type(psb_d_csrg_sparse_mat), target  :: acsrg
  ! type(psb_d_hybg_sparse_mat), target  :: ahybg
  type(psb_d_hll_sparse_mat), target   :: ahll
  ! type(psb_d_hlg_sparse_mat), target   :: ahlg
  class(psb_d_base_sparse_mat), pointer :: amold
  ! other variables
  integer            :: i,info,j,nrt, ns, nr, ipart
  integer            :: internal, m,ii,nnzero
  real(psb_dpk_) :: t1, t2, tprec, flops
  real(psb_dpk_) :: tt1, tt2, tflops, gt1, gt2,gflops, gtint, bdwdth
  integer :: nrhs, nrow, n_row, dim, nv, ne
  integer, allocatable :: ivg(:), ipv(:)


  call psb_init(ictxt)
  call psb_info(ictxt,iam,np)
  call psb_rsb_init()

  if (iam < 0) then 
    ! This should not happen, but just in case
    call psb_exit(ictxt)
    stop
  endif


  name='file_spmv'
  if(psb_get_errstatus() /= 0) goto 9999
  info=psb_success_  
  call psb_set_errverbosity(2)

  if (iam == 0) then
     write(*,*) 'path-fileformat(MM/HB)-matrixformat(CSR/ELL/ecc..)-partition(0:block,2:graph)'
     call read_data(mtrx_file,psb_inp_unit)
     call read_data(filefmt,psb_inp_unit)
     call read_data(afmt,psb_inp_unit)
     call read_data(ipart,psb_inp_unit)
  end if
  call psb_bcast(ictxt,mtrx_file)
  call psb_bcast(ictxt,filefmt)
  call psb_bcast(ictxt,afmt)
  call psb_bcast(ictxt,ipart)
  call psb_barrier(ictxt)
  t1 = psb_wtime()  
  ! read the input matrix to be processed and (possibly) the rhs 
  nrhs = 1

  if (iam==psb_root_) then
    select case(psb_toupper(filefmt)) 
    case('MM') 
      ! For Matrix Market we have an input file for the matrix
      ! and an (optional) second file for the RHS. 
      call mm_mat_read(aux_a,info,iunit=iunit,filename=mtrx_file)

    case ('HB')
      ! For Harwell-Boeing we have a single file which may or may not
      ! contain an RHS.
      call hb_read(aux_a,info,iunit=iunit,filename=mtrx_file)

    case default
      info = -1 
      write(psb_err_unit,*) 'Wrong choice for fileformat ', filefmt
    end select
    if (info /= 0) then
      write(psb_err_unit,*) 'Error while reading input matrix '
      call psb_abort(ictxt)
    end if

    nrt = aux_a%get_nrows()
    call psb_bcast(ictxt,nrt)

    write(psb_out_unit,'("Generating an rhs...")')
    write(psb_out_unit,'(" ")')
    call psb_realloc(nrt,1,aux_b,info)
    if (info /= 0) then
      call psb_errpush(4000,name)
      goto 9999
    endif

    b_col_glob => aux_b(:,1)
    do i=1, nrt
      b_col_glob(i) = 1.d0
    enddo
    call psb_bcast(ictxt,b_col_glob(1:nrt))

  else

    call psb_bcast(ictxt,nrt)
    call psb_realloc(nrt,1,aux_b,info)
    if (info /= 0) then
      call psb_errpush(4000,name)
      goto 9999
    endif
    b_col_glob =>aux_b(:,1)
    call psb_bcast(ictxt,b_col_glob(1:nrt)) 

  end if


  select case(psb_toupper(afmt))
  case('CSR')
     amold => acsr
  case('ELL')
     amold => aell
  case('HLL')
     amold => ahll
  case('DIA')
     amold => adia
  case('HDIA')
     amold => ahdia
  case('RSB')
     amold => arsb
  case default
     write(*,*) 'Unknown format defaulting to HLG'
     amold => acsr
  end select

  ! switch over different partition types
  if (ipart == 0) then 
    call psb_barrier(ictxt)
    if (iam==psb_root_) write(psb_out_unit,'("Partition type: block")')
    allocate(ivg(nrt),ipv(np))
    do i=1,nrt
      call part_block(i,nrt,np,ipv,nv)
      ivg(i) = ipv(1)
    enddo
    call psb_matdist(aux_a, a, ictxt, &
         & desc_a,b_col_glob,bv,info,v=ivg)
  else if (ipart == 2) then 
    if (iam==psb_root_) then 
      write(psb_out_unit,'("Partition type: graph")')
      write(psb_out_unit,'(" ")')
      !      write(psb_err_unit,'("Build type: graph")')
      call build_mtpart(aux_a,np)
    endif
    call psb_barrier(ictxt)
    call distr_mtpart(psb_root_,ictxt)
    call getv_mtpart(ivg)
    call psb_matdist(aux_a, a, ictxt, &
         & desc_a,b_col_glob,bv,info,v=ivg)
  else 
    if (iam==psb_root_) write(psb_out_unit,'("Partition type default: block")')
    call psb_matdist(aux_a, a,  ictxt, &
         & desc_a,b_col_glob,bv,info,parts=part_block)
  end if

  call psb_geall(x_col,desc_a,info)
  ns = size(x_col)
  do i=1, ns
    x_col(i) = 1.0 + (1.0*i)/ns
  end do
  call psb_geasb(x_col,desc_a,info)
  t2 = psb_wtime() - t1

  !call a%cscnv(agpu,info,mold=amold)
call xv%bld(x_col)
 ! call xg%bld(x_col,mold=vmold)
 call psb_geasb(bv,desc_a,info,scratch=.true.)
!  call psb_geasb(bg,desc_a,info,scratch=.true.,mold=vmold)
 call psb_amx(ictxt, t2)

  if (iam==psb_root_) then
    write(psb_out_unit,'(" ")')
    write(psb_out_unit,'("Time to read and partition matrix : ",es12.5)')t2
    write(psb_out_unit,'(" ")')
  end if

  call psb_barrier(ictxt)
  t1 = psb_wtime()
  do i=1,ntests 
    call psb_spmm(done,a,xv,dzero,bv,desc_a,info)
  end do
  call psb_barrier(ictxt)
  t2 = psb_wtime() - t1
  call psb_amx(ictxt,t2)

  ! FIXME: cache flush needed here
  call psb_barrier(ictxt)
  tt1 = psb_wtime()
  ! do i=1,ntests 
  !   call psb_spmm(done,agpu,xv,dzero,bg,desc_a,info)
  !   if ((info /= 0).or.(psb_get_errstatus()/=0)) then 
  !     write(0,*) 'From 1 spmm',info,i,ntests
  !     call psb_error()
  !     stop
  !   end if

  ! end do
 call psb_barrier(ictxt)
  tt2 = psb_wtime() - tt1
! call psb_amx(ictxt,tt2)
! xc1 = bv%get_vect()
! xc2 = bg%get_vect()
  nr       = desc_a%get_local_rows() 
! eps = maxval(abs(xc1(1:nr)-xc2(1:nr)))
! call psb_amx(ictxt,eps)
! if (iam==0) write(*,*) 'Max diff on xGPU',eps


 !call xg%sync()
  ! FIXME: cache flush needed here

  call psb_barrier(ictxt)
  gt1 = psb_wtime()
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
  call psb_barrier(ictxt)
  gt2 = psb_wtime() - gt1
 !call psb_amx(ictxt,gt2)
 !call bg%sync()
 !xc1 = bv%get_vect()
 !xc2 = bg%get_vect()
 !call psb_geaxpby(-done,bg,+done,bv,desc_a,info)
 !eps = psb_geamax(bv,desc_a,info)

 ! call psb_amx(ictxt,t2)
  nr       = desc_a%get_local_rows() 
 !eps = maxval(abs(xc1(1:nr)-xc2(1:nr)))
 !call psb_amx(ictxt,eps)
 !if (iam==0) write(*,*) 'Max diff on GPU',eps

  annz     = a%get_nzeros()
  amatsize = a%sizeof()
  descsize = psb_sizeof(desc_a)
  call psb_sum(ictxt,annz)
  call psb_sum(ictxt,amatsize)
  call psb_sum(ictxt,descsize)

  if (iam == psb_root_) then
    write(psb_out_unit,'("Matrix: ",a)') mtrx_file
    write(psb_out_unit,&
         &'("Test on                          : ",i20," processors")') np
    write(psb_out_unit,&
         &'("Size of matrix                   : ",i20,"           ")') nr!t
    write(psb_out_unit,&
         &'("Number of nonzeros               : ",i20,"           ")') annz
    write(psb_out_unit,&
         &'("Memory occupation                : ",i20,"           ")') amatsize
    flops  = ntests*(2*1.d0*annz-1.d0*nr)
    tflops = flops
    gflops = flops * ngpu
    flops  = flops / (t2)
    tflops = tflops / (tt2)
    gflops = gflops / (gt2)
    write(psb_out_unit,'("Storage type for    A: ",a)') a%get_fmt()
 !  write(psb_out_unit,'("Storage type for AGPU: ",a)') agpu%get_fmt()
    write(psb_out_unit,&
         & '("Number of flops (",i0," prod)        : ",F20.0,"           ")') &
         &  ntests,flops
    write(psb_out_unit,'("Time for ",i6," products (s) (CPU)   : ",F20.3)')&
         &  ntests,t2
    write(psb_out_unit,'("Time per product    (ms)     (CPU)   : ",F20.3)')&
         & t2*1.d3/(1.d0*ntests)
    write(psb_out_unit,'("MFLOPS                       (CPU)   : ",F20.3)')&
         & flops/1.d6

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
    !
    ! This computation assumes the data movement associated with CSR:
    ! it is minimal in terms of coefficients. Other formats may either move
    ! more data (padding etc.) or less data (if they can save on the indices). 
    !
    nbytes = nr*(2*psb_sizeof_dp + psb_sizeof_int)+&
         & annz*(psb_sizeof_dp + psb_sizeof_int)
    bdwdth = ntests*nbytes/(t2*1.d6)
    write(psb_out_unit,*)
    write(psb_out_unit,'("MBYTES/S                  (CPU)  : ",F20.3)') bdwdth
    bdwdth = ngpu*ntests*nbytes/(gt2*1.d6)
    write(psb_out_unit,'("MBYTES/S                  (GPU)  : ",F20.3)') bdwdth
    write(psb_out_unit,'("Storage type for DESC_A: ",a)') desc_a%indxmap%get_fmt()
    write(psb_out_unit,'("Total memory occupation for DESC_A: ",i12)')descsize

  end if

  call psb_gefree(b_col, desc_a,info)
  call psb_gefree(x_col, desc_a,info)
  call psb_gefree(xv, desc_a,info)
  call psb_gefree(bv, desc_a,info)
  call psb_gefree(xg, desc_a,info)
  call psb_gefree(bg, desc_a,info)
  call psb_spfree(a, desc_a,info)
  call psb_spfree(agpu,desc_a,info)
  call psb_cdfree(desc_a,info)

9999 continue
  if(info /= 0) then
    call psb_error(ictxt)
  end if
  call psb_exit(ictxt)
  stop

end program compformat