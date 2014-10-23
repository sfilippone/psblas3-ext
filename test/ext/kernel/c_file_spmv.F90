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
program c_file_spmv
  use psb_base_mod
  use psb_util_mod
  use psb_ext_mod
  use data_input
  implicit none

  ! input parameters
  character(len=200) :: mtrx_file

  ! sparse matrices
  type(psb_cspmat_type) :: a, aux_a

  ! dense matrices
  complex(psb_spk_), allocatable, target :: aux_b(:,:), d(:)
  complex(psb_spk_), allocatable , save  :: b_col(:), x_col(:), r_col(:), &
       & x_col_glob(:), r_col_glob(:), bres(:)
  complex(psb_spk_), pointer  :: b_col_glob(:)
  type(psb_c_vect_type) :: xv, bv
  complex(psb_spk_), allocatable :: xc1(:),xc2(:)
  ! communications data structure
  type(psb_desc_type):: desc_a

  integer            :: ictxt, iam, np
  integer(psb_long_int_k_) :: amatsize, precsize, descsize, annz, nbytes
  complex(psb_spk_)   :: err, eps 

  character(len=5)   :: acfmt
  character(len=20)  :: name
  character(len=2)   :: filefmt
  integer, parameter :: iunit=12
  integer, parameter :: times=2000 
  integer, parameter :: ntests=200, ngpu=50 

  type(psb_c_coo_sparse_mat), target   :: acoo
  type(psb_c_csr_sparse_mat), target   :: acsr
  type(psb_c_ell_sparse_mat), target   :: aell
  type(psb_c_hll_sparse_mat), target   :: ahll
  type(psb_c_dia_sparse_mat), target   :: adia
  class(psb_c_base_sparse_mat), pointer :: acmold, agmold
  ! other variables
  integer            :: i,info,j,nrt, ns, nr, ipart
  integer            :: internal, m,ii,nnzero
  real(psb_dpk_) :: t1, t2, tprec, flops
  real(psb_dpk_) :: tt1, tt2, tflops, gt1, gt2,gflops, gtint, bdwdth
  integer :: nrhs, nrow, n_row, dim, nv, ne
  integer, allocatable :: ivg(:), ipv(:)


  call psb_init(ictxt)
  call psb_info(ictxt,iam,np)

  if (iam < 0) then 
    ! This should not happen, but just in case
    call psb_exit(ictxt)
    stop
  endif


  name='file_spmv'
  if(psb_get_errstatus() /= 0) goto 9999
  info=psb_success_  
  call psb_set_errverbosity(2)
  if (iam == psb_root_) then 
    write(*,*) 'Welcome to PSBLAS version: ',psb_version_string_
    write(*,*) 'This is the ',trim(name),' sample program'
  end if

  if (iam == 0) then 
    call read_data(mtrx_file,psb_inp_unit)
    call read_data(filefmt,psb_inp_unit)
    call read_data(acfmt,psb_inp_unit)
    call read_data(agfmt,psb_inp_unit)
    call read_data(ipart,psb_inp_unit)
  end if
  call psb_bcast(ictxt,mtrx_file)
  call psb_bcast(ictxt,filefmt)
  call psb_bcast(ictxt,acfmt)
  call psb_bcast(ictxt,agfmt)
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


  select case(psb_toupper(acfmt))
  case('COO')
    acmold => acoo
  case('CSR')
    acmold => acsr
  case('ELL')
    acmold => aell
  case('HLL')
    acmold => ahll
  case('DIA')
    acmold => adia
  case default
    write(*,*) 'Unknown format defaulting to CSR'
    acmold => acsr
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

  call a%cscnv(info,mold=acmold)
  call xv%bld(x_col)
  call psb_geasb(bv,desc_a,info,scratch=.true.)

  call psb_amx(ictxt, t2)

  if (iam==psb_root_) then
    write(psb_out_unit,'(" ")')
    write(psb_out_unit,'("Time to read and partition matrix : ",es12.5)')t2
    write(psb_out_unit,'(" ")')
  end if


  call psb_barrier(ictxt)
  t1 = psb_wtime()
  do i=1,ntests 
    call psb_spmm(cone,a,xv,czero,bv,desc_a,info)
  end do
  call psb_barrier(ictxt)
  t2 = psb_wtime() - t1
  call psb_amx(ictxt,t2)


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
         &'("Size of matrix                   : ",i20,"           ")') nrt
    write(psb_out_unit,&
         &'("Number of nonzeros               : ",i20,"           ")') annz
    write(psb_out_unit,&
         &'("Memory occupation                : ",i20,"           ")') amatsize
    flops  = ntests*(2.d0*annz)
    tflops = flops
    gflops = flops * ngpu
    flops  = flops / (t2)
    tflops = tflops / (tt2)
    gflops = gflops / (gt2)
    write(psb_out_unit,'("Storage type for    A: ",a)') a%get_fmt()
    write(psb_out_unit,&
         & '("Number of flops (",i0," prod)        : ",F20.0,"           ")') &
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
    nbytes = nr*(2*2*psb_sizeof_dp + psb_sizeof_int)+&
         & annz*(2*psb_sizeof_dp + psb_sizeof_int)
    bdwdth = ntests*nbytes/(t2*1.d6)
    write(psb_out_unit,*)
    write(psb_out_unit,'("MBYTES/S                  (CPU)  : ",F20.3)') bdwdth
    write(psb_out_unit,'("Storage type for DESC_A: ",a)') desc_a%indxmap%get_fmt()
    write(psb_out_unit,'("Total memory occupation for DESC_A: ",i12)')descsize

  end if

  call psb_gefree(b_col, desc_a,info)
  call psb_gefree(x_col, desc_a,info)
  call psb_gefree(xv, desc_a,info)
  call psb_gefree(bv, desc_a,info)
  call psb_spfree(a, desc_a,info)
  call psb_cdfree(desc_a,info)

9999 continue
  if(info /= 0) then
    call psb_error(ictxt)
  end if
  call psb_exit(ictxt)
  stop

end program c_file_spmv
  




