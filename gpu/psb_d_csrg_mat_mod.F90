!                Parallel Sparse BLAS   GPU plugin 
!      (C) Copyright 2013
!  
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
  

module psb_d_csrg_mat_mod

  use iso_c_binding
  use psb_d_mat_mod
  use cusparse_mod

  integer(psb_ipk_), parameter, private :: is_host = -1
  integer(psb_ipk_), parameter, private :: is_sync = 0 
  integer(psb_ipk_), parameter, private :: is_dev  = 1 

  type, extends(psb_d_csr_sparse_mat) :: psb_d_csrg_sparse_mat
    !
    ! cuSPARSE 4.0 CSR format.
    ! 
    ! 
    ! 
    ! 
    ! 
#ifdef HAVE_SPGPU
    type(d_Cmat)    :: deviceMat
    integer(psb_ipk_) :: devstate  = is_host
    
  contains
    procedure, nopass  :: get_fmt       => d_csrg_get_fmt
    procedure, pass(a) :: sizeof        => d_csrg_sizeof
    procedure, pass(a) :: vect_mv       => psb_d_csrg_vect_mv
    procedure, pass(a) :: in_vect_sv    => psb_d_csrg_inner_vect_sv
    procedure, pass(a) :: csmm          => psb_d_csrg_csmm
    procedure, pass(a) :: csmv          => psb_d_csrg_csmv
    procedure, pass(a) :: scals         => psb_d_csrg_scals
    procedure, pass(a) :: scalv         => psb_d_csrg_scal
    procedure, pass(a) :: reallocate_nz => psb_d_csrg_reallocate_nz
    procedure, pass(a) :: allocate_mnnz => psb_d_csrg_allocate_mnnz
    ! Note: we do *not* need the TO methods, because the parent type
    ! methods will work. 
    procedure, pass(a) :: cp_from_coo   => psb_d_cp_csrg_from_coo
    procedure, pass(a) :: cp_from_fmt   => psb_d_cp_csrg_from_fmt
    procedure, pass(a) :: mv_from_coo   => psb_d_mv_csrg_from_coo
    procedure, pass(a) :: mv_from_fmt   => psb_d_mv_csrg_from_fmt
    procedure, pass(a) :: free          => d_csrg_free
    procedure, pass(a) :: mold          => psb_d_csrg_mold
    procedure, pass(a) :: is_host       => d_csrg_is_host
    procedure, pass(a) :: is_dev        => d_csrg_is_dev
    procedure, pass(a) :: is_sync       => d_csrg_is_sync
    procedure, pass(a) :: set_host      => d_csrg_set_host
    procedure, pass(a) :: set_dev       => d_csrg_set_dev
    procedure, pass(a) :: set_sync      => d_csrg_set_sync
    procedure, pass(a) :: sync          => d_csrg_sync
    procedure, pass(a) :: to_gpu        => psb_d_csrg_to_gpu
    procedure, pass(a) :: from_gpu      => psb_d_csrg_from_gpu
    final              :: d_csrg_finalize
#else 
  contains
    procedure, pass(a) :: mold         => psb_d_csrg_mold
#endif
  end type psb_d_csrg_sparse_mat

#ifdef HAVE_SPGPU
  private :: d_csrg_get_nzeros, d_csrg_free,  d_csrg_get_fmt, &
       & d_csrg_get_size, d_csrg_sizeof, d_csrg_get_nz_row


  interface 
    subroutine psb_d_csrg_inner_vect_sv(alpha,a,x,beta,y,info,trans) 
      import :: psb_d_csrg_sparse_mat, psb_dpk_, psb_d_base_vect_type, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(in) :: a
      real(psb_dpk_), intent(in)       :: alpha, beta
      class(psb_d_base_vect_type), intent(inout) :: x
      class(psb_d_base_vect_type), intent(inout) :: y
      integer(psb_ipk_), intent(out)             :: info
      character, optional, intent(in)  :: trans
    end subroutine psb_d_csrg_inner_vect_sv
  end interface


  interface 
    subroutine psb_d_csrg_vect_mv(alpha,a,x,beta,y,info,trans) 
      import :: psb_d_csrg_sparse_mat, psb_dpk_, psb_d_base_vect_type, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(in) :: a
      real(psb_dpk_), intent(in)       :: alpha, beta
      class(psb_d_base_vect_type), intent(inout) :: x
      class(psb_d_base_vect_type), intent(inout) :: y
      integer(psb_ipk_), intent(out)             :: info
      character, optional, intent(in)  :: trans
    end subroutine psb_d_csrg_vect_mv
  end interface

  interface
    subroutine  psb_d_csrg_reallocate_nz(nz,a) 
      import :: psb_d_csrg_sparse_mat, psb_ipk_
      integer(psb_ipk_), intent(in) :: nz
      class(psb_d_csrg_sparse_mat), intent(inout) :: a
    end subroutine psb_d_csrg_reallocate_nz
  end interface

  interface
    subroutine  psb_d_csrg_allocate_mnnz(m,n,a,nz) 
      import :: psb_d_csrg_sparse_mat, psb_ipk_
      integer(psb_ipk_), intent(in) :: m,n
      class(psb_d_csrg_sparse_mat), intent(inout) :: a
      integer(psb_ipk_), intent(in), optional :: nz
    end subroutine psb_d_csrg_allocate_mnnz
  end interface

  interface 
    subroutine psb_d_csrg_mold(a,b,info) 
      import :: psb_d_csrg_sparse_mat, psb_d_base_sparse_mat, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(in)                 :: a
      class(psb_d_base_sparse_mat), intent(inout), allocatable :: b
      integer(psb_ipk_), intent(out)                           :: info
    end subroutine psb_d_csrg_mold
  end interface

  interface 
    subroutine psb_d_csrg_to_gpu(a,info, nzrm) 
      import :: psb_d_csrg_sparse_mat, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(inout) :: a
      integer(psb_ipk_), intent(out)              :: info
      integer(psb_ipk_), intent(in), optional     :: nzrm
    end subroutine psb_d_csrg_to_gpu
  end interface

  interface 
    subroutine psb_d_csrg_from_gpu(a,info) 
      import :: psb_d_csrg_sparse_mat, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(inout) :: a
      integer(psb_ipk_), intent(out)             :: info
    end subroutine psb_d_csrg_from_gpu
  end interface

  interface 
    subroutine psb_d_cp_csrg_from_coo(a,b,info) 
      import :: psb_d_csrg_sparse_mat, psb_d_coo_sparse_mat, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(inout) :: a
      class(psb_d_coo_sparse_mat), intent(in)     :: b
      integer(psb_ipk_), intent(out)              :: info
    end subroutine psb_d_cp_csrg_from_coo
  end interface
  
  interface 
    subroutine psb_d_cp_csrg_from_fmt(a,b,info) 
      import :: psb_d_csrg_sparse_mat, psb_d_base_sparse_mat, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(inout) :: a
      class(psb_d_base_sparse_mat), intent(in)    :: b
      integer(psb_ipk_), intent(out)              :: info
    end subroutine psb_d_cp_csrg_from_fmt
  end interface
  
  interface 
    subroutine psb_d_mv_csrg_from_coo(a,b,info) 
      import :: psb_d_csrg_sparse_mat, psb_d_coo_sparse_mat, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(inout) :: a
      class(psb_d_coo_sparse_mat), intent(inout)  :: b
      integer(psb_ipk_), intent(out)              :: info
    end subroutine psb_d_mv_csrg_from_coo
  end interface
  
  interface 
    subroutine psb_d_mv_csrg_from_fmt(a,b,info) 
      import :: psb_d_csrg_sparse_mat, psb_d_base_sparse_mat, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(inout) :: a
      class(psb_d_base_sparse_mat), intent(inout) :: b
      integer(psb_ipk_), intent(out)              :: info
    end subroutine psb_d_mv_csrg_from_fmt
  end interface
  
  interface 
    subroutine psb_d_csrg_csmv(alpha,a,x,beta,y,info,trans) 
      import :: psb_d_csrg_sparse_mat, psb_dpk_, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(in) :: a
      real(psb_dpk_), intent(in)          :: alpha, beta, x(:)
      real(psb_dpk_), intent(inout)       :: y(:)
      integer(psb_ipk_), intent(out)      :: info
      character, optional, intent(in)     :: trans
    end subroutine psb_d_csrg_csmv
  end interface
  interface 
    subroutine psb_d_csrg_csmm(alpha,a,x,beta,y,info,trans) 
      import :: psb_d_csrg_sparse_mat, psb_dpk_, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(in) :: a
      real(psb_dpk_), intent(in)          :: alpha, beta, x(:,:)
      real(psb_dpk_), intent(inout)       :: y(:,:)
      integer(psb_ipk_), intent(out)      :: info
      character, optional, intent(in)     :: trans
    end subroutine psb_d_csrg_csmm
  end interface
  
  interface 
    subroutine psb_d_csrg_scal(d,a,info,side) 
      import :: psb_d_csrg_sparse_mat, psb_dpk_, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(inout) :: a
      real(psb_dpk_), intent(in)      :: d(:)
      integer(psb_ipk_), intent(out)  :: info
      character, intent(in), optional :: side
    end subroutine psb_d_csrg_scal
  end interface
  
  interface
    subroutine psb_d_csrg_scals(d,a,info) 
      import :: psb_d_csrg_sparse_mat, psb_dpk_, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(inout) :: a
      real(psb_dpk_), intent(in)      :: d
      integer(psb_ipk_), intent(out)   :: info
    end subroutine psb_d_csrg_scals
  end interface
  

contains 

  ! == ===================================
  !
  !
  !
  ! Getters 
  !
  !
  !
  !
  !
  ! == ===================================

  
  function d_csrg_sizeof(a) result(res)
    implicit none 
    class(psb_d_csrg_sparse_mat), intent(in) :: a
    integer(psb_epk_) :: res
    if (a%is_dev()) call a%sync()
    res = 8 
    res = res + psb_sizeof_dp  * size(a%val)
    res = res + psb_sizeof_ip * size(a%irp)
    res = res + psb_sizeof_ip * size(a%ja)
    ! Should we account for the shadow data structure
    ! on the GPU device side? 
    ! res = 2*res
      
  end function d_csrg_sizeof

  function d_csrg_get_fmt() result(res)
    implicit none 
    character(len=5) :: res
    res = 'CSRG'
  end function d_csrg_get_fmt
  


  ! == ===================================
  !
  !
  !
  ! Data management
  !
  !
  !
  !
  !
  ! == ===================================  


  subroutine d_csrg_set_host(a)
    implicit none 
    class(psb_d_csrg_sparse_mat), intent(inout) :: a
    
    a%devstate = is_host
  end subroutine d_csrg_set_host

  subroutine d_csrg_set_dev(a)
    implicit none 
    class(psb_d_csrg_sparse_mat), intent(inout) :: a
    
    a%devstate = is_dev
  end subroutine d_csrg_set_dev

  subroutine d_csrg_set_sync(a)
    implicit none 
    class(psb_d_csrg_sparse_mat), intent(inout) :: a
    
    a%devstate = is_sync
  end subroutine d_csrg_set_sync

  function d_csrg_is_dev(a) result(res)
    implicit none 
    class(psb_d_csrg_sparse_mat), intent(in) :: a
    logical  :: res
  
    res = (a%devstate == is_dev)
  end function d_csrg_is_dev
  
  function d_csrg_is_host(a) result(res)
    implicit none 
    class(psb_d_csrg_sparse_mat), intent(in) :: a
    logical  :: res

    res = (a%devstate == is_host)
  end function d_csrg_is_host

  function d_csrg_is_sync(a) result(res)
    implicit none 
    class(psb_d_csrg_sparse_mat), intent(in) :: a
    logical  :: res

    res = (a%devstate == is_sync)
  end function d_csrg_is_sync


  subroutine  d_csrg_sync(a) 
    implicit none 
    class(psb_d_csrg_sparse_mat), target, intent(in) :: a
    class(psb_d_csrg_sparse_mat), pointer :: tmpa
    integer(psb_ipk_) :: info

    tmpa => a
    if (tmpa%is_host()) then 
      call tmpa%to_gpu(info)
    else if (tmpa%is_dev()) then 
      call tmpa%from_gpu(info)
    end if
    call tmpa%set_sync()
    return

  end subroutine d_csrg_sync

  subroutine  d_csrg_free(a) 
    use cusparse_mod
    implicit none 
    integer(psb_ipk_) :: info

    class(psb_d_csrg_sparse_mat), intent(inout) :: a

    info = CSRGDeviceFree(a%deviceMat)
    call a%psb_d_csr_sparse_mat%free()
    
    return

  end subroutine d_csrg_free

  subroutine  d_csrg_finalize(a) 
    use cusparse_mod
    implicit none 
    integer(psb_ipk_) :: info
    
    type(psb_d_csrg_sparse_mat), intent(inout) :: a

    info = CSRGDeviceFree(a%deviceMat)
    
    return

  end subroutine d_csrg_finalize

#else 
  interface 
    subroutine psb_d_csrg_mold(a,b,info) 
      import :: psb_d_csrg_sparse_mat, psb_d_base_sparse_mat, psb_ipk_
      class(psb_d_csrg_sparse_mat), intent(in)               :: a
      class(psb_d_base_sparse_mat), intent(inout), allocatable :: b
      integer(psb_ipk_), intent(out)                         :: info
    end subroutine psb_d_csrg_mold
  end interface

#endif

end module psb_d_csrg_mat_mod
