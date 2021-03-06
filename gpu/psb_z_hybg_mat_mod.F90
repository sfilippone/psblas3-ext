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
  
#if CUDA_SHORT_VERSION <= 10 

module psb_z_hybg_mat_mod

  use iso_c_binding
  use psb_z_mat_mod
  use cusparse_mod

  type, extends(psb_z_csr_sparse_mat) :: psb_z_hybg_sparse_mat
    !
    ! HYBG. An interface to the cuSPARSE HYB
    ! On the CPU side we keep a CSR storage. 
    ! 
    ! 
    ! 
    ! 
#ifdef HAVE_SPGPU
    type(z_Hmat) :: deviceMat

  contains
    procedure, nopass  :: get_fmt       => z_hybg_get_fmt
    procedure, pass(a) :: sizeof        => z_hybg_sizeof
    procedure, pass(a) :: vect_mv       => psb_z_hybg_vect_mv
    procedure, pass(a) :: in_vect_sv    => psb_z_hybg_inner_vect_sv
    procedure, pass(a) :: csmm          => psb_z_hybg_csmm
    procedure, pass(a) :: csmv          => psb_z_hybg_csmv
    procedure, pass(a) :: scals         => psb_z_hybg_scals
    procedure, pass(a) :: scalv         => psb_z_hybg_scal
    procedure, pass(a) :: reallocate_nz => psb_z_hybg_reallocate_nz
    procedure, pass(a) :: allocate_mnnz => psb_z_hybg_allocate_mnnz
    ! Note: we do *not* need the TO methods, because the parent type
    ! methods will work. 
    procedure, pass(a) :: cp_from_coo   => psb_z_cp_hybg_from_coo
    procedure, pass(a) :: cp_from_fmt   => psb_z_cp_hybg_from_fmt
    procedure, pass(a) :: mv_from_coo   => psb_z_mv_hybg_from_coo
    procedure, pass(a) :: mv_from_fmt   => psb_z_mv_hybg_from_fmt
    procedure, pass(a) :: free          => z_hybg_free
    procedure, pass(a) :: mold          => psb_z_hybg_mold
    procedure, pass(a) :: to_gpu        => psb_z_hybg_to_gpu
    final              :: z_hybg_finalize
#else 
  contains
    procedure, pass(a) :: mold    => psb_z_hybg_mold
#endif
  end type psb_z_hybg_sparse_mat

#ifdef HAVE_SPGPU
  private :: z_hybg_get_nzeros, z_hybg_free,  z_hybg_get_fmt, &
       & z_hybg_get_size, z_hybg_sizeof, z_hybg_get_nz_row


  interface 
    subroutine psb_z_hybg_inner_vect_sv(alpha,a,x,beta,y,info,trans) 
      import :: psb_z_hybg_sparse_mat, psb_dpk_, psb_z_base_vect_type, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(in)   :: a
      complex(psb_dpk_), intent(in)                 :: alpha, beta
      class(psb_z_base_vect_type), intent(inout) :: x
      class(psb_z_base_vect_type), intent(inout) :: y
      integer(psb_ipk_), intent(out)             :: info
      character, optional, intent(in)            :: trans
    end subroutine psb_z_hybg_inner_vect_sv
  end interface

  interface 
    subroutine psb_z_hybg_vect_mv(alpha,a,x,beta,y,info,trans) 
      import :: psb_z_hybg_sparse_mat, psb_dpk_, psb_z_base_vect_type, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(in)   :: a
      complex(psb_dpk_), intent(in)                 :: alpha, beta
      class(psb_z_base_vect_type), intent(inout) :: x
      class(psb_z_base_vect_type), intent(inout) :: y
      integer(psb_ipk_), intent(out)             :: info
      character, optional, intent(in)            :: trans
    end subroutine psb_z_hybg_vect_mv
  end interface

  interface
    subroutine  psb_z_hybg_reallocate_nz(nz,a) 
      import :: psb_z_hybg_sparse_mat, psb_ipk_
      integer(psb_ipk_), intent(in)               :: nz
      class(psb_z_hybg_sparse_mat), intent(inout) :: a
    end subroutine psb_z_hybg_reallocate_nz
  end interface

  interface
    subroutine  psb_z_hybg_allocate_mnnz(m,n,a,nz) 
      import :: psb_z_hybg_sparse_mat, psb_ipk_
      integer(psb_ipk_), intent(in)               :: m,n
      class(psb_z_hybg_sparse_mat), intent(inout) :: a
      integer(psb_ipk_), intent(in), optional     :: nz
    end subroutine psb_z_hybg_allocate_mnnz
  end interface

  interface 
    subroutine psb_z_hybg_mold(a,b,info) 
      import :: psb_z_hybg_sparse_mat, psb_z_base_sparse_mat, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(in)                 :: a
      class(psb_z_base_sparse_mat), intent(inout), allocatable :: b
      integer(psb_ipk_), intent(out)                           :: info
    end subroutine psb_z_hybg_mold
  end interface

  interface 
    subroutine psb_z_hybg_to_gpu(a,info, nzrm) 
      import :: psb_z_hybg_sparse_mat, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(inout) :: a
      integer(psb_ipk_), intent(out)              :: info
      integer(psb_ipk_), intent(in), optional     :: nzrm
    end subroutine psb_z_hybg_to_gpu
  end interface

  interface 
    subroutine psb_z_cp_hybg_from_coo(a,b,info) 
      import :: psb_z_hybg_sparse_mat, psb_z_coo_sparse_mat, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(inout) :: a
      class(psb_z_coo_sparse_mat), intent(in)     :: b
      integer(psb_ipk_), intent(out)              :: info
    end subroutine psb_z_cp_hybg_from_coo
  end interface
  
  interface 
    subroutine psb_z_cp_hybg_from_fmt(a,b,info) 
      import :: psb_z_hybg_sparse_mat, psb_z_base_sparse_mat, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(inout) :: a
      class(psb_z_base_sparse_mat), intent(in)    :: b
      integer(psb_ipk_), intent(out)              :: info
    end subroutine psb_z_cp_hybg_from_fmt
  end interface
  
  interface 
    subroutine psb_z_mv_hybg_from_coo(a,b,info) 
      import :: psb_z_hybg_sparse_mat, psb_z_coo_sparse_mat, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(inout) :: a
      class(psb_z_coo_sparse_mat), intent(inout)  :: b
      integer(psb_ipk_), intent(out)              :: info
    end subroutine psb_z_mv_hybg_from_coo
  end interface

  interface 
    subroutine psb_z_mv_hybg_from_fmt(a,b,info) 
      import :: psb_z_hybg_sparse_mat, psb_z_base_sparse_mat, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(inout) :: a
      class(psb_z_base_sparse_mat), intent(inout) :: b
      integer(psb_ipk_), intent(out)              :: info
    end subroutine psb_z_mv_hybg_from_fmt
  end interface

  interface 
    subroutine psb_z_hybg_csmv(alpha,a,x,beta,y,info,trans) 
      import :: psb_z_hybg_sparse_mat, psb_dpk_, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(in) :: a
      complex(psb_dpk_), intent(in)               :: alpha, beta, x(:)
      complex(psb_dpk_), intent(inout)            :: y(:)
      integer(psb_ipk_), intent(out)           :: info
      character, optional, intent(in)          :: trans
    end subroutine psb_z_hybg_csmv
  end interface
  interface 
    subroutine psb_z_hybg_csmm(alpha,a,x,beta,y,info,trans) 
      import :: psb_z_hybg_sparse_mat, psb_dpk_, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(in) :: a
      complex(psb_dpk_), intent(in)               :: alpha, beta, x(:,:)
      complex(psb_dpk_), intent(inout)            :: y(:,:)
      integer(psb_ipk_), intent(out)           :: info
      character, optional, intent(in)          :: trans
    end subroutine psb_z_hybg_csmm
  end interface
  
  interface 
    subroutine psb_z_hybg_scal(d,a,info,side) 
      import :: psb_z_hybg_sparse_mat, psb_dpk_, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(inout) :: a
      complex(psb_dpk_), intent(in)                  :: d(:)
      integer(psb_ipk_), intent(out)              :: info
      character, intent(in), optional             :: side
    end subroutine psb_z_hybg_scal
  end interface
  
  interface
    subroutine psb_z_hybg_scals(d,a,info) 
      import :: psb_z_hybg_sparse_mat, psb_dpk_, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(inout) :: a
      complex(psb_dpk_), intent(in)                  :: d
      integer(psb_ipk_), intent(out)              :: info
    end subroutine psb_z_hybg_scals 
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

  
  function z_hybg_sizeof(a) result(res)
    implicit none 
    class(psb_z_hybg_sparse_mat), intent(in) :: a
    integer(psb_epk_)                 :: res
    res = 8 
    res = res + (2*psb_sizeof_dp)  * size(a%val)
    res = res + psb_sizeof_ip * size(a%irp)
    res = res + psb_sizeof_ip * size(a%ja)
    ! Should we account for the shadow data structure
    ! on the GPU device side? 
    ! res = 2*res
      
  end function z_hybg_sizeof

  function z_hybg_get_fmt() result(res)
    implicit none 
    character(len=5) :: res
    res = 'HYBG'
  end function z_hybg_get_fmt
  


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

  subroutine  z_hybg_free(a) 
    use cusparse_mod
    implicit none 
    integer(psb_ipk_)                           :: info
    class(psb_z_hybg_sparse_mat), intent(inout) :: a

    info = HYBGDeviceFree(a%deviceMat)
    call a%psb_z_csr_sparse_mat%free()
    
    return

  end subroutine z_hybg_free

  subroutine  z_hybg_finalize(a) 
    use cusparse_mod
    implicit none 
    integer(psb_ipk_)                           :: info
    type(psb_z_hybg_sparse_mat), intent(inout) :: a

    info = HYBGDeviceFree(a%deviceMat)
    
    return
  end subroutine z_hybg_finalize

#else 

  interface 
    subroutine psb_z_hybg_mold(a,b,info) 
      import :: psb_z_hybg_sparse_mat, psb_z_base_sparse_mat, psb_ipk_
      class(psb_z_hybg_sparse_mat), intent(in)               :: a
      class(psb_z_base_sparse_mat), intent(inout), allocatable :: b
      integer(psb_ipk_), intent(out)                         :: info
    end subroutine psb_z_hybg_mold
  end interface

#endif

end module psb_z_hybg_mat_mod
#endif
