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
  

module psb_d_hdiag_mat_mod

  use iso_c_binding
  use psb_base_mod
  use psb_d_hdia_mat_mod

  type, extends(psb_d_hdia_sparse_mat) :: psb_d_hdiag_sparse_mat
    !
#ifdef HAVE_SPGPU
    type(c_ptr) :: deviceMat = c_null_ptr

  contains
    procedure, nopass  :: get_fmt       => d_hdiag_get_fmt
    !  procedure, pass(a) :: sizeof        => d_hdiag_sizeof
    procedure, pass(a) :: vect_mv       => psb_d_hdiag_vect_mv
    ! procedure, pass(a) :: csmm          => psb_d_hdiag_csmm
    procedure, pass(a) :: csmv          => psb_d_hdiag_csmv
    !    procedure, pass(a) :: in_vect_sv    => psb_d_hdiag_inner_vect_sv
    !    procedure, pass(a) :: scals         => psb_d_hdiag_scals
    !    procedure, pass(a) :: scalv         => psb_d_hdiag_scal
    !    procedure, pass(a) :: reallocate_nz => psb_d_hdiag_reallocate_nz
    !    procedure, pass(a) :: allocate_mnnz => psb_d_hdiag_allocate_mnnz
    ! Note: we do *not* need the TO methods, because the parent type
    ! methods will work. 
    procedure, pass(a) :: cp_from_coo   => psb_d_cp_hdiag_from_coo
    !    procedure, pass(a) :: cp_from_fmt   => psb_d_cp_hdiag_from_fmt
    procedure, pass(a) :: mv_from_coo   => psb_d_mv_hdiag_from_coo
    !    procedure, pass(a) :: mv_from_fmt   => psb_d_mv_hdiag_from_fmt
    procedure, pass(a) :: free          => d_hdiag_free
    procedure, pass(a) :: mold          => psb_d_hdiag_mold
    procedure, pass(a) :: to_gpu        => psb_d_hdiag_to_gpu
    final              :: d_hdiag_finalize
#else 
  contains
    procedure, pass(a) :: mold         => psb_d_hdiag_mold
#endif
  end type psb_d_hdiag_sparse_mat

#ifdef HAVE_SPGPU
  private :: d_hdiag_get_nzeros, d_hdiag_free,  d_hdiag_get_fmt, &
       & d_hdiag_get_size, d_hdiag_sizeof, d_hdiag_get_nz_row


  interface 
    subroutine psb_d_hdiag_vect_mv(alpha,a,x,beta,y,info,trans) 
      import :: psb_d_hdiag_sparse_mat, psb_dpk_, psb_d_base_vect_type, psb_ipk_
      class(psb_d_hdiag_sparse_mat), intent(in)    :: a
      real(psb_dpk_), intent(in)                 :: alpha, beta
      class(psb_d_base_vect_type), intent(inout) :: x
      class(psb_d_base_vect_type), intent(inout) :: y
      integer(psb_ipk_), intent(out)             :: info
      character, optional, intent(in)            :: trans
    end subroutine psb_d_hdiag_vect_mv
  end interface

!!$  interface 
!!$    subroutine psb_d_hdiag_inner_vect_sv(alpha,a,x,beta,y,info,trans) 
!!$      import :: psb_ipk_, psb_d_hdiag_sparse_mat, psb_dpk_,  psb_d_base_vect_type
!!$      class(psb_d_hdiag_sparse_mat), intent(in)    :: a
!!$      real(psb_dpk_), intent(in)                 :: alpha, beta
!!$      class(psb_d_base_vect_type), intent(inout) :: x, y
!!$      integer(psb_ipk_), intent(out)             :: info
!!$      character, optional, intent(in)            :: trans
!!$    end subroutine psb_d_hdiag_inner_vect_sv
!!$  end interface
!!$
!!$  interface
!!$    subroutine  psb_d_hdiag_reallocate_nz(nz,a) 
!!$      import :: psb_d_hdiag_sparse_mat, psb_ipk_
!!$      integer(psb_ipk_), intent(in)              :: nz
!!$      class(psb_d_hdiag_sparse_mat), intent(inout) :: a
!!$    end subroutine psb_d_hdiag_reallocate_nz
!!$  end interface
!!$
!!$  interface
!!$    subroutine  psb_d_hdiag_allocate_mnnz(m,n,a,nz) 
!!$      import :: psb_d_hdiag_sparse_mat, psb_ipk_
!!$      integer(psb_ipk_), intent(in)              :: m,n
!!$      class(psb_d_hdiag_sparse_mat), intent(inout) :: a
!!$      integer(psb_ipk_), intent(in), optional    :: nz
!!$    end subroutine psb_d_hdiag_allocate_mnnz
!!$  end interface

  interface 
    subroutine psb_d_hdiag_mold(a,b,info) 
      import :: psb_d_hdiag_sparse_mat, psb_d_base_sparse_mat, psb_ipk_
      class(psb_d_hdiag_sparse_mat), intent(in)                  :: a
      class(psb_d_base_sparse_mat), intent(inout), allocatable :: b
      integer(psb_ipk_), intent(out)                           :: info
    end subroutine psb_d_hdiag_mold
  end interface

  interface 
    subroutine psb_d_hdiag_to_gpu(a,info) 
      import :: psb_d_hdiag_sparse_mat, psb_ipk_
      class(psb_d_hdiag_sparse_mat), intent(inout) :: a
      integer(psb_ipk_), intent(out)             :: info
    end subroutine psb_d_hdiag_to_gpu
  end interface

  interface 
    subroutine psb_d_cp_hdiag_from_coo(a,b,info) 
      import :: psb_d_hdiag_sparse_mat, psb_d_coo_sparse_mat, psb_ipk_
      class(psb_d_hdiag_sparse_mat), intent(inout) :: a
      class(psb_d_coo_sparse_mat), intent(in)    :: b
      integer(psb_ipk_), intent(out)             :: info
    end subroutine psb_d_cp_hdiag_from_coo
  end interface
  
!!$  interface 
!!$    subroutine psb_d_cp_hdiag_from_fmt(a,b,info) 
!!$      import :: psb_d_hdiag_sparse_mat, psb_d_base_sparse_mat, psb_ipk_
!!$      class(psb_d_hdiag_sparse_mat), intent(inout) :: a
!!$      class(psb_d_base_sparse_mat), intent(in)   :: b
!!$      integer(psb_ipk_), intent(out)             :: info
!!$    end subroutine psb_d_cp_hdiag_from_fmt
!!$  end interface
!!$  
  interface 
    subroutine psb_d_mv_hdiag_from_coo(a,b,info) 
      import :: psb_d_hdiag_sparse_mat, psb_d_coo_sparse_mat, psb_ipk_
      class(psb_d_hdiag_sparse_mat), intent(inout) :: a
      class(psb_d_coo_sparse_mat), intent(inout) :: b
      integer(psb_ipk_), intent(out)             :: info
    end subroutine psb_d_mv_hdiag_from_coo
  end interface
  
!!$
!!$  interface 
!!$    subroutine psb_d_mv_hdiag_from_fmt(a,b,info) 
!!$      import :: psb_d_hdiag_sparse_mat, psb_d_base_sparse_mat, psb_ipk_
!!$      class(psb_d_hdiag_sparse_mat), intent(inout)  :: a
!!$      class(psb_d_base_sparse_mat), intent(inout) :: b
!!$      integer(psb_ipk_), intent(out)              :: info
!!$    end subroutine psb_d_mv_hdiag_from_fmt
!!$  end interface
!!$  
  interface 
    subroutine psb_d_hdiag_csmv(alpha,a,x,beta,y,info,trans) 
      import :: psb_d_hdiag_sparse_mat, psb_dpk_, psb_ipk_
      class(psb_d_hdiag_sparse_mat), intent(in) :: a
      real(psb_dpk_), intent(in)              :: alpha, beta, x(:)
      real(psb_dpk_), intent(inout)           :: y(:)
      integer(psb_ipk_), intent(out)          :: info
      character, optional, intent(in)         :: trans
    end subroutine psb_d_hdiag_csmv
  end interface

!!$  interface 
!!$    subroutine psb_d_hdiag_csmm(alpha,a,x,beta,y,info,trans) 
!!$      import :: psb_d_hdiag_sparse_mat, psb_dpk_, psb_ipk_
!!$      class(psb_d_hdiag_sparse_mat), intent(in) :: a
!!$      real(psb_dpk_), intent(in)              :: alpha, beta, x(:,:)
!!$      real(psb_dpk_), intent(inout)           :: y(:,:)
!!$      integer(psb_ipk_), intent(out)          :: info
!!$      character, optional, intent(in)         :: trans
!!$    end subroutine psb_d_hdiag_csmm
!!$  end interface
!!$  
!!$  interface 
!!$    subroutine psb_d_hdiag_scal(d,a,info, side) 
!!$      import :: psb_d_hdiag_sparse_mat, psb_dpk_, psb_ipk_
!!$      class(psb_d_hdiag_sparse_mat), intent(inout) :: a
!!$      real(psb_dpk_), intent(in)                 :: d(:)
!!$      integer(psb_ipk_), intent(out)             :: info
!!$      character, intent(in), optional            :: side
!!$    end subroutine psb_d_hdiag_scal
!!$  end interface
!!$  
!!$  interface
!!$    subroutine psb_d_hdiag_scals(d,a,info) 
!!$      import :: psb_d_hdiag_sparse_mat, psb_dpk_, psb_ipk_
!!$      class(psb_d_hdiag_sparse_mat), intent(inout) :: a
!!$      real(psb_dpk_), intent(in)                 :: d
!!$      integer(psb_ipk_), intent(out)             :: info
!!$    end subroutine psb_d_hdiag_scals
!!$  end interface
!!$  

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

  function d_hdiag_get_fmt() result(res)
    implicit none 
    character(len=5) :: res
    res = 'HDIAG'
  end function d_hdiag_get_fmt
  


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

  subroutine  d_hdiag_free(a) 
    use hdiagdev_mod
    implicit none 
    integer(psb_ipk_) :: info
    class(psb_d_hdiag_sparse_mat), intent(inout) :: a

    if (c_associated(a%deviceMat)) &
         & call freeHdiagDevice(a%deviceMat)
    a%deviceMat = c_null_ptr
    call a%psb_d_hdia_sparse_mat%free()
    
    return

  end subroutine d_hdiag_free

  subroutine  d_hdiag_finalize(a) 
    use hdiagdev_mod
    implicit none 
    type(psb_d_hdiag_sparse_mat), intent(inout) :: a

    if (c_associated(a%deviceMat)) &
         &  call freeHdiagDevice(a%deviceMat)
    a%deviceMat = c_null_ptr
    call a%psb_d_hdia_sparse_mat%free()
    
    return
  end subroutine d_hdiag_finalize

#else 

  interface 
    subroutine psb_d_hdiag_mold(a,b,info) 
      import :: psb_d_hdiag_sparse_mat, psb_d_base_sparse_mat, psb_ipk_
      class(psb_d_hdiag_sparse_mat), intent(in)                :: a
      class(psb_d_base_sparse_mat), intent(inout), allocatable :: b
      integer(psb_ipk_), intent(out)                         :: info
    end subroutine psb_d_hdiag_mold
  end interface

#endif

end module psb_d_hdiag_mat_mod
