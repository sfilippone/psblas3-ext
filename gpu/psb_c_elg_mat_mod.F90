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
  

module psb_c_elg_mat_mod

  use iso_c_binding
  use psb_c_mat_mod
  use psb_c_ell_mat_mod
  use psb_i_gpu_vect_mod

  integer(psb_ipk_), parameter, private :: is_host = -1
  integer(psb_ipk_), parameter, private :: is_sync = 0 
  integer(psb_ipk_), parameter, private :: is_dev  = 1 

  type, extends(psb_c_ell_sparse_mat) :: psb_c_elg_sparse_mat
    !
    ! ITPACK/ELL format, extended.
    ! We are adding here the routines to create a copy of the data
    ! into the GPU. 
    ! If HAVE_SPGPU is undefined this is just
    ! a copy of ELL, indistinguishable.
    ! 
#ifdef HAVE_SPGPU
    type(c_ptr)       :: deviceMat = c_null_ptr
    integer(psb_ipk_) :: devstate  = is_host

  contains
    procedure, nopass  :: get_fmt       => c_elg_get_fmt
    procedure, pass(a) :: sizeof        => c_elg_sizeof
    procedure, pass(a) :: vect_mv       => psb_c_elg_vect_mv
    procedure, pass(a) :: csmm          => psb_c_elg_csmm
    procedure, pass(a) :: csmv          => psb_c_elg_csmv
    procedure, pass(a) :: in_vect_sv    => psb_c_elg_inner_vect_sv
    procedure, pass(a) :: scals         => psb_c_elg_scals
    procedure, pass(a) :: scalv         => psb_c_elg_scal
    procedure, pass(a) :: reallocate_nz => psb_c_elg_reallocate_nz
    procedure, pass(a) :: allocate_mnnz => psb_c_elg_allocate_mnnz
    procedure, pass(a) :: reinit        => c_elg_reinit
    ! Note: we do *not* need the TO methods, because the parent type
    ! methods will work. 
    procedure, pass(a) :: cp_from_coo   => psb_c_cp_elg_from_coo
    procedure, pass(a) :: cp_from_fmt   => psb_c_cp_elg_from_fmt
    procedure, pass(a) :: mv_from_coo   => psb_c_mv_elg_from_coo
    procedure, pass(a) :: mv_from_fmt   => psb_c_mv_elg_from_fmt
    procedure, pass(a) :: free          => c_elg_free
    procedure, pass(a) :: mold          => psb_c_elg_mold
    procedure, pass(a) :: csput_a       => psb_c_elg_csput_a
    procedure, pass(a) :: csput_v       => psb_c_elg_csput_v
    procedure, pass(a) :: is_host       => c_elg_is_host
    procedure, pass(a) :: is_dev        => c_elg_is_dev
    procedure, pass(a) :: is_sync       => c_elg_is_sync
    procedure, pass(a) :: set_host      => c_elg_set_host
    procedure, pass(a) :: set_dev       => c_elg_set_dev
    procedure, pass(a) :: set_sync      => c_elg_set_sync
    procedure, pass(a) :: sync          => c_elg_sync
    procedure, pass(a) :: from_gpu      => psb_c_elg_from_gpu
    procedure, pass(a) :: to_gpu        => psb_c_elg_to_gpu
    procedure, pass(a) :: asb           => psb_c_elg_asb
    final              :: c_elg_finalize
#else 
  contains
    procedure, pass(a) :: mold         => psb_c_elg_mold
    procedure, pass(a) :: asb          => psb_c_elg_asb
#endif
  end type psb_c_elg_sparse_mat

#ifdef HAVE_SPGPU
  private :: c_elg_get_nzeros, c_elg_free,  c_elg_get_fmt, &
       & c_elg_get_size, c_elg_sizeof, c_elg_get_nz_row, c_elg_sync


  interface 
    subroutine psb_c_elg_vect_mv(alpha,a,x,beta,y,info,trans) 
      import :: psb_c_elg_sparse_mat, psb_spk_, psb_c_base_vect_type, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(in) :: a
      complex(psb_spk_), intent(in)       :: alpha, beta
      class(psb_c_base_vect_type), intent(inout) :: x
      class(psb_c_base_vect_type), intent(inout) :: y
      integer(psb_ipk_), intent(out)             :: info
      character, optional, intent(in)  :: trans
    end subroutine psb_c_elg_vect_mv
  end interface

  interface 
    subroutine psb_c_elg_inner_vect_sv(alpha,a,x,beta,y,info,trans) 
      import :: psb_ipk_, psb_c_elg_sparse_mat, psb_spk_,  psb_c_base_vect_type
      class(psb_c_elg_sparse_mat), intent(in) :: a
      complex(psb_spk_), intent(in)       :: alpha, beta
      class(psb_c_base_vect_type), intent(inout) :: x, y
      integer(psb_ipk_), intent(out)             :: info
      character, optional, intent(in)  :: trans
    end subroutine psb_c_elg_inner_vect_sv
  end interface

  interface
    subroutine  psb_c_elg_reallocate_nz(nz,a) 
      import :: psb_c_elg_sparse_mat, psb_ipk_
      integer(psb_ipk_), intent(in) :: nz
      class(psb_c_elg_sparse_mat), intent(inout) :: a
    end subroutine psb_c_elg_reallocate_nz
  end interface

  interface
    subroutine  psb_c_elg_allocate_mnnz(m,n,a,nz) 
      import :: psb_c_elg_sparse_mat, psb_ipk_
      integer(psb_ipk_), intent(in) :: m,n
      class(psb_c_elg_sparse_mat), intent(inout) :: a
      integer(psb_ipk_), intent(in), optional :: nz
    end subroutine psb_c_elg_allocate_mnnz
  end interface

  interface 
    subroutine psb_c_elg_mold(a,b,info) 
      import :: psb_c_elg_sparse_mat, psb_c_base_sparse_mat, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(in)                  :: a
      class(psb_c_base_sparse_mat), intent(inout), allocatable :: b
      integer(psb_ipk_), intent(out)                           :: info
    end subroutine psb_c_elg_mold
  end interface
  
  interface 
    subroutine psb_c_elg_csput_a(nz,ia,ja,val,a,imin,imax,jmin,jmax,info) 
      import :: psb_c_elg_sparse_mat, psb_spk_, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(inout) :: a
      complex(psb_spk_), intent(in)      :: val(:)
      integer(psb_ipk_), intent(in)             :: nz,ia(:), ja(:),&
           &  imin,imax,jmin,jmax
      integer(psb_ipk_), intent(out)            :: info
    end subroutine psb_c_elg_csput_a
  end interface

  interface 
    subroutine psb_c_elg_csput_v(nz,ia,ja,val,a,imin,imax,jmin,jmax,info) 
      import :: psb_c_elg_sparse_mat, psb_dpk_, psb_ipk_, psb_c_base_vect_type,&
           & psb_i_base_vect_type
      class(psb_c_elg_sparse_mat), intent(inout) :: a
      class(psb_c_base_vect_type), intent(inout) :: val
      class(psb_i_base_vect_type), intent(inout) :: ia, ja
      integer(psb_ipk_), intent(in)              :: nz, imin,imax,jmin,jmax
      integer(psb_ipk_), intent(out)             :: info
    end subroutine psb_c_elg_csput_v
  end interface

  interface 
    subroutine psb_c_elg_from_gpu(a,info) 
      import :: psb_c_elg_sparse_mat, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(inout) :: a
      integer(psb_ipk_), intent(out)             :: info
    end subroutine psb_c_elg_from_gpu
  end interface

  interface 
    subroutine psb_c_elg_to_gpu(a,info, nzrm) 
      import :: psb_c_elg_sparse_mat, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(inout) :: a
      integer(psb_ipk_), intent(out)             :: info
      integer(psb_ipk_), intent(in), optional    :: nzrm
    end subroutine psb_c_elg_to_gpu
  end interface

  interface 
    subroutine psb_c_cp_elg_from_coo(a,b,info) 
      import :: psb_c_elg_sparse_mat, psb_c_coo_sparse_mat, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(inout) :: a
      class(psb_c_coo_sparse_mat), intent(in)    :: b
      integer(psb_ipk_), intent(out)             :: info
    end subroutine psb_c_cp_elg_from_coo
  end interface
  
  interface 
    subroutine psb_c_cp_elg_from_fmt(a,b,info) 
      import :: psb_c_elg_sparse_mat, psb_c_base_sparse_mat, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(inout) :: a
      class(psb_c_base_sparse_mat), intent(in)   :: b
      integer(psb_ipk_), intent(out)             :: info
    end subroutine psb_c_cp_elg_from_fmt
  end interface
  
  interface 
    subroutine psb_c_mv_elg_from_coo(a,b,info) 
      import :: psb_c_elg_sparse_mat, psb_c_coo_sparse_mat, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(inout) :: a
      class(psb_c_coo_sparse_mat), intent(inout) :: b
      integer(psb_ipk_), intent(out)             :: info
    end subroutine psb_c_mv_elg_from_coo
  end interface
  

  interface 
    subroutine psb_c_mv_elg_from_fmt(a,b,info) 
      import :: psb_c_elg_sparse_mat, psb_c_base_sparse_mat, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(inout)  :: a
      class(psb_c_base_sparse_mat), intent(inout) :: b
      integer(psb_ipk_), intent(out)              :: info
    end subroutine psb_c_mv_elg_from_fmt
  end interface

  interface 
    subroutine psb_c_elg_csmv(alpha,a,x,beta,y,info,trans) 
      import :: psb_c_elg_sparse_mat, psb_spk_, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(in) :: a
      complex(psb_spk_), intent(in)          :: alpha, beta, x(:)
      complex(psb_spk_), intent(inout)       :: y(:)
      integer(psb_ipk_), intent(out)     :: info
      character, optional, intent(in)    :: trans
    end subroutine psb_c_elg_csmv
  end interface
  interface 
    subroutine psb_c_elg_csmm(alpha,a,x,beta,y,info,trans) 
      import :: psb_c_elg_sparse_mat, psb_spk_, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(in) :: a
      complex(psb_spk_), intent(in)          :: alpha, beta, x(:,:)
      complex(psb_spk_), intent(inout)       :: y(:,:)
      integer(psb_ipk_), intent(out)      :: info
      character, optional, intent(in)     :: trans
    end subroutine psb_c_elg_csmm
  end interface
  
  interface 
    subroutine psb_c_elg_scal(d,a,info, side) 
      import :: psb_c_elg_sparse_mat, psb_spk_, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(inout) :: a
      complex(psb_spk_), intent(in)      :: d(:)
      integer(psb_ipk_), intent(out)  :: info
      character, intent(in), optional :: side
    end subroutine psb_c_elg_scal
  end interface
  
  interface
    subroutine psb_c_elg_scals(d,a,info) 
      import :: psb_c_elg_sparse_mat, psb_spk_, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(inout) :: a
      complex(psb_spk_), intent(in)      :: d
      integer(psb_ipk_), intent(out)  :: info
    end subroutine psb_c_elg_scals
  end interface
  
  interface 
    subroutine psb_c_elg_asb(a)
      import :: psb_c_elg_sparse_mat
      class(psb_c_elg_sparse_mat), intent(inout) :: a   
    end subroutine psb_c_elg_asb
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

  
  function c_elg_sizeof(a) result(res)
    implicit none 
    class(psb_c_elg_sparse_mat), intent(in) :: a
    integer(psb_epk_) :: res

    if (a%is_dev()) call a%sync()
    res = 8 
    res = res + (2*psb_sizeof_sp)  * size(a%val)
    res = res + psb_sizeof_ip * size(a%irn)
    res = res + psb_sizeof_ip * size(a%idiag)
    res = res + psb_sizeof_ip * size(a%ja)
    ! Should we account for the shadow data structure
    ! on the GPU device side? 
    ! res = 2*res
      
  end function c_elg_sizeof

  function c_elg_get_fmt() result(res)
    implicit none 
    character(len=5) :: res
    res = 'ELG'
  end function c_elg_get_fmt
  


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
  subroutine  c_elg_reinit(a,clear) 
    use elldev_mod
    implicit none 
    integer(psb_ipk_) :: info

    class(psb_c_elg_sparse_mat), intent(inout) :: a
    logical, intent(in), optional :: clear
    integer(psb_ipk_) :: isz, err_act
    character(len=20)  :: name='reinit'
    logical            :: clear_
    logical, parameter :: debug=.false.

    call psb_erractionsave(err_act)
    info = psb_success_

    if (present(clear)) then 
      clear_ = clear
    else
      clear_ = .true.
    end if

    if (a%is_bld() .or. a%is_upd()) then 
      ! do nothing
      return
    else if (a%is_asb()) then
      if (a%is_dev().or.a%is_sync()) then
        if (clear_) call zeroEllDevice(a%deviceMat)
        call a%set_dev()
      else if (a%is_host()) then 
        a%val(:,:) = czero
      end if
      call a%set_upd()
    else
      info = psb_err_invalid_mat_state_
      call psb_errpush(info,name)
      goto 9999
    end if

    call psb_erractionrestore(err_act)
    return

9999 call psb_error_handler(err_act)
    return

  end subroutine c_elg_reinit

  subroutine  c_elg_free(a) 
    use elldev_mod
    implicit none 
    integer(psb_ipk_) :: info

    class(psb_c_elg_sparse_mat), intent(inout) :: a
    
    if (c_associated(a%deviceMat)) &
         & call freeEllDevice(a%deviceMat)
    a%deviceMat = c_null_ptr
    call a%psb_c_ell_sparse_mat%free()
    call a%set_sync()
    
    return

  end subroutine c_elg_free

  subroutine  c_elg_sync(a) 
    implicit none 
    class(psb_c_elg_sparse_mat), target, intent(in) :: a
    class(psb_c_elg_sparse_mat), pointer :: tmpa
    integer(psb_ipk_) :: info

    tmpa => a
    if (tmpa%is_host()) then 
      call tmpa%to_gpu(info)
    else if (tmpa%is_dev()) then 
      call tmpa%from_gpu(info)
    end if
    call tmpa%set_sync()
    return

  end subroutine c_elg_sync

  subroutine c_elg_set_host(a)
    implicit none 
    class(psb_c_elg_sparse_mat), intent(inout) :: a
    
    a%devstate = is_host
  end subroutine c_elg_set_host

  subroutine c_elg_set_dev(a)
    implicit none 
    class(psb_c_elg_sparse_mat), intent(inout) :: a
    
    a%devstate = is_dev
  end subroutine c_elg_set_dev

  subroutine c_elg_set_sync(a)
    implicit none 
    class(psb_c_elg_sparse_mat), intent(inout) :: a
    
    a%devstate = is_sync
  end subroutine c_elg_set_sync

  function c_elg_is_dev(a) result(res)
    implicit none 
    class(psb_c_elg_sparse_mat), intent(in) :: a
    logical  :: res
  
    res = (a%devstate == is_dev)
  end function c_elg_is_dev
  
  function c_elg_is_host(a) result(res)
    implicit none 
    class(psb_c_elg_sparse_mat), intent(in) :: a
    logical  :: res

    res = (a%devstate == is_host)
  end function c_elg_is_host

  function c_elg_is_sync(a) result(res)
    implicit none 
    class(psb_c_elg_sparse_mat), intent(in) :: a
    logical  :: res

    res = (a%devstate == is_sync)
  end function c_elg_is_sync

  subroutine  c_elg_finalize(a) 
    use elldev_mod
    implicit none 
    type(psb_c_elg_sparse_mat), intent(inout) :: a

    if (c_associated(a%deviceMat)) &
         & call freeEllDevice(a%deviceMat)
    a%deviceMat = c_null_ptr
    return

  end subroutine c_elg_finalize

#else 

  interface 
    subroutine psb_c_elg_asb(a)
      import :: psb_c_elg_sparse_mat
      class(psb_c_elg_sparse_mat), intent(inout) :: a   
    end subroutine psb_c_elg_asb
  end interface

  interface 
    subroutine psb_c_elg_mold(a,b,info) 
      import :: psb_c_elg_sparse_mat, psb_c_base_sparse_mat, psb_ipk_
      class(psb_c_elg_sparse_mat), intent(in)                :: a
      class(psb_c_base_sparse_mat), intent(inout), allocatable :: b
      integer(psb_ipk_), intent(out)                         :: info
    end subroutine psb_c_elg_mold
  end interface

#endif

end module psb_c_elg_mat_mod
