!!$              Parallel Sparse BLAS   GPU plugin 
!!$    (C) Copyright 2013
!!$
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
  

module psb_i_gpu_vect_mod
  use iso_c_binding
  use psb_const_mod
  use psb_i_vect_mod
#ifdef HAVE_SPGPU
  use psb_i_vectordev_mod
#endif

  integer(psb_ipk_), parameter, private :: is_host = -1
  integer(psb_ipk_), parameter, private :: is_sync = 0 
  integer(psb_ipk_), parameter, private :: is_dev  = 1 
  
  type, extends(psb_i_base_vect_type) ::  psb_i_vect_gpu
#ifdef HAVE_SPGPU
    integer     :: state      = is_host
    type(c_ptr) :: deviceVect = c_null_ptr
  contains
    procedure, pass(y) :: axpby_v  => i_gpu_axpby_v
    procedure, pass(y) :: axpby_a  => i_gpu_axpby_a
    procedure, pass(y) :: mlt_v    => i_gpu_mlt_v
    procedure, pass(y) :: mlt_a    => i_gpu_mlt_a
    procedure, pass(z) :: mlt_a_2  => i_gpu_mlt_a_2
    procedure, pass(z) :: mlt_v_2  => i_gpu_mlt_v_2
    procedure, pass(x) :: scal     => i_gpu_scal
    procedure, pass(x) :: all      => i_gpu_all
    procedure, pass(x) :: zero     => i_gpu_zero
    procedure, pass(x) :: asb      => i_gpu_asb
    procedure, pass(x) :: sync     => i_gpu_sync
    procedure, pass(x) :: sync_space => i_gpu_sync_space
    procedure, pass(x) :: bld_x    => i_gpu_bld_x
    procedure, pass(x) :: bld_n    => i_gpu_bld_n
    procedure, pass(x) :: free     => i_gpu_free
    procedure, pass(x) :: ins_a    => i_gpu_ins_a
    procedure, pass(x) :: is_host  => i_gpu_is_host
    procedure, pass(x) :: is_dev   => i_gpu_is_dev
    procedure, pass(x) :: is_sync  => i_gpu_is_sync
    procedure, pass(x) :: set_host => i_gpu_set_host
    procedure, pass(x) :: set_dev  => i_gpu_set_dev
    procedure, pass(x) :: set_sync => i_gpu_set_sync
    procedure, pass(x) :: set_scal => i_gpu_set_scal
    procedure, pass(x) :: set_vect => i_gpu_set_vect
    procedure, nopass  :: get_fmt   => i_gpu_get_fmt
!!$    procedure, pass(x) :: gthzv    => i_gpu_gthzv
!!$    procedure, pass(y) :: sctb     => i_gpu_sctb
#ifdef HAVE_FINAL
    final              :: i_gpu_vect_finalize
#endif
#endif
  end type psb_i_vect_gpu

  public  :: psb_i_vect_gpu_
  private :: constructor
  interface psb_i_vect_gpu_
    module procedure constructor
  end interface psb_i_vect_gpu_

contains
  
  function constructor(x) result(this)
    real(psb_dpk_)       :: x(:)
    type(psb_i_vect_gpu) :: this
    integer(psb_ipk_) :: info

    this%v = x
    call this%asb(size(x),info)

  end function constructor
    
#ifdef HAVE_SPGPU

  !
  !> Function  base_get_fmt
  !! \memberof  psb_d_base_vect_type
  !! \brief  Format
  !!           
  !
  function i_gpu_get_fmt() result(res)
    implicit none 
    character(len=5) :: res
    res = 'IGPU'
  end function i_gpu_get_fmt
  
  subroutine i_gpu_bld_x(x,this)
    use psb_base_mod
    integer(psb_ipk_), intent(in)           :: this(:)
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: info

    call psb_realloc(size(this),x%v,info)
    if (info /= 0) then 
      info=psb_err_alloc_request_
      call psb_errpush(info,'i_gpu_bld_x',&
           & i_err=(/size(this),izero,izero,izero,izero/))
    end if
    x%v(:)  = this(:) 
    call x%set_host()
    call x%sync()

  end subroutine i_gpu_bld_x

  subroutine i_gpu_bld_n(x,n)
    integer(psb_ipk_), intent(in) :: n
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: info

    call x%all(n,info)
    
  end subroutine i_gpu_bld_n


  subroutine i_gpu_set_host(x)
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    
    x%state = is_host
  end subroutine i_gpu_set_host

  subroutine i_gpu_set_dev(x)
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    
    x%state = is_dev
  end subroutine i_gpu_set_dev

  subroutine i_gpu_set_sync(x)
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    
    x%state = is_sync
  end subroutine i_gpu_set_sync

  function i_gpu_is_dev(x) result(res)
    implicit none 
    class(psb_i_vect_gpu), intent(in) :: x
    logical  :: res
  
    res = (x%state == is_dev)
  end function i_gpu_is_dev
  
  function i_gpu_is_host(x) result(res)
    implicit none 
    class(psb_i_vect_gpu), intent(in) :: x
    logical  :: res

    res = (x%state == is_host)
  end function i_gpu_is_host

  function i_gpu_is_sync(x) result(res)
    implicit none 
    class(psb_i_vect_gpu), intent(in) :: x
    logical  :: res

    res = (x%state == is_sync)
  end function i_gpu_is_sync

  subroutine i_gpu_axpby_v(m,alpha, x, beta, y, info)
    use psi_serial_mod
    implicit none 
    integer(psb_ipk_), intent(in)              :: m
    class(psb_i_base_vect_type), intent(inout) :: x
    class(psb_i_vect_gpu), intent(inout)       :: y
    integer(psb_ipk_), intent (in)                :: alpha, beta
    integer(psb_ipk_), intent(out)             :: info
    integer(psb_ipk_) :: nx, ny

    call x%sync()
    call y%axpby(m,alpha,x%v,beta,info)
    call y%set_host()

  end subroutine i_gpu_axpby_v

  subroutine i_gpu_axpby_a(m,alpha, x, beta, y, info)
    use psi_serial_mod
    implicit none 
    integer(psb_ipk_), intent(in)        :: m
    integer(psb_ipk_), intent(in)           :: x(:)
    class(psb_i_vect_gpu), intent(inout) :: y
    integer(psb_ipk_), intent (in)          :: alpha, beta
    integer(psb_ipk_), intent(out)       :: info

    if (y%is_dev()) call y%sync()
    call psb_geaxpby(m,alpha,x,beta,y%v,info)
    call y%set_host()
  end subroutine i_gpu_axpby_a

  subroutine i_gpu_mlt_v(x, y, info)
    use psi_serial_mod
    implicit none 
    class(psb_i_base_vect_type), intent(inout) :: x
    class(psb_i_vect_gpu), intent(inout)       :: y
    integer(psb_ipk_), intent(out)             :: info

    integer(psb_ipk_) :: i, n
    
    info = 0    
    call x%sync()
    call y%mlt(x%v,info)
    call y%set_host()

  end subroutine i_gpu_mlt_v

  subroutine i_gpu_mlt_a(x, y, info)
    use psi_serial_mod
    implicit none 
    integer(psb_ipk_), intent(in)           :: x(:)
    class(psb_i_vect_gpu), intent(inout) :: y
    integer(psb_ipk_), intent(out)       :: info
    integer(psb_ipk_) :: i, n
    
    info = 0    
    call y%sync()
    call y%psb_i_base_vect_type%mlt(x,info)
    call y%set_host()
  end subroutine i_gpu_mlt_a

  subroutine i_gpu_mlt_a_2(alpha,x,y,beta,z,info)
    use psi_serial_mod
    implicit none 
    integer(psb_ipk_), intent(in)           :: alpha,beta
    integer(psb_ipk_), intent(in)           :: x(:)
    integer(psb_ipk_), intent(in)           :: y(:)
    class(psb_i_vect_gpu), intent(inout) :: z
    integer(psb_ipk_), intent(out)       :: info
    integer(psb_ipk_) :: i, n

    info = 0    
    call z%sync()
    call z%psb_i_base_vect_type%mlt(alpha,x,y,beta,info)
    call z%set_host()
  end subroutine i_gpu_mlt_a_2

  subroutine i_gpu_mlt_v_2(alpha,x,y, beta,z,info,conjgx,conjgy)
    use psi_serial_mod
    use psb_string_mod
    implicit none 
    integer(psb_ipk_), intent(in)                 :: alpha,beta
    class(psb_i_base_vect_type), intent(inout) :: x
    class(psb_i_base_vect_type), intent(inout) :: y
    class(psb_i_vect_gpu), intent(inout)       :: z
    integer(psb_ipk_), intent(out)             :: info
    character(len=1), intent(in), optional     :: conjgx, conjgy
    integer(psb_ipk_) :: i, n
    logical :: conjgx_, conjgy_
    call x%sync()
    call y%sync()
    call z%psb_i_base_vect_type%mlt(alpha,x,y,beta,info)
    call z%set_host()

  end subroutine i_gpu_mlt_v_2


  subroutine i_gpu_set_scal(x,val)
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)           :: val
        
    integer(psb_ipk_) :: info
    call x%sync()
    call x%psb_i_base_vect_type%set_scal(val)
    call x%set_host()
  end subroutine i_gpu_set_scal

  subroutine i_gpu_set_vect(x,val)
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)           :: val(:)
    integer(psb_ipk_) :: nr
    integer(psb_ipk_) :: info

    call x%psb_i_base_vect_type%set_vect(val)
    call x%set_host()

  end subroutine i_gpu_set_vect



  subroutine i_gpu_scal(alpha, x)
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent (in)          :: alpha
    
    call x%sync()
    call x%psb_i_base_vect_type%scal(alpha)
    call x%set_host()
  end subroutine i_gpu_scal


  function i_gpu_nrm2(n,x) result(res)
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n
    integer(psb_ipk_)                    :: res
    integer(psb_ipk_) :: info
    ! WARNING: this should be changed. 
    call x%sync()
    res = x%psb_i_base_vect_type%nrm2(n)
    
  end function i_gpu_nrm2
  
  function i_gpu_amax(n,x) result(res)
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n
    integer(psb_ipk_)                :: res

    ! WARNING: this should be changed. 
    call x%sync()
    res = x%psb_i_base_vect_type%amax(n)

  end function i_gpu_amax

  function i_gpu_asum(n,x) result(res)
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n
    integer(psb_ipk_)                :: res

    ! WARNING: this should be changed. 
    call x%sync()
    res = x%psb_i_base_vect_type%asum(n)

  end function i_gpu_asum
  
  subroutine i_gpu_all(n, x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    integer(psb_ipk_), intent(in)      :: n
    class(psb_i_vect_gpu), intent(out) :: x
    integer(psb_ipk_), intent(out)     :: info
    
    call psb_realloc(n,x%v,info)
    if (info == 0) call x%set_host()
    if (info == 0) call x%sync_space(info)
    if (info /= 0) then 
      info=psb_err_alloc_request_
      call psb_errpush(info,'i_gpu_all',&
           & i_err=(/n,n,n,n,n/))
    end if
  end subroutine i_gpu_all

  subroutine i_gpu_zero(x)
    use psi_serial_mod
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    
    if (allocated(x%v)) x%v=izero
    call x%set_host()
  end subroutine i_gpu_zero

  subroutine i_gpu_asb(n, x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    integer(psb_ipk_), intent(in)        :: n
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(out)       :: info
    
    call x%psb_i_base_vect_type%asb(n,info)
        
    if (info == 0) then 
      call x%set_host()
      if (c_associated(x%deviceVect)) then 
        call freeMultiVecDevice(x%deviceVect)
        x%deviceVect=c_null_ptr
      end if
      call x%sync()
    else
      info=psb_err_internal_error_
      call psb_errpush(info,'i_gpu_asb')
    end if
  end subroutine i_gpu_asb

  subroutine i_gpu_sync_space(x,info)
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(out)       :: info 
    integer(psb_ipk_) :: n
    
    info = 0
    if (x%is_host()) then 
      if (.not.c_associated(x%deviceVect)) then 
        n    = size(x%v)
        info = FallocMultiVecDevice(x%deviceVect,1,n,spgpu_type_int)
        if  (info /= 0) then 
!!$          write(0,*) 'Error from FallocMultiVecDevice',info,n
          if (info == spgpu_outofmem) then 
            info = psb_err_alloc_request_
          end if
        end if
      end if
    else if (x%is_dev()) then 
      ! 
      write(0,*) 'What is going on??? ' 
    end if
    
  end subroutine i_gpu_sync_space

  subroutine i_gpu_sync(x)
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: n,info
    
!!$    write(0,*) 'Sync in i_gpu_sync'
    info = 0
    if (x%is_host()) then 
      if (.not.c_associated(x%deviceVect)) then 
        n    = size(x%v)
        info = FallocMultiVecDevice(x%deviceVect,1,n,spgpu_type_int)
      end if
      if (info == 0) &
           & info = writeMultiVecDevice(x%deviceVect,x%v)
    else if (x%is_dev()) then 
      info = readMultiVecDevice(x%deviceVect,x%v)
    end if
    if (info == 0)  call x%set_sync()
    if (info /= 0) then
      info=psb_err_internal_error_
      call psb_errpush(info,'i_gpu_sync')
    end if
    
  end subroutine i_gpu_sync

  subroutine i_gpu_free(x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    class(psb_i_vect_gpu), intent(inout)  :: x
    integer(psb_ipk_), intent(out)        :: info
    
    info = 0
    if (c_associated(x%deviceVect)) then 
      call freeMultiVecDevice(x%deviceVect)
      x%deviceVect=c_null_ptr
    end if
    if (allocated(x%v)) deallocate(x%v, stat=info)
    call x%set_host()
  end subroutine i_gpu_free

#ifdef HAVE_FINAL
  subroutine i_gpu_vect_finalize(x)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    type(psb_i_vect_gpu), intent(inout)  :: x
    integer(psb_ipk_)        :: info
    
    info = 0
    if (c_associated(x%deviceVect)) then 
      call freeMultiVecDevice(x%deviceVect)
      x%deviceVect=c_null_ptr
    end if
    if (allocated(x%v)) deallocate(x%v, stat=info)
    call x%set_host()
  end subroutine i_gpu_vect_finalize
#endif

  subroutine i_gpu_ins_a(n,irl,val,dupl,x,info)
    use psi_serial_mod
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n, dupl
    integer(psb_ipk_), intent(in)        :: irl(:)
    integer(psb_ipk_), intent(in)           :: val(:)
    integer(psb_ipk_), intent(out)       :: info

    integer(psb_ipk_) :: i

    info = 0
    call x%psb_i_base_vect_type%ins(n,irl,val,dupl,info)
    call x%set_host()

  end subroutine i_gpu_ins_a

#endif

end module psb_i_gpu_vect_mod



module psb_i_gpu_multivect_mod
  use iso_c_binding
  use psb_const_mod
  use psb_i_base_multivect_mod

#ifdef HAVE_SPGPU
  use psb_vectordev_mod
#endif

  integer(psb_ipk_), parameter, private :: is_host = -1
  integer(psb_ipk_), parameter, private :: is_sync = 0 
  integer(psb_ipk_), parameter, private :: is_dev  = 1 
  
  type, extends(psb_i_base_multivect_type) ::  psb_i_multivect_gpu
#ifdef HAVE_SPGPU
    integer(psb_ipk_)  :: state      = is_host, m_nrows=0, m_ncols=0
    type(c_ptr) :: deviceVect = c_null_ptr
    integer(c_int), allocatable :: buffer(:,:)
    type(c_ptr) :: d_val = c_null_ptr
  contains
    procedure, pass(x) :: get_nrows => i_gpu_multi_get_nrows
    procedure, pass(x) :: get_ncols => i_gpu_multi_get_ncols
    procedure, nopass  :: get_fmt   => i_gpu_multi_get_fmt
!!$    procedure, pass(y) :: axpby_v  => i_gpu_multi_axpby_v
!!$    procedure, pass(y) :: axpby_a  => i_gpu_multi_axpby_a
!!$    procedure, pass(y) :: mlt_v    => i_gpu_multi_mlt_v
!!$    procedure, pass(y) :: mlt_a    => i_gpu_multi_mlt_a
!!$    procedure, pass(z) :: mlt_a_2  => i_gpu_multi_mlt_a_2
!!$    procedure, pass(z) :: mlt_v_2  => i_gpu_multi_mlt_v_2
!!$    procedure, pass(x) :: scal     => i_gpu_multi_scal
    procedure, pass(x) :: all      => i_gpu_multi_all
    procedure, pass(x) :: zero     => i_gpu_multi_zero
    procedure, pass(x) :: asb      => i_gpu_multi_asb
    procedure, pass(x) :: sync     => i_gpu_multi_sync
    procedure, pass(x) :: sync_space => i_gpu_multi_sync_space
    procedure, pass(x) :: bld_x    => i_gpu_multi_bld_x
    procedure, pass(x) :: bld_n    => i_gpu_multi_bld_n
    procedure, pass(x) :: free     => i_gpu_multi_free
    procedure, pass(x) :: ins      => i_gpu_multi_ins
    procedure, pass(x) :: is_host  => i_gpu_multi_is_host
    procedure, pass(x) :: is_dev   => i_gpu_multi_is_dev
    procedure, pass(x) :: is_sync  => i_gpu_multi_is_sync
    procedure, pass(x) :: set_host => i_gpu_multi_set_host
    procedure, pass(x) :: set_dev  => i_gpu_multi_set_dev
    procedure, pass(x) :: set_sync => i_gpu_multi_set_sync
    procedure, pass(x) :: set_scal => i_gpu_multi_set_scal
    procedure, pass(x) :: set_vect => i_gpu_multi_set_vect
!!$    procedure, pass(x) :: gthzv    => i_gpu_multi_gthzv
!!$    procedure, pass(y) :: sctb     => i_gpu_multi_sctb
#ifdef HAVE_FINAL
    final              :: i_gpu_multi_vect_finalize
#endif
#endif
  end type psb_i_multivect_gpu

  public  :: psb_i_multivect_gpu_
  private :: constructor
  interface psb_i_multivect_gpu_
    module procedure constructor
  end interface

contains
  
  function constructor(x) result(this)
    real(psb_dpk_)       :: x(:,:)
    type(psb_i_multivect_gpu) :: this
    integer(psb_ipk_) :: info

    this%v = x
    call this%asb(size(x,1),size(x,2),info)

  end function constructor
    
#ifdef HAVE_SPGPU

  
  subroutine i_gpu_multi_bld_x(x,this)
    use psb_base_mod
    integer(psb_ipk_), intent(in)           :: this(:,:)
    class(psb_i_multivect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: info, m, n
    
    m=size(this,1)
    n=size(this,2)
    x%m_nrows = m
    x%m_ncols = n
    call psb_realloc(m,n,x%v,info)
    if (info /= 0) then 
      info=psb_err_alloc_request_
      call psb_errpush(info,'i_gpu_multi_bld_x',&
           & i_err=(/size(this,1),size(this,2),izero,izero,izero,izero/))
    end if
    x%v(1:m,1:n)  = this(1:m,1:n) 
    call x%set_host()
    call x%sync()

  end subroutine i_gpu_multi_bld_x

  subroutine i_gpu_multi_bld_n(x,m,n)
    use psb_error_mod
    integer(psb_ipk_), intent(in) :: m,n
    class(psb_i_multivect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: info

    call x%all(m,n,info)
    if (info /= 0) then 
      call psb_errpush(info,'i_gpu_multi_bld_n',i_err=(/m,n,n,n,n/))
    end if

  end subroutine i_gpu_multi_bld_n


  subroutine i_gpu_multi_set_host(x)
    implicit none 
    class(psb_i_multivect_gpu), intent(inout) :: x
    
    x%state = is_host
  end subroutine i_gpu_multi_set_host

  subroutine i_gpu_multi_set_dev(x)
    implicit none 
    class(psb_i_multivect_gpu), intent(inout) :: x
    
    x%state = is_dev
  end subroutine i_gpu_multi_set_dev

  subroutine i_gpu_multi_set_sync(x)
    implicit none 
    class(psb_i_multivect_gpu), intent(inout) :: x
    
    x%state = is_sync
  end subroutine i_gpu_multi_set_sync

  function i_gpu_multi_is_dev(x) result(res)
    implicit none 
    class(psb_i_multivect_gpu), intent(in) :: x
    logical  :: res
  
    res = (x%state == is_dev)
  end function i_gpu_multi_is_dev
  
  function i_gpu_multi_is_host(x) result(res)
    implicit none 
    class(psb_i_multivect_gpu), intent(in) :: x
    logical  :: res

    res = (x%state == is_host)
  end function i_gpu_multi_is_host

  function i_gpu_multi_is_sync(x) result(res)
    implicit none 
    class(psb_i_multivect_gpu), intent(in) :: x
    logical  :: res

    res = (x%state == is_sync)
  end function i_gpu_multi_is_sync

  
  function i_gpu_multi_get_nrows(x) result(res)
    implicit none 
    class(psb_i_multivect_gpu), intent(in) :: x
    integer(psb_ipk_) :: res

    res = x%m_nrows
!!$    if (x%is_dev()) then 
!!$      res  = getMultiVecDevicePitch(x%deviceVect)
!!$    else if (allocated(x%v)) then 
!!$      res = size(x%v,1)
!!$    end if

  end function i_gpu_multi_get_nrows
  
  function i_gpu_multi_get_ncols(x) result(res)
    implicit none 
    class(psb_i_multivect_gpu), intent(in) :: x
    integer(psb_ipk_) :: res


    res = x%m_ncols
!!$
!!$    if (x%is_dev()) then 
!!$      res  = getMultiVecDeviceCount(x%deviceVect)
!!$    else if (allocated(x%v)) then 
!!$      res = size(x%v,2)
!!$    end if


  end function i_gpu_multi_get_ncols

  function i_gpu_multi_get_fmt() result(res)
    implicit none 
    character(len=5) :: res
    res = 'iGPU'
  end function i_gpu_multi_get_fmt

!!$  subroutine i_gpu_multi_axpby_v(m,alpha, x, beta, y, info)
!!$    use psi_serial_mod
!!$    implicit none 
!!$    integer(psb_ipk_), intent(in)              :: m
!!$    class(psb_i_base_multivect_type), intent(inout) :: x
!!$    class(psb_i_multivect_gpu), intent(inout)       :: y
!!$    integer(psb_ipk_), intent (in)                :: alpha, beta
!!$    integer(psb_ipk_), intent(out)             :: info
!!$    integer(psb_ipk_) :: nx, ny
!!$
!!$    call x%sync()
!!$    call y%axpby(m,alpha,x%v,beta,info)
!!$    call y%set_host()
!!$
!!$  end subroutine i_gpu_multi_axpby_v
!!$
!!$  subroutine i_gpu_multi_axpby_a(m,alpha, x, beta, y, info)
!!$    use psi_serial_mod
!!$    implicit none 
!!$    integer(psb_ipk_), intent(in)        :: m
!!$    integer(psb_ipk_), intent(in)           :: x(:)
!!$    class(psb_i_multivect_gpu), intent(inout) :: y
!!$    integer(psb_ipk_), intent (in)          :: alpha, beta
!!$    integer(psb_ipk_), intent(out)       :: info
!!$
!!$    if (y%is_dev()) call y%sync()
!!$    call psb_geaxpby(m,alpha,x,beta,y%v,info)
!!$    call y%set_host()
!!$  end subroutine i_gpu_multi_axpby_a
!!$
!!$  subroutine i_gpu_multi_mlt_v(x, y, info)
!!$    use psi_serial_mod
!!$    implicit none 
!!$    class(psb_i_base_multivect_type), intent(inout) :: x
!!$    class(psb_i_multivect_gpu), intent(inout)       :: y
!!$    integer(psb_ipk_), intent(out)             :: info
!!$
!!$    integer(psb_ipk_) :: i, n
!!$    
!!$    info = 0    
!!$    call x%sync()
!!$    call y%mlt(x%v,info)
!!$    call y%set_host()
!!$
!!$  end subroutine i_gpu_multi_mlt_v
!!$
!!$  subroutine i_gpu_multi_mlt_a(x, y, info)
!!$    use psi_serial_mod
!!$    implicit none 
!!$    integer(psb_ipk_), intent(in)           :: x(:)
!!$    class(psb_i_multivect_gpu), intent(inout) :: y
!!$    integer(psb_ipk_), intent(out)       :: info
!!$    integer(psb_ipk_) :: i, n
!!$    
!!$    info = 0    
!!$    call y%sync()
!!$    call y%psb_i_base_multivect_type%mlt(x,info)
!!$    call y%set_host()
!!$  end subroutine i_gpu_multi_mlt_a
!!$
!!$  subroutine i_gpu_multi_mlt_a_2(alpha,x,y,beta,z,info)
!!$    use psi_serial_mod
!!$    implicit none 
!!$    integer(psb_ipk_), intent(in)           :: alpha,beta
!!$    integer(psb_ipk_), intent(in)           :: x(:)
!!$    integer(psb_ipk_), intent(in)           :: y(:)
!!$    class(psb_i_multivect_gpu), intent(inout) :: z
!!$    integer(psb_ipk_), intent(out)       :: info
!!$    integer(psb_ipk_) :: i, n
!!$
!!$    info = 0    
!!$    call z%sync()
!!$    call z%psb_i_base_multivect_type%mlt(alpha,x,y,beta,info)
!!$    call z%set_host()
!!$  end subroutine i_gpu_multi_mlt_a_2
!!$
!!$  subroutine i_gpu_multi_mlt_v_2(alpha,x,y, beta,z,info,conjgx,conjgy)
!!$    use psi_serial_mod
!!$    use psb_string_mod
!!$    implicit none 
!!$    integer(psb_ipk_), intent(in)                 :: alpha,beta
!!$    class(psb_i_base_multivect_type), intent(inout) :: x
!!$    class(psb_i_base_multivect_type), intent(inout) :: y
!!$    class(psb_i_multivect_gpu), intent(inout)       :: z
!!$    integer(psb_ipk_), intent(out)             :: info
!!$    character(len=1), intent(in), optional     :: conjgx, conjgy
!!$    integer(psb_ipk_) :: i, n
!!$    logical :: conjgx_, conjgy_
!!$    call x%sync()
!!$    call y%sync()
!!$    call z%psb_i_base_multivect_type%mlt(alpha,x,y,beta,info)
!!$    call z%set_host()
!!$
!!$  end subroutine i_gpu_multi_mlt_v_2
!!$

  subroutine i_gpu_multi_set_scal(x,val)
    class(psb_i_multivect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)           :: val
        
    integer(psb_ipk_) :: info
    if (x%is_dev()) call x%sync()
    call x%psb_i_base_multivect_type%set_scal(val)
    call x%set_host()
  end subroutine i_gpu_multi_set_scal

  subroutine i_gpu_multi_set_vect(x,val)
    class(psb_i_multivect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)           :: val(:,:)
    integer(psb_ipk_) :: nr
    integer(psb_ipk_) :: info

    if (x%is_dev()) call x%sync()
    call x%psb_i_base_multivect_type%set_vect(val)
    call x%set_host()

  end subroutine i_gpu_multi_set_vect


!!$
!!$  subroutine i_gpu_multi_scal(alpha, x)
!!$    implicit none 
!!$    class(psb_i_multivect_gpu), intent(inout) :: x
!!$    integer(psb_ipk_), intent (in)          :: alpha
!!$    
!!$    call x%sync()
!!$    call x%psb_i_base_multivect_type%scal(alpha)
!!$    call x%set_host()
!!$  end subroutine i_gpu_multi_scal
!!$
!!$
!!$  function i_gpu_multi_nrm2(n,x) result(res)
!!$    implicit none 
!!$    class(psb_i_multivect_gpu), intent(inout) :: x
!!$    integer(psb_ipk_), intent(in)        :: n
!!$    integer(psb_ipk_)                    :: res
!!$    integer(psb_ipk_) :: info
!!$    ! WARNING: this should be changed. 
!!$    call x%sync()
!!$    res = x%psb_i_base_multivect_type%nrm2(n)
!!$    
!!$  end function i_gpu_multi_nrm2
!!$  
!!$  function i_gpu_multi_amax(n,x) result(res)
!!$    implicit none 
!!$    class(psb_i_multivect_gpu), intent(inout) :: x
!!$    integer(psb_ipk_), intent(in)        :: n
!!$    integer(psb_ipk_)                :: res
!!$
!!$    ! WARNING: this should be changed. 
!!$    call x%sync()
!!$    res = x%psb_i_base_multivect_type%amax(n)
!!$
!!$  end function i_gpu_multi_amax
!!$
!!$  function i_gpu_multi_asum(n,x) result(res)
!!$    implicit none 
!!$    class(psb_i_multivect_gpu), intent(inout) :: x
!!$    integer(psb_ipk_), intent(in)        :: n
!!$    integer(psb_ipk_)                :: res
!!$
!!$    ! WARNING: this should be changed. 
!!$    call x%sync()
!!$    res = x%psb_i_base_multivect_type%asum(n)
!!$
!!$  end function i_gpu_multi_asum
  
  subroutine i_gpu_multi_all(m,n, x, info)
    use psi_serial_mod
    use psb_realloc_mod
    use psb_error_mod
    implicit none 
    integer(psb_ipk_), intent(in)      :: m,n
    class(psb_i_multivect_gpu), intent(out) :: x
    integer(psb_ipk_), intent(out)     :: info
    
    call psb_realloc(m,n,x%v,info)
    x%m_nrows = m
    x%m_ncols = n
    if (info == 0) call x%set_host()
    if (info == 0) call x%sync_space(info)
    if (info /= 0) then 
      info=psb_err_alloc_request_
      call psb_errpush(info,'i_gpu_multi_all',&
           & i_err=(/m,n,n,n,n/))
    end if
  end subroutine i_gpu_multi_all

  subroutine i_gpu_multi_zero(x)
    use psi_serial_mod
    implicit none 
    class(psb_i_multivect_gpu), intent(inout) :: x
    
    if (allocated(x%v)) x%v=izero
    call x%set_host()
  end subroutine i_gpu_multi_zero

  subroutine i_gpu_multi_asb(m,n, x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    integer(psb_ipk_), intent(in)        :: m,n
    class(psb_i_multivect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(out)       :: info
    integer(psb_ipk_) :: nd, nc


    x%m_nrows = m
    x%m_ncols = n
    if (x%is_host()) then 
      call x%psb_i_base_multivect_type%asb(m,n,info)
      if (info == psb_success_) call x%sync_space(info)
    else if (x%is_dev()) then 
      nd  = getMultiVecDevicePitch(x%deviceVect)
      nc  = getMultiVecDeviceCount(x%deviceVect)
      if ((nd < m).or.(nc<n)) then 
        call x%sync()
        call x%psb_i_base_multivect_type%asb(m,n,info)      
        if (info == psb_success_) call x%sync_space(info)
        call x%set_host()
      end if
    end if
  end subroutine i_gpu_multi_asb

  subroutine i_gpu_multi_sync_space(x,info)
    use psb_realloc_mod
    implicit none 
    class(psb_i_multivect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(out)       :: info 
    integer(psb_ipk_) :: mh,nh,md,nd
    
    info = 0
    if (x%is_host()) then 
      if (allocated(x%v)) then 
        mh = size(x%v,1)
        nh = size(x%v,2)
      else
        mh=0
        nh=0
      end if
      if (c_associated(x%deviceVect)) then 
        md  = getMultiVecDevicePitch(x%deviceVect)
        nd  = getMultiVecDeviceCount(x%deviceVect)
        if ((md < mh).or.(nd<nh)) then 
          call freeMultiVecDevice(x%deviceVect)
          x%deviceVect=c_null_ptr
        end if
      end if

      if (.not.c_associated(x%deviceVect)) then 
        info = FallocMultiVecDevice(x%deviceVect,nh,mh,spgpu_type_double)
        if (info == 0) &
             & call psb_realloc(getMultiVecDevicePitch(x%deviceVect),&
             & getMultiVecDeviceCount(x%deviceVect),x%v,info)
        if  (info /= 0) then 
!!$          write(0,*) 'Error from FallocMultiVecDevice',info,n
          if (info == spgpu_outofmem) then 
            info = psb_err_alloc_request_
          end if
        end if
        
      end if
    else if (x%is_dev()) then 
      ! 
      if (allocated(x%v)) then 
        mh = size(x%v,1)
        nh = size(x%v,2)
      else
        mh=0
        nh=0
      end if
      md  = getMultiVecDevicePitch(x%deviceVect)
      nd  = getMultiVecDeviceCount(x%deviceVect)
      if ((mh /= md).or.(nh /= nd)) then 
        call psb_realloc(getMultiVecDevicePitch(x%deviceVect),&
             & getMultiVecDeviceCount(x%deviceVect),x%v,info)
      end if
      
    end if
    
  end subroutine i_gpu_multi_sync_space

  subroutine i_gpu_multi_sync(x)
    use psb_error_mod
    implicit none 
    class(psb_i_multivect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: n,info
    
    info = 0
    if (x%is_host()) then 
      if (.not.c_associated(x%deviceVect)) then 
        call x%sync_space(info)
      end if
      if (info == 0) &
           & info = writeMultiVecDevice(x%deviceVect,x%v,size(x%v,1))
    else if (x%is_dev()) then 
      info = readMultiVecDevice(x%deviceVect,x%v,size(x%v,1))
    end if
    if (info == 0)  call x%set_sync()
    if (info /= 0) then
      info=psb_err_internal_error_
      call psb_errpush(info,'i_gpu_multi_sync')
    end if
    
  end subroutine i_gpu_multi_sync

  subroutine i_gpu_multi_free(x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    class(psb_i_multivect_gpu), intent(inout)  :: x
    integer(psb_ipk_), intent(out)        :: info
    
    info = 0
    if (c_associated(x%deviceVect)) then 
      call freeMultiVecDevice(x%deviceVect)
      x%deviceVect=c_null_ptr
    end if
    if (allocated(x%buffer)) then 
!!$      call inner_unregister(x%buffer)
      deallocate(x%buffer, stat=info)
    end if

    if (allocated(x%v)) deallocate(x%v, stat=info)
    call x%set_sync()
  end subroutine i_gpu_multi_free


#ifdef HAVE_FINAL
  subroutine i_gpu_multi_vect_finalize(x)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    type(psb_i_multivect_gpu), intent(inout)  :: x
    integer(psb_ipk_)        :: info
    
    
    info = 0
    if (c_associated(x%deviceVect)) then 
      call freeMultiVecDevice(x%deviceVect)
      x%deviceVect=c_null_ptr
    end if
    if (allocated(x%buffer)) then 
!!$      call inner_unregister(x%buffer)
      deallocate(x%buffer, stat=info)
    end if

    if (allocated(x%v)) deallocate(x%v, stat=info)
    call x%set_sync()
  end subroutine i_gpu_multi_vect_finalize
#endif

  subroutine i_gpu_multi_ins(n,irl,val,dupl,x,info)
    use psi_serial_mod
    implicit none 
    class(psb_i_multivect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n, dupl
    integer(psb_ipk_), intent(in)        :: irl(:)
    integer(psb_ipk_), intent(in)           :: val(:,:)
    integer(psb_ipk_), intent(out)       :: info

    integer(psb_ipk_) :: i

    info = 0
    if (x%is_dev()) call x%sync()
    call x%psb_i_base_multivect_type%ins(n,irl,val,dupl,info)
    call x%set_host()

  end subroutine i_gpu_multi_ins

#endif

end module psb_i_gpu_multivect_mod



