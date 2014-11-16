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
  

module psb_d_gpu_vect_mod
  use iso_c_binding
  use psb_const_mod
  use psb_error_mod
  use psb_d_vect_mod
  use psb_i_vect_mod
#ifdef HAVE_SPGPU
  use psb_i_gpu_vect_mod
  use psb_d_vectordev_mod
#endif

  integer(psb_ipk_), parameter, private :: is_host = -1
  integer(psb_ipk_), parameter, private :: is_sync = 0 
  integer(psb_ipk_), parameter, private :: is_dev  = 1 
  
  type, extends(psb_d_base_vect_type) ::  psb_d_vect_gpu
#ifdef HAVE_SPGPU
    integer     :: state      = is_host
    type(c_ptr) :: deviceVect = c_null_ptr
    real(c_double), allocatable :: buffer(:)
    type(c_ptr) :: d_val = c_null_ptr
  contains
    procedure, pass(x) :: get_nrows => d_gpu_get_nrows
    procedure, nopass  :: get_fmt   => d_gpu_get_fmt
    procedure, pass(x) :: dot_v    => d_gpu_dot_v
    procedure, pass(x) :: dot_a    => d_gpu_dot_a
    procedure, pass(y) :: axpby_v  => d_gpu_axpby_v
    procedure, pass(y) :: axpby_a  => d_gpu_axpby_a
    procedure, pass(y) :: mlt_v    => d_gpu_mlt_v
    procedure, pass(y) :: mlt_a    => d_gpu_mlt_a
    procedure, pass(z) :: mlt_a_2  => d_gpu_mlt_a_2
    procedure, pass(z) :: mlt_v_2  => d_gpu_mlt_v_2
    procedure, pass(x) :: scal     => d_gpu_scal
    procedure, pass(x) :: nrm2     => d_gpu_nrm2
    procedure, pass(x) :: amax     => d_gpu_amax
    procedure, pass(x) :: asum     => d_gpu_asum
    procedure, pass(x) :: all      => d_gpu_all
    procedure, pass(x) :: zero     => d_gpu_zero
    procedure, pass(x) :: asb      => d_gpu_asb
    procedure, pass(x) :: sync     => d_gpu_sync
    procedure, pass(x) :: sync_space => d_gpu_sync_space
    procedure, pass(x) :: bld_x    => d_gpu_bld_x
    procedure, pass(x) :: bld_n    => d_gpu_bld_n
    procedure, pass(x) :: free     => d_gpu_free
    procedure, pass(x) :: ins_a    => d_gpu_ins_a
    procedure, pass(x) :: is_host  => d_gpu_is_host
    procedure, pass(x) :: is_dev   => d_gpu_is_dev
    procedure, pass(x) :: is_sync  => d_gpu_is_sync
    procedure, pass(x) :: set_host => d_gpu_set_host
    procedure, pass(x) :: set_dev  => d_gpu_set_dev
    procedure, pass(x) :: set_sync => d_gpu_set_sync
    procedure, pass(x) :: set_scal => d_gpu_set_scal
    procedure, pass(x) :: set_vect => d_gpu_set_vect
    procedure, pass(x) :: gthzv_x  => d_gpu_gthzv_x
    procedure, pass(y) :: sctb     => d_gpu_sctb
    procedure, pass(y) :: sctb_x   => d_gpu_sctb_x
#ifdef HAVE_FINAL
    final              :: d_gpu_vect_finalize
#endif
#endif
  end type psb_d_vect_gpu

  public  :: psb_d_vect_gpu_
  private :: constructor
  interface psb_d_vect_gpu_
    module procedure constructor
  end interface psb_d_vect_gpu_

contains
  
  function constructor(x) result(this)
    real(psb_dpk_)       :: x(:)
    type(psb_d_vect_gpu) :: this
    integer(psb_ipk_) :: info

    this%v = x
    call this%asb(size(x),info)

  end function constructor
    
#ifdef HAVE_SPGPU

  subroutine d_gpu_gthzv_x(i,n,idx,x,y)
    use psi_serial_mod
    integer(psb_ipk_) :: i,n
    class(psb_i_base_vect_type) :: idx
    real(psb_dpk_) ::  y(:)
    class(psb_d_vect_gpu) :: x

    select type(ii=> idx) 
    class is (psb_i_vect_gpu) 
      if (ii%is_host()) call ii%sync()
      if (x%is_host())  call x%sync()

      if (allocated(x%buffer)) then 
        if (size(x%buffer) < n) then 
          call inner_unregister(x%buffer)
          deallocate(x%buffer, stat=info)
        end if
      end if
      
      if (.not.allocated(x%buffer)) then
        allocate(x%buffer(n),stat=info)
        if (info == 0) info = inner_register(x%buffer,x%d_val)        
      endif
      info = igathMultiVecDeviceDouble(x%deviceVect,&
           & 0, i, n, ii%deviceVect, x%d_val, 1)
      call psb_cudaSync()
      y(1:n) = x%buffer(1:n)
      
    class default
      call x%gth(n,ii%v(i:),y)
    end select


  end subroutine d_gpu_gthzv_x



  subroutine d_gpu_sctb(n,idx,x,beta,y)
    implicit none
    !use psb_const_mod
    integer(psb_ipk_)     :: n, idx(:)
    real(psb_dpk_)        :: beta, x(:)
    class(psb_d_vect_gpu) :: y
    integer(psb_ipk_)     :: info

    if (n == 0) return
    
    if (y%is_dev())  call y%sync()
          
    call y%psb_d_base_vect_type%sctb(n,idx,x,beta)
    call y%set_host()

  end subroutine d_gpu_sctb

  subroutine d_gpu_sctb_x(i,n,idx,x,beta,y)
    use psi_serial_mod
    integer(psb_ipk_) :: i, n
    class(psb_i_base_vect_type) :: idx
    real(psb_dpk_) :: beta, x(:)
    class(psb_d_vect_gpu) :: y

    select type(ii=> idx) 
    class is (psb_i_vect_gpu) 
      if (ii%is_host()) call ii%sync()
      if (y%is_host())  call y%sync()

      if (allocated(y%buffer)) then 
        if (size(y%buffer) < n) then 
          call inner_unregister(y%buffer)
          deallocate(y%buffer, stat=info)
        end if
      end if
      
      if (.not.allocated(y%buffer)) then
        allocate(y%buffer(n),stat=info)
        if (info == 0) info = inner_register(y%buffer,y%d_val)        
      endif
      y%buffer(1:n) = x(1:n) 
      info = iscatMultiVecDeviceDouble(y%deviceVect,&
           & 0, i, n, ii%deviceVect, y%d_val, 1,beta)

      call y%set_dev()
      call psb_cudaSync()   
      
    class default
      call y%sct(n,ii%v(i:),x,beta)
    end select

  end subroutine d_gpu_sctb_x


  subroutine d_gpu_bld_x(x,this)
    use psb_base_mod
    real(psb_dpk_), intent(in)           :: this(:)
    class(psb_d_vect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: info

    call psb_realloc(size(this),x%v,info)
    if (info /= 0) then 
      info=psb_err_alloc_request_
      call psb_errpush(info,'d_gpu_bld_x',&
           & i_err=(/size(this),izero,izero,izero,izero/))
    end if
    x%v(:)  = this(:) 
    call x%set_host()
    call x%sync()

  end subroutine d_gpu_bld_x

  subroutine d_gpu_bld_n(x,n)
    integer(psb_ipk_), intent(in) :: n
    class(psb_d_vect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: info

    call x%all(n,info)
    if (info /= 0) then 
      call psb_errpush(info,'d_gpu_bld_n',i_err=(/n,n,n,n,n/))
    end if
    
  end subroutine d_gpu_bld_n


  subroutine d_gpu_set_host(x)
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    
    x%state = is_host
  end subroutine d_gpu_set_host

  subroutine d_gpu_set_dev(x)
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    
    x%state = is_dev
  end subroutine d_gpu_set_dev

  subroutine d_gpu_set_sync(x)
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    
    x%state = is_sync
  end subroutine d_gpu_set_sync

  function d_gpu_is_dev(x) result(res)
    implicit none 
    class(psb_d_vect_gpu), intent(in) :: x
    logical  :: res
  
    res = (x%state == is_dev)
  end function d_gpu_is_dev
  
  function d_gpu_is_host(x) result(res)
    implicit none 
    class(psb_d_vect_gpu), intent(in) :: x
    logical  :: res

    res = (x%state == is_host)
  end function d_gpu_is_host

  function d_gpu_is_sync(x) result(res)
    implicit none 
    class(psb_d_vect_gpu), intent(in) :: x
    logical  :: res

    res = (x%state == is_sync)
  end function d_gpu_is_sync

  
  function d_gpu_get_nrows(x) result(res)
    implicit none 
    class(psb_d_vect_gpu), intent(in) :: x
    integer(psb_ipk_) :: res

    res = 0
    if (allocated(x%v)) res = size(x%v)
  end function d_gpu_get_nrows

  function d_gpu_get_fmt() result(res)
    implicit none 
    character(len=5) :: res
    res = 'dGPU'
  end function d_gpu_get_fmt

  function d_gpu_dot_v(n,x,y) result(res)
    implicit none 
    class(psb_d_vect_gpu), intent(inout)       :: x
    class(psb_d_base_vect_type), intent(inout) :: y
    integer(psb_ipk_), intent(in)              :: n
    real(psb_dpk_)                :: res
    real(psb_dpk_), external      :: ddot
    integer(psb_ipk_) :: info
    
    res = dzero
    !
    ! Note: this is the gpu implementation.
    !  When we get here, we are sure that X is of
    !  TYPE psb_d_vect
    !
    select type(yy => y)
    type is (psb_d_base_vect_type)
      if (x%is_dev()) call x%sync()
      res = ddot(n,x%v,1,yy%v,1)
    type is (psb_d_vect_gpu)
      if (x%is_host()) call x%sync()
      if (yy%is_host()) call yy%sync()
      info = dotMultiVecDevice(res,n,x%deviceVect,yy%deviceVect)
      if (info /= 0) then 
        info = psb_err_internal_error_
        call psb_errpush(info,'d_gpu_dot_v')
      end if

    class default
      ! y%sync is done in dot_a
      call x%sync()      
      res = y%dot(n,x%v)
    end select

  end function d_gpu_dot_v

  function d_gpu_dot_a(n,x,y) result(res)
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    real(psb_dpk_), intent(in)           :: y(:)
    integer(psb_ipk_), intent(in)        :: n
    real(psb_dpk_)                :: res
    real(psb_dpk_), external      :: ddot
    
    if (x%is_dev()) call x%sync()
    res = ddot(n,y,1,x%v,1)

  end function d_gpu_dot_a
    
  subroutine d_gpu_axpby_v(m,alpha, x, beta, y, info)
    use psi_serial_mod
    implicit none 
    integer(psb_ipk_), intent(in)              :: m
    class(psb_d_base_vect_type), intent(inout) :: x
    class(psb_d_vect_gpu), intent(inout)       :: y
    real(psb_dpk_), intent (in)                :: alpha, beta
    integer(psb_ipk_), intent(out)             :: info
    integer(psb_ipk_) :: nx, ny

    info = psb_success_

    select type(xx => x)
    type is (psb_d_base_vect_type)
      if ((beta /= dzero).and.(y%is_dev()))&
           & call y%sync()
      call psb_geaxpby(m,alpha,xx%v,beta,y%v,info)
      call y%set_host()
    type is (psb_d_vect_gpu)
      ! Do something different here 
      if ((beta /= dzero).and.y%is_host())&
           &  call y%sync()
      if (xx%is_host()) call xx%sync()
      nx = getMultiVecDeviceSize(xx%deviceVect)
      ny = getMultiVecDeviceSize(y%deviceVect)
      if ((nx<m).or.(ny<m)) then
        info = psb_err_internal_error_
      else
        info = axpbyMultiVecDevice(m,alpha,xx%deviceVect,beta,y%deviceVect)
      end if
      call y%set_dev()
    class default
      call x%sync()
      call y%axpby(m,alpha,x%v,beta,info)
    end select

  end subroutine d_gpu_axpby_v

  subroutine d_gpu_axpby_a(m,alpha, x, beta, y, info)
    use psi_serial_mod
    implicit none 
    integer(psb_ipk_), intent(in)        :: m
    real(psb_dpk_), intent(in)           :: x(:)
    class(psb_d_vect_gpu), intent(inout) :: y
    real(psb_dpk_), intent (in)          :: alpha, beta
    integer(psb_ipk_), intent(out)       :: info

    if (y%is_dev()) call y%sync()
    call psb_geaxpby(m,alpha,x,beta,y%v,info)
    call y%set_host()
  end subroutine d_gpu_axpby_a

  subroutine d_gpu_mlt_v(x, y, info)
    use psi_serial_mod
    implicit none 
    class(psb_d_base_vect_type), intent(inout) :: x
    class(psb_d_vect_gpu), intent(inout)       :: y
    integer(psb_ipk_), intent(out)             :: info

    integer(psb_ipk_) :: i, n
    
    info = 0    
    n = min(x%get_nrows(),y%get_nrows())
    select type(xx => x)
    type is (psb_d_base_vect_type)
      if (y%is_dev()) call y%sync()
      do i=1, n
        y%v(i) = y%v(i) * xx%v(i)
      end do
      call y%set_host()
    type is (psb_d_vect_gpu)
      ! Do something different here 
      if (y%is_host())  call y%sync()
      if (xx%is_host()) call xx%sync()
      info = axyMultiVecDevice(n,done,xx%deviceVect,y%deviceVect)
      call y%set_dev()
    class default
      if (xx%is_dev()) call xx%sync()
      if (y%is_dev())  call y%sync()
      call y%mlt(xx%v,info)
      call y%set_host()
    end select

  end subroutine d_gpu_mlt_v

  subroutine d_gpu_mlt_a(x, y, info)
    use psi_serial_mod
    implicit none 
    real(psb_dpk_), intent(in)           :: x(:)
    class(psb_d_vect_gpu), intent(inout) :: y
    integer(psb_ipk_), intent(out)       :: info
    integer(psb_ipk_) :: i, n
    
    info = 0    
    call y%sync()
    call y%psb_d_base_vect_type%mlt(x,info)
    call y%set_host()
  end subroutine d_gpu_mlt_a

  subroutine d_gpu_mlt_a_2(alpha,x,y,beta,z,info)
    use psi_serial_mod
    implicit none 
    real(psb_dpk_), intent(in)           :: alpha,beta
    real(psb_dpk_), intent(in)           :: x(:)
    real(psb_dpk_), intent(in)           :: y(:)
    class(psb_d_vect_gpu), intent(inout) :: z
    integer(psb_ipk_), intent(out)       :: info
    integer(psb_ipk_) :: i, n
    
    info = 0    
    if (z%is_dev()) call z%sync()
    call z%psb_d_base_vect_type%mlt(alpha,x,y,beta,info)
    call z%set_host()
  end subroutine d_gpu_mlt_a_2

  subroutine d_gpu_mlt_v_2(alpha,x,y, beta,z,info,conjgx,conjgy)
    use psi_serial_mod
    use psb_string_mod
    implicit none 
    real(psb_dpk_), intent(in)                 :: alpha,beta
    class(psb_d_base_vect_type), intent(inout) :: x
    class(psb_d_base_vect_type), intent(inout) :: y
    class(psb_d_vect_gpu), intent(inout)       :: z
    integer(psb_ipk_), intent(out)             :: info
    character(len=1), intent(in), optional     :: conjgx, conjgy
    integer(psb_ipk_) :: i, n
    logical :: conjgx_, conjgy_

    if (.false.) then 
      ! These are present just for coherence with the
      ! complex versions; they do nothing here. 
      conjgx_=.false.
      if (present(conjgx)) conjgx_ = (psb_toupper(conjgx)=='C')
      conjgy_=.false.
      if (present(conjgy)) conjgy_ = (psb_toupper(conjgy)=='C')
    end if
    
    n = min(x%get_nrows(),y%get_nrows(),z%get_nrows())
    
    !
    ! Need to reconsider BETA in the GPU side
    !  of things.
    !
    info = 0    
    select type(xx => x) 
    type is (psb_d_vect_gpu)
      select type (yy => y) 
      type is (psb_d_vect_gpu)
        if (xx%is_host()) call xx%sync()
        if (yy%is_host()) call yy%sync()
        if ((beta /= dzero).and.(z%is_host())) call z%sync()
        info = axybzMultiVecDevice(n,alpha,xx%deviceVect,&
             & yy%deviceVect,beta,z%deviceVect)
        call z%set_dev()
      class default
        if (xx%is_dev()) call xx%sync()
        if (yy%is_dev()) call yy%sync()
        if ((beta /= dzero).and.(z%is_dev())) call z%sync()
        call z%psb_d_base_vect_type%mlt(alpha,xx,yy,beta,info)
        call z%set_host()
      end select
      
    class default
      if (x%is_dev()) call x%sync()
      if (y%is_dev()) call y%sync()
      if ((beta /= dzero).and.(z%is_dev())) call z%sync()
      call z%psb_d_base_vect_type%mlt(alpha,x,y,beta,info)
      call z%set_host()
    end select
  end subroutine d_gpu_mlt_v_2


  subroutine d_gpu_set_scal(x,val)
    class(psb_d_vect_gpu), intent(inout) :: x
    real(psb_dpk_), intent(in)           :: val
        
    integer(psb_ipk_) :: info

    if (x%is_dev()) call x%sync()
    call x%psb_d_base_vect_type%set_scal(val)
    call x%set_host()
  end subroutine d_gpu_set_scal

  subroutine d_gpu_set_vect(x,val)
    class(psb_d_vect_gpu), intent(inout) :: x
    real(psb_dpk_), intent(in)           :: val(:)
    integer(psb_ipk_) :: nr
    integer(psb_ipk_) :: info

    if (x%is_dev()) call x%sync()
    call x%psb_d_base_vect_type%set_vect(val)
    call x%set_host()

  end subroutine d_gpu_set_vect



  subroutine d_gpu_scal(alpha, x)
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    real(psb_dpk_), intent (in)          :: alpha
    
    if (x%is_dev()) call x%sync()
    call x%psb_d_base_vect_type%scal(alpha)
    call x%set_host()
  end subroutine d_gpu_scal


  function d_gpu_nrm2(n,x) result(res)
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n
    real(psb_dpk_)                       :: res
    integer(psb_ipk_) :: info
    ! WARNING: this should be changed. 
    if (x%is_host()) call x%sync()
    info = nrm2MultiVecDevice(res,n,x%deviceVect)
    
  end function d_gpu_nrm2
  
  function d_gpu_amax(n,x) result(res)
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n
    real(psb_dpk_)                :: res

    if (x%is_dev()) call x%sync()
    res =  maxval(abs(x%v(1:n)))

  end function d_gpu_amax

  function d_gpu_asum(n,x) result(res)
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n
    real(psb_dpk_)                :: res

    if (x%is_dev()) call x%sync()
    res =  sum(abs(x%v(1:n)))

  end function d_gpu_asum
  
  subroutine d_gpu_all(n, x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    integer(psb_ipk_), intent(in)      :: n
    class(psb_d_vect_gpu), intent(out) :: x
    integer(psb_ipk_), intent(out)     :: info
    
    call psb_realloc(n,x%v,info)
    if (info == 0) call x%set_host()
    if (info == 0) call x%sync_space(info)
    if (info /= 0) then 
      info=psb_err_alloc_request_
      call psb_errpush(info,'d_gpu_all',&
           & i_err=(/n,n,n,n,n/))
    end if
  end subroutine d_gpu_all

  subroutine d_gpu_zero(x)
    use psi_serial_mod
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    
    if (allocated(x%v)) x%v=dzero
    call x%set_host()
  end subroutine d_gpu_zero

  subroutine d_gpu_asb(n, x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    integer(psb_ipk_), intent(in)        :: n
    class(psb_d_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(out)       :: info
    
    call x%psb_d_base_vect_type%asb(n,info)
        
    if (info == 0) then 
      if (c_associated(x%deviceVect)) then 
        call freeMultiVecDevice(x%deviceVect)
        x%deviceVect=c_null_ptr
      end if
      call x%sync()
    else
      info=psb_err_internal_error_
      call psb_errpush(info,'d_gpu_asb')
    end if
  end subroutine d_gpu_asb

  subroutine d_gpu_sync_space(x,info)
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(out)       :: info 
    integer(psb_ipk_) :: n
    
    info = 0
    if (x%is_host()) then 
      if (.not.c_associated(x%deviceVect)) then 
        n    = size(x%v)
        info = FallocMultiVecDevice(x%deviceVect,1,n,spgpu_type_double)
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
    
  end subroutine d_gpu_sync_space

  subroutine d_gpu_sync(x)
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: n,info
    
    info = 0
    if (x%is_host()) then 
      if (.not.c_associated(x%deviceVect)) then 
        n    = size(x%v)
        info = FallocMultiVecDevice(x%deviceVect,1,n,spgpu_type_double)
      end if
      if (info == 0) &
           & info = writeMultiVecDevice(x%deviceVect,x%v)
    else if (x%is_dev()) then 
      info = readMultiVecDevice(x%deviceVect,x%v)
    end if
    if (info == 0)  call x%set_sync()
    if (info /= 0) then
      info=psb_err_internal_error_
      call psb_errpush(info,'d_gpu_sync')
    end if
    
  end subroutine d_gpu_sync

  subroutine d_gpu_free(x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    class(psb_d_vect_gpu), intent(inout)  :: x
    integer(psb_ipk_), intent(out)        :: info
    
    info = 0
    if (c_associated(x%deviceVect)) then 
      call freeMultiVecDevice(x%deviceVect)
      x%deviceVect=c_null_ptr
    end if
    if (allocated(x%buffer)) then 
      call inner_unregister(x%buffer)
      deallocate(x%buffer, stat=info)
    end if

    if (allocated(x%v)) deallocate(x%v, stat=info)
    call x%set_sync()
  end subroutine d_gpu_free

#ifdef HAVE_FINAL
  subroutine d_gpu_vect_finalize(x)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    type(psb_d_vect_gpu), intent(inout)  :: x
    integer(psb_ipk_)        :: info
    
    info = 0
    if (c_associated(x%deviceVect)) then 
      call freeMultiVecDevice(x%deviceVect)
      x%deviceVect=c_null_ptr
    end if
    if (allocated(x%buffer)) then 
      call inner_unregister(x%buffer)
      deallocate(x%buffer, stat=info)
    end if

    if (allocated(x%v)) deallocate(x%v, stat=info)
    call x%set_sync()
  end subroutine d_gpu_vect_finalize
#endif

  subroutine d_gpu_ins_a(n,irl,val,dupl,x,info)
    use psi_serial_mod
    implicit none 
    class(psb_d_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n, dupl
    integer(psb_ipk_), intent(in)        :: irl(:)
    real(psb_dpk_), intent(in)           :: val(:)
    integer(psb_ipk_), intent(out)       :: info

    integer(psb_ipk_) :: i

    info = 0
    if (x%is_dev()) call x%sync()
    call x%psb_d_base_vect_type%ins(n,irl,val,dupl,info)
    call x%set_host()

  end subroutine d_gpu_ins_a

#endif

end module psb_d_gpu_vect_mod
