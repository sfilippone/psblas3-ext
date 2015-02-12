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
  

module psb_z_gpu_vect_mod
  use iso_c_binding
  use psb_const_mod
  use psb_error_mod
  use psb_z_vect_mod
  use psb_i_vect_mod
#ifdef HAVE_SPGPU
  ! use psb_gpu_env_mod
  use psb_i_gpu_vect_mod
  use psb_z_vectordev_mod
#endif

  integer(psb_ipk_), parameter, private :: is_host = -1
  integer(psb_ipk_), parameter, private :: is_sync = 0 
  integer(psb_ipk_), parameter, private :: is_dev  = 1 
  
  type, extends(psb_z_base_vect_type) ::  psb_z_vect_gpu
#ifdef HAVE_SPGPU
    integer     :: state      = is_host
    type(c_ptr) :: deviceVect = c_null_ptr
    complex(c_double_complex), allocatable :: pinned_buffer(:)
    type(c_ptr) :: d_p_buf = c_null_ptr
    complex(c_double_complex), allocatable :: buffer(:)
    type(c_ptr) :: d_buf = c_null_ptr
    integer :: d_buf_sz = 0
    type(c_ptr) :: i_buf = c_null_ptr
    integer :: i_buf_sz = 0
  contains
    procedure, pass(x) :: get_nrows => z_gpu_get_nrows
    procedure, nopass  :: get_fmt   => z_gpu_get_fmt
    procedure, pass(x) :: dot_v    => z_gpu_dot_v
    procedure, pass(x) :: dot_a    => z_gpu_dot_a
    procedure, pass(y) :: axpby_v  => z_gpu_axpby_v
    procedure, pass(y) :: axpby_a  => z_gpu_axpby_a
    procedure, pass(y) :: mlt_v    => z_gpu_mlt_v
    procedure, pass(y) :: mlt_a    => z_gpu_mlt_a
    procedure, pass(z) :: mlt_a_2  => z_gpu_mlt_a_2
    procedure, pass(z) :: mlt_v_2  => z_gpu_mlt_v_2
    procedure, pass(x) :: scal     => z_gpu_scal
    procedure, pass(x) :: nrm2     => z_gpu_nrm2
    procedure, pass(x) :: amax     => z_gpu_amax
    procedure, pass(x) :: asum     => z_gpu_asum
    procedure, pass(x) :: all      => z_gpu_all
    procedure, pass(x) :: zero     => z_gpu_zero
    procedure, pass(x) :: asb      => z_gpu_asb
    procedure, pass(x) :: sync     => z_gpu_sync
    procedure, pass(x) :: sync_space => z_gpu_sync_space
    procedure, pass(x) :: bld_x    => z_gpu_bld_x
    procedure, pass(x) :: bld_n    => z_gpu_bld_n
    procedure, pass(x) :: free     => z_gpu_free
    procedure, pass(x) :: ins_a    => z_gpu_ins_a
    procedure, pass(x) :: is_host  => z_gpu_is_host
    procedure, pass(x) :: is_dev   => z_gpu_is_dev
    procedure, pass(x) :: is_sync  => z_gpu_is_sync
    procedure, pass(x) :: set_host => z_gpu_set_host
    procedure, pass(x) :: set_dev  => z_gpu_set_dev
    procedure, pass(x) :: set_sync => z_gpu_set_sync
    procedure, pass(x) :: set_scal => z_gpu_set_scal
    procedure, pass(x) :: set_vect => z_gpu_set_vect
    procedure, pass(x) :: gthzv_x  => z_gpu_gthzv_x
    procedure, pass(y) :: sctb     => z_gpu_sctb
    procedure, pass(y) :: sctb_x   => z_gpu_sctb_x
#ifdef HAVE_FINAL
    final              :: z_gpu_vect_finalize
#endif
#endif
  end type psb_z_vect_gpu

  public  :: psb_z_vect_gpu_
  private :: constructor
  interface psb_z_vect_gpu_
    module procedure constructor
  end interface psb_z_vect_gpu_

contains
  
  function constructor(x) result(this)
    complex(psb_dpk_)       :: x(:)
    type(psb_z_vect_gpu) :: this
    integer(psb_ipk_) :: info

    this%v = x
    call this%asb(size(x),info)

  end function constructor
    
#ifdef HAVE_SPGPU

  subroutine z_gpu_gthzv_x(i,n,idx,x,y)
    use psb_gpu_env_mod
    use psi_serial_mod
    integer(psb_ipk_) :: i,n
    class(psb_i_base_vect_type) :: idx
    complex(psb_dpk_) ::  y(:)
    class(psb_z_vect_gpu) :: x
    integer ::  info, ni

    info = 0 

    select type(ii=> idx) 
    class is (psb_i_vect_gpu) 
      if (ii%is_host()) call ii%sync()
      if (x%is_host())  call x%sync()

      if (psb_gpu_DeviceHasUVA()) then 
!!$        write(*,*) 'Pinned memory version'
        if (allocated(x%pinned_buffer)) then  
          if (size(x%pinned_buffer) < n) then 
            call inner_unregister(x%pinned_buffer)
            deallocate(x%pinned_buffer, stat=info)
          end if
        end if

        if (.not.allocated(x%pinned_buffer)) then
          allocate(x%pinned_buffer(n),stat=info)
          if (info == 0) info = inner_register(x%pinned_buffer,x%d_p_buf)        
          if (info /= 0) &
               & write(0,*) 'Error from inner_register ',info
        endif
        info = igathMultiVecDeviceDoubleComplexVecIdx(x%deviceVect,&
             & 0, i, n, ii%deviceVect, x%d_p_buf, 1)
        !call psb_cudaSync()
        y(1:n) = x%pinned_buffer(1:n)

      else
!!$        write(*,*) 'Gather/scatter version 1'
        if (allocated(x%buffer)) then 
          if (size(x%buffer) < n) then 
            deallocate(x%buffer, stat=info)
          end if
        end if

        if (.not.allocated(x%buffer)) then
          allocate(x%buffer(n),stat=info)
        end if

        if (x%d_buf_sz < n) then 
          if (c_associated(x%d_buf)) then 
            call freeDoubleComplex(x%d_buf)
          end if
          info =  allocateDoubleComplex(x%d_buf,n)
          x%d_buf_sz=n
        end if
        if (info == 0) &
             & info = igathMultiVecDeviceDoubleComplexVecIdx(x%deviceVect,&
             & 0, i, n, ii%deviceVect, x%d_buf, 1)
        !call psb_cudaSync()
        if (info == 0) &
             &  info = readDoubleComplex(x%d_buf,y,n)
        !call psb_cudaSync()

      endif

    class default
      ! Do not go for brute force, but move the index vector
      ni = size(ii%v)

      if (x%i_buf_sz < ni) then 
        if (c_associated(x%i_buf)) then 
          call freeInt(x%i_buf)
        end if
        info =  allocateInt(x%i_buf,ni)
        x%i_buf_sz=ni
      end if
      if (allocated(x%buffer)) then 
        if (size(x%buffer) < n) then 
          deallocate(x%buffer, stat=info)
        end if
      end if

      if (.not.allocated(x%buffer)) then
        allocate(x%buffer(n),stat=info)
      end if

      if (x%d_buf_sz < n) then 
        if (c_associated(x%d_buf)) then 
          call freeDoubleComplex(x%d_buf)
        end if
        info =  allocateDoubleComplex(x%d_buf,n)
        x%d_buf_sz=n
      end if

      if (info == 0) &
           & info = writeInt(x%i_buf,ii%v,ni)
      ! call x%gth(n,ii%v(i:),y)
      if (info == 0) &
           & info = igathMultiVecDeviceDoubleComplex(x%deviceVect,&
           & 0, i, n, x%i_buf, x%d_buf, 1)
      if (info == 0) &
           &  info = readDoubleComplex(x%d_buf,y,n)

    end select

  end subroutine z_gpu_gthzv_x


  subroutine z_gpu_sctb(n,idx,x,beta,y)
    implicit none
    !use psb_const_mod
    integer(psb_ipk_)     :: n, idx(:)
    complex(psb_dpk_)        :: beta, x(:)
    class(psb_z_vect_gpu) :: y
    integer(psb_ipk_)     :: info

    if (n == 0) return
    
    if (y%is_dev())  call y%sync()
          
    call y%psb_z_base_vect_type%sctb(n,idx,x,beta)
    call y%set_host()

  end subroutine z_gpu_sctb

  subroutine z_gpu_sctb_x(i,n,idx,x,beta,y)
    use psb_gpu_env_mod
    use psi_serial_mod
    integer(psb_ipk_) :: i, n
    class(psb_i_base_vect_type) :: idx
    complex(psb_dpk_) :: beta, x(:)
    class(psb_z_vect_gpu) :: y
    integer :: info, ni

    select type(ii=> idx) 
    class is (psb_i_vect_gpu) 
      if (ii%is_host()) call ii%sync()
      if (y%is_host())  call y%sync()

      if (psb_gpu_DeviceHasUVA()) then 
!!$        write(*,*) 'Pinned memory version'
        if (allocated(y%pinned_buffer)) then  
          if (size(y%pinned_buffer) < n) then 
            call inner_unregister(y%pinned_buffer)
            deallocate(y%pinned_buffer, stat=info)
          end if
        end if

        if (.not.allocated(y%pinned_buffer)) then
          allocate(y%pinned_buffer(n),stat=info)
          if (info == 0) info = inner_register(y%pinned_buffer,y%d_p_buf)        
          if (info /= 0) &
               & write(0,*) 'Error from inner_register ',info
        endif
        y%buffer(1:n) = x(1:n) 
        call psb_cudaSync()   
        info = iscatMultiVecDeviceDoubleComplexVecIdx(y%deviceVect,&
             & 0, i, n, ii%deviceVect, y%d_p_buf, 1,beta)
      else
        
        if (allocated(y%buffer)) then 
          if (size(y%buffer) < n) then 
            deallocate(y%buffer, stat=info)
          end if
        end if
        
        if (.not.allocated(y%buffer)) then
          allocate(y%buffer(n),stat=info)
        end if

        if (y%d_buf_sz < n) then 
          if (c_associated(y%d_buf)) then 
            call freeDoubleComplex(y%d_buf)
          end if
          info =  allocateDoubleComplex(y%d_buf,n)
          y%d_buf_sz=n
        end if
        info = writeDoubleComplex(y%d_buf,x,n)
        info = iscatMultiVecDeviceDoubleComplexVecIdx(y%deviceVect,&
             & 0, i, n, ii%deviceVect, y%d_buf, 1,beta)

      end if
      
    class default
      !call y%sct(n,ii%v(i:),x,beta)
            ni = size(ii%v)

      if (y%i_buf_sz < ni) then 
        if (c_associated(y%i_buf)) then 
          call freeInt(y%i_buf)
        end if
        info =  allocateInt(y%i_buf,ni)
        y%i_buf_sz=ni
      end if
      if (allocated(y%buffer)) then 
        if (size(y%buffer) < n) then 
          deallocate(y%buffer, stat=info)
        end if
      end if

      if (.not.allocated(y%buffer)) then
        allocate(y%buffer(n),stat=info)
      end if

      if (y%d_buf_sz < n) then 
        if (c_associated(y%d_buf)) then 
          call freeDoubleComplex(y%d_buf)
        end if
        info =  allocateDoubleComplex(y%d_buf,n)
        y%d_buf_sz=n
      end if

      if (info == 0) &
           & info = writeInt(y%i_buf,ii%v,ni)
      info = writeDoubleComplex(y%d_buf,x,n)
      info = iscatMultiVecDeviceDoubleComplex(y%deviceVect,&
           & 0, i, n, y%i_buf, y%d_buf, 1,beta)


    end select
    
    call y%set_dev()

  end subroutine z_gpu_sctb_x


  subroutine z_gpu_bld_x(x,this)
    use psb_base_mod
    complex(psb_dpk_), intent(in)           :: this(:)
    class(psb_z_vect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: info

    call psb_realloc(size(this),x%v,info)
    if (info /= 0) then 
      info=psb_err_alloc_request_
      call psb_errpush(info,'z_gpu_bld_x',&
           & i_err=(/size(this),izero,izero,izero,izero/))
    end if
    x%v(:)  = this(:) 
    call x%set_host()
    call x%sync()

  end subroutine z_gpu_bld_x

  subroutine z_gpu_bld_n(x,n)
    integer(psb_ipk_), intent(in) :: n
    class(psb_z_vect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: info

    call x%all(n,info)
    if (info /= 0) then 
      call psb_errpush(info,'z_gpu_bld_n',i_err=(/n,n,n,n,n/))
    end if
    
  end subroutine z_gpu_bld_n


  subroutine z_gpu_set_host(x)
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    
    x%state = is_host
  end subroutine z_gpu_set_host

  subroutine z_gpu_set_dev(x)
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    
    x%state = is_dev
  end subroutine z_gpu_set_dev

  subroutine z_gpu_set_sync(x)
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    
    x%state = is_sync
  end subroutine z_gpu_set_sync

  function z_gpu_is_dev(x) result(res)
    implicit none 
    class(psb_z_vect_gpu), intent(in) :: x
    logical  :: res
  
    res = (x%state == is_dev)
  end function z_gpu_is_dev
  
  function z_gpu_is_host(x) result(res)
    implicit none 
    class(psb_z_vect_gpu), intent(in) :: x
    logical  :: res

    res = (x%state == is_host)
  end function z_gpu_is_host

  function z_gpu_is_sync(x) result(res)
    implicit none 
    class(psb_z_vect_gpu), intent(in) :: x
    logical  :: res

    res = (x%state == is_sync)
  end function z_gpu_is_sync

  
  function z_gpu_get_nrows(x) result(res)
    implicit none 
    class(psb_z_vect_gpu), intent(in) :: x
    integer(psb_ipk_) :: res

    res = 0
    if (allocated(x%v)) res = size(x%v)
  end function z_gpu_get_nrows

  function z_gpu_get_fmt() result(res)
    implicit none 
    character(len=5) :: res
    res = 'zGPU'
  end function z_gpu_get_fmt

  function z_gpu_dot_v(n,x,y) result(res)
    implicit none 
    class(psb_z_vect_gpu), intent(inout)       :: x
    class(psb_z_base_vect_type), intent(inout) :: y
    integer(psb_ipk_), intent(in)              :: n
    complex(psb_dpk_)                :: res
    complex(psb_dpk_), external      :: ddot
    integer(psb_ipk_) :: info
    
    res = zzero
    !
    ! Note: this is the gpu implementation.
    !  When we get here, we are sure that X is of
    !  TYPE psb_z_vect
    !
    select type(yy => y)
    type is (psb_z_base_vect_type)
      if (x%is_dev()) call x%sync()
      res = ddot(n,x%v,1,yy%v,1)
    type is (psb_z_vect_gpu)
      if (x%is_host()) call x%sync()
      if (yy%is_host()) call yy%sync()
      info = dotMultiVecDevice(res,n,x%deviceVect,yy%deviceVect)
      if (info /= 0) then 
        info = psb_err_internal_error_
        call psb_errpush(info,'z_gpu_dot_v')
      end if

    class default
      ! y%sync is done in dot_a
      call x%sync()      
      res = y%dot(n,x%v)
    end select

  end function z_gpu_dot_v

  function z_gpu_dot_a(n,x,y) result(res)
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    complex(psb_dpk_), intent(in)           :: y(:)
    integer(psb_ipk_), intent(in)        :: n
    complex(psb_dpk_)                :: res
    complex(psb_dpk_), external      :: ddot
    
    if (x%is_dev()) call x%sync()
    res = ddot(n,y,1,x%v,1)

  end function z_gpu_dot_a
    
  subroutine z_gpu_axpby_v(m,alpha, x, beta, y, info)
    use psi_serial_mod
    implicit none 
    integer(psb_ipk_), intent(in)              :: m
    class(psb_z_base_vect_type), intent(inout) :: x
    class(psb_z_vect_gpu), intent(inout)       :: y
    complex(psb_dpk_), intent (in)                :: alpha, beta
    integer(psb_ipk_), intent(out)             :: info
    integer(psb_ipk_) :: nx, ny

    info = psb_success_

    select type(xx => x)
    type is (psb_z_base_vect_type)
      if ((beta /= zzero).and.(y%is_dev()))&
           & call y%sync()
      call psb_geaxpby(m,alpha,xx%v,beta,y%v,info)
      call y%set_host()
    type is (psb_z_vect_gpu)
      ! Do something different here 
      if ((beta /= zzero).and.y%is_host())&
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

  end subroutine z_gpu_axpby_v

  subroutine z_gpu_axpby_a(m,alpha, x, beta, y, info)
    use psi_serial_mod
    implicit none 
    integer(psb_ipk_), intent(in)        :: m
    complex(psb_dpk_), intent(in)           :: x(:)
    class(psb_z_vect_gpu), intent(inout) :: y
    complex(psb_dpk_), intent (in)          :: alpha, beta
    integer(psb_ipk_), intent(out)       :: info

    if (y%is_dev()) call y%sync()
    call psb_geaxpby(m,alpha,x,beta,y%v,info)
    call y%set_host()
  end subroutine z_gpu_axpby_a

  subroutine z_gpu_mlt_v(x, y, info)
    use psi_serial_mod
    implicit none 
    class(psb_z_base_vect_type), intent(inout) :: x
    class(psb_z_vect_gpu), intent(inout)       :: y
    integer(psb_ipk_), intent(out)             :: info

    integer(psb_ipk_) :: i, n
    
    info = 0    
    n = min(x%get_nrows(),y%get_nrows())
    select type(xx => x)
    type is (psb_z_base_vect_type)
      if (y%is_dev()) call y%sync()
      do i=1, n
        y%v(i) = y%v(i) * xx%v(i)
      end do
      call y%set_host()
    type is (psb_z_vect_gpu)
      ! Do something different here 
      if (y%is_host())  call y%sync()
      if (xx%is_host()) call xx%sync()
      info = axyMultiVecDevice(n,zone,xx%deviceVect,y%deviceVect)
      call y%set_dev()
    class default
      if (xx%is_dev()) call xx%sync()
      if (y%is_dev())  call y%sync()
      call y%mlt(xx%v,info)
      call y%set_host()
    end select

  end subroutine z_gpu_mlt_v

  subroutine z_gpu_mlt_a(x, y, info)
    use psi_serial_mod
    implicit none 
    complex(psb_dpk_), intent(in)           :: x(:)
    class(psb_z_vect_gpu), intent(inout) :: y
    integer(psb_ipk_), intent(out)       :: info
    integer(psb_ipk_) :: i, n
    
    info = 0    
    call y%sync()
    call y%psb_z_base_vect_type%mlt(x,info)
    call y%set_host()
  end subroutine z_gpu_mlt_a

  subroutine z_gpu_mlt_a_2(alpha,x,y,beta,z,info)
    use psi_serial_mod
    implicit none 
    complex(psb_dpk_), intent(in)           :: alpha,beta
    complex(psb_dpk_), intent(in)           :: x(:)
    complex(psb_dpk_), intent(in)           :: y(:)
    class(psb_z_vect_gpu), intent(inout) :: z
    integer(psb_ipk_), intent(out)       :: info
    integer(psb_ipk_) :: i, n
    
    info = 0    
    if (z%is_dev()) call z%sync()
    call z%psb_z_base_vect_type%mlt(alpha,x,y,beta,info)
    call z%set_host()
  end subroutine z_gpu_mlt_a_2

  subroutine z_gpu_mlt_v_2(alpha,x,y, beta,z,info,conjgx,conjgy)
    use psi_serial_mod
    use psb_string_mod
    implicit none 
    complex(psb_dpk_), intent(in)                 :: alpha,beta
    class(psb_z_base_vect_type), intent(inout) :: x
    class(psb_z_base_vect_type), intent(inout) :: y
    class(psb_z_vect_gpu), intent(inout)       :: z
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
    type is (psb_z_vect_gpu)
      select type (yy => y) 
      type is (psb_z_vect_gpu)
        if (xx%is_host()) call xx%sync()
        if (yy%is_host()) call yy%sync()
        if ((beta /= zzero).and.(z%is_host())) call z%sync()
        info = axybzMultiVecDevice(n,alpha,xx%deviceVect,&
             & yy%deviceVect,beta,z%deviceVect)
        call z%set_dev()
      class default
        if (xx%is_dev()) call xx%sync()
        if (yy%is_dev()) call yy%sync()
        if ((beta /= zzero).and.(z%is_dev())) call z%sync()
        call z%psb_z_base_vect_type%mlt(alpha,xx,yy,beta,info)
        call z%set_host()
      end select
      
    class default
      if (x%is_dev()) call x%sync()
      if (y%is_dev()) call y%sync()
      if ((beta /= zzero).and.(z%is_dev())) call z%sync()
      call z%psb_z_base_vect_type%mlt(alpha,x,y,beta,info)
      call z%set_host()
    end select
  end subroutine z_gpu_mlt_v_2


  subroutine z_gpu_set_scal(x,val)
    class(psb_z_vect_gpu), intent(inout) :: x
    complex(psb_dpk_), intent(in)           :: val
        
    integer(psb_ipk_) :: info

    if (x%is_dev()) call x%sync()
    call x%psb_z_base_vect_type%set_scal(val)
    call x%set_host()
  end subroutine z_gpu_set_scal

  subroutine z_gpu_set_vect(x,val)
    class(psb_z_vect_gpu), intent(inout) :: x
    complex(psb_dpk_), intent(in)           :: val(:)
    integer(psb_ipk_) :: nr
    integer(psb_ipk_) :: info

    if (x%is_dev()) call x%sync()
    call x%psb_z_base_vect_type%set_vect(val)
    call x%set_host()

  end subroutine z_gpu_set_vect

  subroutine z_gpu_scal(alpha, x)
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    complex(psb_dpk_), intent (in)          :: alpha
    
    if (x%is_dev()) call x%sync()
    call x%psb_z_base_vect_type%scal(alpha)
    call x%set_host()
  end subroutine z_gpu_scal


  function z_gpu_nrm2(n,x) result(res)
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n
    real(psb_dpk_)                       :: res
    integer(psb_ipk_) :: info
    ! WARNING: this should be changed. 
    if (x%is_host()) call x%sync()
    info = nrm2MultiVecDeviceComplex(res,n,x%deviceVect)
    
  end function z_gpu_nrm2
  
  function z_gpu_amax(n,x) result(res)
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n
    real(psb_dpk_)                :: res
    integer(psb_ipk_) :: info

    if (x%is_host()) call x%sync()
    info = amaxMultiVecDeviceComplex(res,n,x%deviceVect)

  end function z_gpu_amax

  function z_gpu_asum(n,x) result(res)
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n
    real(psb_dpk_)                :: res
    integer(psb_ipk_) :: info

    if (x%is_host()) call x%sync()
    info = asumMultiVecDeviceComplex(res,n,x%deviceVect)

  end function z_gpu_asum
  
  subroutine z_gpu_all(n, x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    integer(psb_ipk_), intent(in)      :: n
    class(psb_z_vect_gpu), intent(out) :: x
    integer(psb_ipk_), intent(out)     :: info
    
    call psb_realloc(n,x%v,info)
    if (info == 0) call x%set_host()
    if (info == 0) call x%sync_space(info)
    if (info /= 0) then 
      info=psb_err_alloc_request_
      call psb_errpush(info,'z_gpu_all',&
           & i_err=(/n,n,n,n,n/))
    end if
  end subroutine z_gpu_all

  subroutine z_gpu_zero(x)
    use psi_serial_mod
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    
    if (allocated(x%v)) x%v=zzero
    call x%set_host()
  end subroutine z_gpu_zero

  subroutine z_gpu_asb(n, x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    integer(psb_ipk_), intent(in)        :: n
    class(psb_z_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(out)       :: info
    integer(psb_ipk_) :: nd
    
    if (x%is_dev()) then 
      nd  = getMultiVecDeviceSize(x%deviceVect)
      if (nd < n) then 
        call x%sync()
        call x%psb_z_base_vect_type%asb(n,info)      
        if (info == psb_success_) call x%sync_space(info)
        call x%set_host()
      end if
    else   !
      if (x%get_nrows()<n) then 
        call x%psb_z_base_vect_type%asb(n,info)      
        if (info == psb_success_) call x%sync_space(info)
        call x%set_host()      
      end if
    end if

  end subroutine z_gpu_asb

  subroutine z_gpu_sync_space(x,info)
    use psb_base_mod, only : psb_realloc
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(out)       :: info 
    integer(psb_ipk_) :: nh, nd
    
    info = 0
    if (x%is_dev()) then 
      ! 
      if (.not.allocated(x%v)) then 
        nh = 0
      else
        nh    = size(x%v)
      end if
      nd  = getMultiVecDeviceSize(x%deviceVect)
      if (nh < nd ) then 
        call psb_realloc(nd,x%v,info)
      end if
    else  !    if (x%is_host()) then 
      if (.not.allocated(x%v)) then 
        nh = 0
      else
        nh    = size(x%v)
      end if
      if (c_associated(x%deviceVect)) then 
        nd  = getMultiVecDeviceSize(x%deviceVect)
        if (nd < nh ) then 
          call freeMultiVecDevice(x%deviceVect)
          x%deviceVect=c_null_ptr
        end if
      end if
      if (.not.c_associated(x%deviceVect)) then 
        info = FallocMultiVecDevice(x%deviceVect,1,nh,spgpu_type_complex_double)
        if  (info /= 0) then 
          if (info == spgpu_outofmem) then 
            info = psb_err_alloc_request_
          end if
        end if
      end if
    end if
    
  end subroutine z_gpu_sync_space

  subroutine z_gpu_sync(x)
    use psb_base_mod, only : psb_realloc
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: n,info
    
    info = 0
    if (x%is_host()) then 
      if (.not.c_associated(x%deviceVect)) then 
        n    = size(x%v)
        info = FallocMultiVecDevice(x%deviceVect,1,n,spgpu_type_complex_double)
      end if
      if (info == 0) &
           & info = writeMultiVecDevice(x%deviceVect,x%v)
    else if (x%is_dev()) then 
      n    = getMultiVecDeviceSize(x%deviceVect)
      if (.not.allocated(x%v)) then 
!!$        write(0,*) 'Incoherent situation : x%v not allocated'
        call psb_realloc(n,x%v,info)
      end if
      if ((n > size(x%v)).or.(n > x%get_nrows())) then 
!!$        write(0,*) 'Incoherent situation : sizes',n,size(x%v),x%get_nrows()
        call psb_realloc(n,x%v,info)
      end if
      info = readMultiVecDevice(x%deviceVect,x%v)
    end if
    if (info == 0)  call x%set_sync()
    if (info /= 0) then
      info=psb_err_internal_error_
      call psb_errpush(info,'d_gpu_sync')
    end if
    
  end subroutine z_gpu_sync

  subroutine z_gpu_free(x, info)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    class(psb_z_vect_gpu), intent(inout)  :: x
    integer(psb_ipk_), intent(out)        :: info
    
    info = 0
    if (c_associated(x%deviceVect)) then 
      call freeMultiVecDevice(x%deviceVect)
      x%deviceVect=c_null_ptr
    end if
    if (allocated(x%pinned_buffer)) then 
      call inner_unregister(x%pinned_buffer)
      deallocate(x%pinned_buffer, stat=info)
    end if
    if (allocated(x%buffer)) then 
      deallocate(x%buffer, stat=info)
    end if
    if (c_associated(x%d_buf)) &
         &  call freeDoubleComplex(x%d_buf)
    if (c_associated(x%i_buf)) &
         &  call freeInt(x%i_buf)
    x%d_buf_sz=0
    x%i_buf_sz=0

    if (allocated(x%v)) deallocate(x%v, stat=info)
    call x%set_sync()
  end subroutine z_gpu_free

#ifdef HAVE_FINAL
  subroutine z_gpu_vect_finalize(x)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    type(psb_z_vect_gpu), intent(inout)  :: x
    integer(psb_ipk_)        :: info
    
    info = 0
    call x%free(info)
  end subroutine z_gpu_vect_finalize
#endif

  subroutine z_gpu_ins_a(n,irl,val,dupl,x,info)
    use psi_serial_mod
    implicit none 
    class(psb_z_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)        :: n, dupl
    integer(psb_ipk_), intent(in)        :: irl(:)
    complex(psb_dpk_), intent(in)           :: val(:)
    integer(psb_ipk_), intent(out)       :: info

    integer(psb_ipk_) :: i

    info = 0
    if (x%is_dev()) call x%sync()
    call x%psb_z_base_vect_type%ins(n,irl,val,dupl,info)
    call x%set_host()

  end subroutine z_gpu_ins_a

#endif

end module psb_z_gpu_vect_mod
