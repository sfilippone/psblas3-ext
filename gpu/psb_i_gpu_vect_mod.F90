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
  use psb_error_mod
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
    integer(c_int), allocatable :: pinned_buffer(:)
    type(c_ptr) :: dt_p_buf = c_null_ptr
    integer(c_int), allocatable :: buffer(:)
    type(c_ptr) :: dt_buf = c_null_ptr
    integer :: dt_buf_sz = 0
    type(c_ptr) :: i_buf = c_null_ptr
    integer :: i_buf_sz = 0
  contains
    procedure, pass(x) :: get_nrows => i_gpu_get_nrows
    procedure, nopass  :: get_fmt   => i_gpu_get_fmt

    procedure, pass(x) :: all      => i_gpu_all
    procedure, pass(x) :: zero     => i_gpu_zero
    procedure, pass(x) :: asb      => i_gpu_asb
    procedure, pass(x) :: sync     => i_gpu_sync
    procedure, pass(x) :: sync_space => i_gpu_sync_space
    procedure, pass(x) :: bld_x    => i_gpu_bld_x
    procedure, pass(x) :: bld_n    => i_gpu_bld_n
    procedure, pass(x) :: free     => i_gpu_free
    procedure, pass(x) :: ins_a    => i_gpu_ins_a
    procedure, pass(x) :: ins_v    => i_gpu_ins_v
    procedure, pass(x) :: is_host  => i_gpu_is_host
    procedure, pass(x) :: is_dev   => i_gpu_is_dev
    procedure, pass(x) :: is_sync  => i_gpu_is_sync
    procedure, pass(x) :: set_host => i_gpu_set_host
    procedure, pass(x) :: set_dev  => i_gpu_set_dev
    procedure, pass(x) :: set_sync => i_gpu_set_sync
!!$    procedure, pass(x) :: set_scal => i_gpu_set_scal
!!$    procedure, pass(x) :: set_vect => i_gpu_set_vect
    procedure, pass(x) :: gthzv_x  => i_gpu_gthzv_x
    procedure, pass(y) :: sctb     => i_gpu_sctb
    procedure, pass(y) :: sctb_x   => i_gpu_sctb_x
    procedure, pass(x) :: gthzbuf  => i_gpu_gthzbuf
    procedure, pass(y) :: sctb_buf => i_gpu_sctb_buf
    procedure, pass(x) :: new_buffer   => i_gpu_new_buffer
    procedure, nopass  :: device_wait  => i_gpu_device_wait
    procedure, pass(x) :: free_buffer  => i_gpu_free_buffer

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
    integer(psb_ipk_)       :: x(:)
    type(psb_i_vect_gpu) :: this
    integer(psb_ipk_) :: info

    this%v = x
    call this%asb(size(x),info)

  end function constructor
    
#ifdef HAVE_SPGPU

  subroutine i_gpu_device_wait()
    call psb_cudaSync()
  end subroutine i_gpu_device_wait

  subroutine i_gpu_new_buffer(n,x,info)
    use psb_realloc_mod
    use psb_gpu_env_mod
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(in)              :: n
    integer(psb_ipk_), intent(out)             :: info

    
    if (psb_gpu_DeviceHasUVA()) then       
      if (allocated(x%combuf)) then 
        if (size(x%combuf)<n) then 
          call inner_unregister(x%combuf)
          deallocate(x%combuf,stat=info)
        end if
      end if
      if (.not.allocated(x%combuf)) then 
        call psb_realloc(n,x%combuf,info)
        if (info == 0) info = inner_register(x%combuf,x%dt_p_buf)        
      end if
    else
      call psb_realloc(n,x%combuf,info)
    end if
    if (c_associated(x%dt_buf)) then 
      call freeInt(x%dt_buf)
    end if
    info       =  allocateInt(x%dt_buf,n)
    x%dt_buf_sz = n
    if (c_associated(x%i_buf)) then 
      call freeInt(x%i_buf)
    end if
    info       =  allocateInt(x%i_buf,n)
    x%i_buf_sz = n
    
  end subroutine i_gpu_new_buffer

  subroutine i_gpu_free_buffer(x,info)
    use psb_realloc_mod
    use psb_gpu_env_mod
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_), intent(out)             :: info
    
    if (allocated(x%pinned_buffer)) then 
      call inner_unregister(x%pinned_buffer)
      deallocate(x%pinned_buffer, stat=info)
    end if
    if (allocated(x%buffer)) then 
      deallocate(x%buffer, stat=info)
    end if
    if (c_associated(x%dt_buf)) then 
      call freeInt(x%dt_buf)
      x%dt_buf  = c_null_ptr
    end if
    if (c_associated(x%i_buf)) then 
      call freeInt(x%i_buf)
      x%i_buf = c_null_ptr
    end if
    x%dt_buf_sz=0
    x%i_buf_sz=0

  end subroutine i_gpu_free_buffer

  subroutine i_gpu_gthzv_x(i,n,idx,x,y)
    use psb_gpu_env_mod
    use psi_serial_mod
    integer(psb_ipk_) :: i,n
    class(psb_i_base_vect_type) :: idx
    integer(psb_ipk_) ::  y(:)
    class(psb_i_vect_gpu) :: x
    integer ::  info, ni

    info = 0 

    select type(ii=> idx) 
    class is (psb_i_vect_gpu) 
      if (ii%is_host()) call ii%sync()
      if (x%is_host())  call x%sync()

      if (psb_gpu_DeviceHasUVA()) then 
        !
        ! Only need a sync in this branch; in the others
        ! cudamemCpy acts as a sync point.
        !
        if (allocated(x%pinned_buffer)) then  
          if (size(x%pinned_buffer) < n) then 
            call inner_unregister(x%pinned_buffer)
            deallocate(x%pinned_buffer, stat=info)
          end if
        end if

        if (.not.allocated(x%pinned_buffer)) then
          allocate(x%pinned_buffer(n),stat=info)
          if (info == 0) info = inner_register(x%pinned_buffer,x%dt_p_buf)        
          if (info /= 0) &
               & write(0,*) 'Error from inner_register ',info
        endif
        info = igathMultiVecDeviceIntVecIdx(x%deviceVect,&
             & 0, n, i, ii%deviceVect, 1, x%dt_p_buf, 1)
        call psb_cudaSync()
        y(1:n) = x%pinned_buffer(1:n)

      else
        if (allocated(x%buffer)) then 
          if (size(x%buffer) < n) then 
            deallocate(x%buffer, stat=info)
          end if
        end if

        if (.not.allocated(x%buffer)) then
          allocate(x%buffer(n),stat=info)
        end if

        if (x%dt_buf_sz < n) then 
          if (c_associated(x%dt_buf)) then 
            call freeInt(x%dt_buf)
          end if
          info =  allocateInt(x%dt_buf,n)
          x%dt_buf_sz=n
        end if
        if (info == 0) &
             & info = igathMultiVecDeviceIntVecIdx(x%deviceVect,&
             & 0, n, i, ii%deviceVect, 1, x%dt_buf, 1)
        if (info == 0) &
             &  info = readInt(x%dt_buf,y,n)

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

      if (x%dt_buf_sz < n) then 
        if (c_associated(x%dt_buf)) then 
          call freeInt(x%dt_buf)
        end if
        info =  allocateInt(x%dt_buf,n)
        x%dt_buf_sz=n
      end if

      if (info == 0) &
           & info = writeInt(x%i_buf,ii%v,ni)
      if (info == 0) &
           & info = igathMultiVecDeviceInt(x%deviceVect,&
           & 0, n, i, x%i_buf, 1, x%dt_buf, 1)
      if (info == 0) &
           &  info = readInt(x%dt_buf,y,n)

    end select
    
  end subroutine i_gpu_gthzv_x

  subroutine i_gpu_gthzbuf(i,n,idx,x)
    use psb_gpu_env_mod
    use psi_serial_mod
    integer(psb_ipk_) :: i,n
    class(psb_i_base_vect_type) :: idx
    class(psb_i_vect_gpu)       :: x
    integer ::  info, ni

    info = 0 
!!$    write(0,*) 'Starting gth_zbuf'
    if (.not.allocated(x%combuf)) then
      call psb_errpush(psb_err_alloc_dealloc_,'gthzbuf')
      return
    end if

    select type(ii=> idx) 
    class is (psb_i_vect_gpu) 
      if (ii%is_host()) call ii%sync()
      if (x%is_host())  call x%sync()

      if (psb_gpu_DeviceHasUVA()) then 
        info = igathMultiVecDeviceIntVecIdx(x%deviceVect,&
             & 0, n, i, ii%deviceVect, i,x%dt_p_buf, 1)

      else
        info = igathMultiVecDeviceIntVecIdx(x%deviceVect,&
             & 0, n, i, ii%deviceVect, i,x%dt_buf, 1)
        if (info == 0) &
             &  info = readInt(i,x%dt_buf,x%combuf(i:),n,1)
      endif

    class default
      ! Do not go for brute force, but move the index vector
      ni = size(ii%v)
      info = 0 
      if (.not.c_associated(x%i_buf)) then 
        info =  allocateInt(x%i_buf,ni)
        x%i_buf_sz=ni
      end if
      if (info == 0) &
           & info = writeInt(i,x%i_buf,ii%v(i:),n,1)

      if (info == 0) &
           & info = igathMultiVecDeviceInt(x%deviceVect,&
           & 0, n, i, x%i_buf, i,x%dt_buf, 1)
      
      if (info == 0) &
           &  info = readInt(i,x%dt_buf,x%combuf(i:),n,1)

    end select

  end subroutine i_gpu_gthzbuf

  subroutine i_gpu_sctb(n,idx,x,beta,y)
    implicit none
    !use psb_const_mod
    integer(psb_ipk_)     :: n, idx(:)
    integer(psb_ipk_)        :: beta, x(:)
    class(psb_i_vect_gpu) :: y
    integer(psb_ipk_)     :: info

    if (n == 0) return
    
    if (y%is_dev())  call y%sync()
          
    call y%psb_i_base_vect_type%sctb(n,idx,x,beta)
    call y%set_host()

  end subroutine i_gpu_sctb

  subroutine i_gpu_sctb_x(i,n,idx,x,beta,y)
    use psb_gpu_env_mod
    use psi_serial_mod
    integer(psb_ipk_) :: i, n
    class(psb_i_base_vect_type) :: idx
    integer(psb_ipk_) :: beta, x(:)
    class(psb_i_vect_gpu) :: y
    integer :: info, ni

    select type(ii=> idx) 
    class is (psb_i_vect_gpu) 
      if (ii%is_host()) call ii%sync()
      if (y%is_host())  call y%sync()
      
      ! 
      if (psb_gpu_DeviceHasUVA()) then 
        if (allocated(y%pinned_buffer)) then  
          if (size(y%pinned_buffer) < n) then 
            call inner_unregister(y%pinned_buffer)
            deallocate(y%pinned_buffer, stat=info)
          end if
        end if

        if (.not.allocated(y%pinned_buffer)) then
          allocate(y%pinned_buffer(n),stat=info)
          if (info == 0) info = inner_register(y%pinned_buffer,y%dt_p_buf)        
          if (info /= 0) &
               & write(0,*) 'Error from inner_register ',info
        endif
        y%pinned_buffer(1:n) = x(1:n) 
        info = iscatMultiVecDeviceIntVecIdx(y%deviceVect,&
             & 0, n, i, ii%deviceVect, 1, y%dt_p_buf, 1,beta)
      else
        
        if (allocated(y%buffer)) then 
          if (size(y%buffer) < n) then 
            deallocate(y%buffer, stat=info)
          end if
        end if
        
        if (.not.allocated(y%buffer)) then
          allocate(y%buffer(n),stat=info)
        end if

        if (y%dt_buf_sz < n) then 
          if (c_associated(y%dt_buf)) then 
            call freeInt(y%dt_buf)
          end if
          info =  allocateInt(y%dt_buf,n)
          y%dt_buf_sz=n
        end if
        info = writeInt(y%dt_buf,x,n)
        info = iscatMultiVecDeviceIntVecIdx(y%deviceVect,&
             & 0, n, i, ii%deviceVect, 1, y%dt_buf, 1,beta)

      end if
      
    class default
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

      if (y%dt_buf_sz < n) then 
        if (c_associated(y%dt_buf)) then 
          call freeInt(y%dt_buf)
        end if
        info =  allocateInt(y%dt_buf,n)
        y%dt_buf_sz=n
      end if

      if (info == 0) &
           & info = writeInt(y%i_buf,ii%v(i:i+n-1),n)
      info = writeInt(y%dt_buf,x,n)
      info = iscatMultiVecDeviceInt(y%deviceVect,&
           & 0, n, 1, y%i_buf, 1, y%dt_buf, 1,beta)


    end select
    !
    !  Need a sync here to make sure we are not reallocating
    !  the buffers before iscatMulti has finished.
    !
    call psb_cudaSync()       
    call y%set_dev()

  end subroutine i_gpu_sctb_x

  subroutine i_gpu_sctb_buf(i,n,idx,beta,y)
    use psi_serial_mod
    use psb_gpu_env_mod
    implicit none 
    integer(psb_ipk_) :: i, n
    class(psb_i_base_vect_type) :: idx
    integer(psb_ipk_) :: beta
    class(psb_i_vect_gpu) :: y
    integer(psb_ipk_) :: info, ni
    
!!$    write(0,*) 'Starting sctb_buf'
    if (.not.allocated(y%combuf)) then 
      call psb_errpush(psb_err_alloc_dealloc_,'sctb_buf')
      return
    end if
    

    select type(ii=> idx) 
    class is (psb_i_vect_gpu) 
              
      if (ii%is_host()) call ii%sync()
      if (y%is_host())  call y%sync()
      if (psb_gpu_DeviceHasUVA()) then 
        info = iscatMultiVecDeviceIntVecIdx(y%deviceVect,&
             & 0, n, i, ii%deviceVect, i, y%dt_p_buf, 1,beta)
      else 
        info = writeInt(i,y%dt_buf,y%combuf(i:),n,1)
        info = iscatMultiVecDeviceIntVecIdx(y%deviceVect,&
             & 0, n, i, ii%deviceVect, i, y%dt_buf, 1,beta)

      end if

    class default
      !call y%sct(n,ii%v(i:),x,beta)
      ni = size(ii%v)
      info = 0 
      if (.not.c_associated(y%i_buf)) then 
        info =  allocateInt(y%i_buf,ni)
        y%i_buf_sz=ni
      end if
      if (info == 0) &
           & info = writeInt(i,y%i_buf,ii%v(i:),n,1)
      if (info == 0) &
           & info = writeInt(i,y%dt_buf,y%combuf(i:),n,1)
      if (info == 0) info = iscatMultiVecDeviceInt(y%deviceVect,&
           & 0, n, i, y%i_buf, i, y%dt_buf, 1,beta)
    end select
!!$    write(0,*) 'Done sctb_buf'

  end subroutine i_gpu_sctb_buf


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
    if (info /= 0) then 
      call psb_errpush(info,'i_gpu_bld_n',i_err=(/n,n,n,n,n/))
    end if
    
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

  
  function i_gpu_get_nrows(x) result(res)
    implicit none 
    class(psb_i_vect_gpu), intent(in) :: x
    integer(psb_ipk_) :: res

    res = 0
    if (allocated(x%v)) res = size(x%v)
  end function i_gpu_get_nrows

  function i_gpu_get_fmt() result(res)
    implicit none 
    character(len=5) :: res
    res = 'iGPU'
  end function i_gpu_get_fmt
  
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
    integer(psb_ipk_) :: nd
    
    if (x%is_dev()) then 
      nd  = getMultiVecDeviceSize(x%deviceVect)
      if (nd < n) then 
        call x%sync()
        call x%psb_i_base_vect_type%asb(n,info)      
        if (info == psb_success_) call x%sync_space(info)
        call x%set_host()
      end if
    else   !
      if (x%get_nrows()<n) then 
        call x%psb_i_base_vect_type%asb(n,info)      
        if (info == psb_success_) call x%sync_space(info)
        call x%set_host()      
      end if
    end if

  end subroutine i_gpu_asb

  subroutine i_gpu_sync_space(x,info)
    use psb_base_mod, only : psb_realloc
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
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
        info = FallocMultiVecDevice(x%deviceVect,1,nh,spgpu_type_int)
        if  (info /= 0) then 
          if (info == spgpu_outofmem) then 
            info = psb_err_alloc_request_
          end if
        end if
      end if
    end if
    
  end subroutine i_gpu_sync_space

  subroutine i_gpu_sync(x)
    use psb_base_mod, only : psb_realloc
    implicit none 
    class(psb_i_vect_gpu), intent(inout) :: x
    integer(psb_ipk_) :: n,info
    
    info = 0
    if (x%is_host()) then 
      if (.not.c_associated(x%deviceVect)) then 
        n    = size(x%v)
        info = FallocMultiVecDevice(x%deviceVect,1,n,spgpu_type_int)
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
    if (allocated(x%pinned_buffer)) then 
      call inner_unregister(x%pinned_buffer)
      deallocate(x%pinned_buffer, stat=info)
    end if
    if (allocated(x%buffer)) then 
      deallocate(x%buffer, stat=info)
    end if
    if (c_associated(x%dt_buf)) &
         &  call freeInt(x%dt_buf)
    if (c_associated(x%i_buf)) &
         &  call freeInt(x%i_buf)
    x%dt_buf_sz=0
    x%i_buf_sz=0

    if (allocated(x%v)) deallocate(x%v, stat=info)
    call x%set_sync()
  end subroutine i_gpu_free


#ifdef HAVE_FINAL
  subroutine i_gpu_vect_finalize(x)
    use psi_serial_mod
    use psb_realloc_mod
    implicit none 
    type(psb_i_vect_gpu), intent(inout)  :: x
    integer(psb_ipk_)        :: info
    
    info = 0
    call x%free(info)
  end subroutine i_gpu_vect_finalize
#endif

  subroutine i_gpu_ins_v(n,irl,val,dupl,x,info)
    use psi_serial_mod
    implicit none 
    class(psb_i_vect_gpu), intent(inout)        :: x
    integer(psb_ipk_), intent(in)               :: n, dupl
    class(psb_i_base_vect_type), intent(inout)  :: irl
    class(psb_i_base_vect_type), intent(inout)  :: val
    integer(psb_ipk_), intent(out)              :: info

    integer(psb_ipk_) :: i, isz
    logical :: done_gpu

    info = 0
    if (psb_errstatus_fatal()) return 

    done_gpu = .false. 
    select type(virl => irl)
    class is (psb_i_vect_gpu) 
      select type(vval => val)
      class is (psb_i_vect_gpu) 
        if (vval%is_host()) call vval%sync()
        if (virl%is_host()) call virl%sync()
        if (x%is_host())    call x%sync()
        info = geinsMultiVecDeviceInt(n,virl%deviceVect,&
             & vval%deviceVect,dupl,1,x%deviceVect)
        call x%set_dev()
        done_gpu=.true.
      end select
    end select

    if (.not.done_gpu) then 
      if (irl%is_dev()) call irl%sync()
      if (val%is_dev()) call val%sync()
      call x%ins(n,irl%v,val%v,dupl,info)
    end if

    if (info /= 0) then 
      call psb_errpush(info,'gpu_vect_ins')
      return
    end if

  end subroutine i_gpu_ins_v
  
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
    if (x%is_dev()) call x%sync()
    call x%psb_i_base_vect_type%ins(n,irl,val,dupl,info)
    call x%set_host()

  end subroutine i_gpu_ins_a

#endif

end module psb_i_gpu_vect_mod
