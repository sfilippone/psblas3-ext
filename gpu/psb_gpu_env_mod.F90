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
  

module psb_gpu_env_mod
  use psb_const_mod
  !use psb_penv_mod
  !use psi_comm_buffers_mod, only : psb_buffer_queue
  use iso_c_binding

!  interface psb_gpu_init
!    module procedure  psb_gpu_init
!  end interface
#if defined(HAVE_CUDA)
  interface 
    function psb_gpuGetHandle() &
         & result(res) bind(c,name='psb_gpuGetHandle')
      use iso_c_binding   
      type(c_ptr)		:: res
    end function psb_gpuGetHandle
  end interface

  interface 
    function psb_gpuGetStream() &
         & result(res) bind(c,name='psb_gpuGetStream')
      use iso_c_binding   
      type(c_ptr)		:: res
    end function psb_gpuGetStream
  end interface

  interface 
    function psb_C_gpu_init(dev) &
         & result(res) bind(c,name='gpuInit')
      use iso_c_binding   
      integer(c_int),value	:: dev
      integer(c_int)		:: res
    end function psb_C_gpu_init
  end interface

  interface 
    function psb_cuda_getDeviceCount() &
         & result(res) bind(c,name='getDeviceCount')
      use iso_c_binding   
      integer(c_int)		:: res
    end function psb_cuda_getDeviceCount
  end interface

  interface 
    function psb_cuda_getDevice() &
         & result(res) bind(c,name='getDevice')
      use iso_c_binding   
      integer(c_int)		:: res
    end function psb_cuda_getDevice
  end interface

  interface 
    function psb_cuda_setDevice(dev) &
         & result(res) bind(c,name='setDevice')
      use iso_c_binding   
      integer(c_int), value	:: dev
      integer(c_int)		:: res
    end function psb_cuda_setDevice
  end interface

  interface 
    function psb_C_DeviceHasUVA() &
         & result(res) bind(c,name='DeviceHasUVA')
      use iso_c_binding   
      integer(c_int)		:: res
    end function psb_C_DeviceHasUVA
  end interface

  interface 
    subroutine psb_gpuCreateHandle() &
         & bind(c,name='psb_gpuCreateHandle')
      use iso_c_binding   
    end subroutine psb_gpuCreateHandle
  end interface

  interface 
    subroutine psb_gpuSetStream(handle,stream) &
         & bind(c,name='psb_gpuSetStream')
      use iso_c_binding   
      type(c_ptr), value :: handle, stream
    end subroutine psb_gpuSetStream
  end interface

  interface 
    subroutine psb_gpuDestroyHandle() &
         & bind(c,name='psb_gpuDestroyHandle')
      use iso_c_binding   
    end subroutine psb_gpuDestroyHandle
  end interface

  interface 
    subroutine psb_cudaSync() &
         & bind(c,name='cudaSync')
      use iso_c_binding   
    end subroutine psb_cudaSync
  end interface

  interface 
    subroutine psb_cudaReset() &
         & bind(c,name='cudaReset')
      use iso_c_binding   
    end subroutine psb_cudaReset
  end interface
#endif

contains
  ! !!!!!!!!!!!!!!!!!!!!!!
  !
  ! Environment handling 
  !
  ! !!!!!!!!!!!!!!!!!!!!!!


  subroutine psb_gpu_init(ictxt,dev)
    use psb_penv_mod
    use psb_const_mod
    use psb_error_mod
    integer, intent(in) :: ictxt
    integer, intent(in), optional :: dev

    integer :: np, npavail, iam, info, count, dev_

#if defined (HAVE_CUDA)
#if defined(SERIAL_MPI) 
    iam = 0
#else
    call psb_info(ictxt,iam,np)
#endif
    count = psb_cuda_getDeviceCount()

    if (present(dev)) then 
      info = psb_C_gpu_init(dev)
    else
      if (count >0) then 
        dev_ = mod(iam,count)
      else
        dev_ = 0
      end if
      info = psb_C_gpu_init(dev_)
    end if
    call psb_gpuCreateHandle()
#endif
  end subroutine psb_gpu_init


  subroutine psb_gpu_DeviceSync()
#if defined(HAVE_CUDA)
    call psb_cudaSync()
#endif
  end subroutine psb_gpu_DeviceSync

  function psb_gpu_getDeviceCount() result(res)
    integer :: res
#if defined(HAVE_CUDA)
    res = psb_cuda_getDeviceCount()
#else 
    res = 0
#endif
  end function psb_gpu_getDeviceCount

  subroutine psb_gpu_exit()
    call psb_gpuDestroyHandle()
    call psb_cudaReset()
  end subroutine psb_gpu_exit

  function psb_gpu_DeviceHasUVA() result(res)
    logical :: res
    res =  (psb_C_DeviceHasUVA() == 1)
  end function psb_gpu_DeviceHasUVA


end module psb_gpu_env_mod
