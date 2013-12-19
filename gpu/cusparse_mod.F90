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
  
#if 1

module cusparse_mod
  use base_cusparse_mod
  use s_cusparse_mod
  use d_cusparse_mod
  use c_cusparse_mod
  use z_cusparse_mod
end module cusparse_mod
#else
module cusparse_mod
  use iso_c_binding 
  ! Interface to CUSPARSE. 

  enum, bind(c)
    enumerator cusparse_status_success
    enumerator cusparse_status_not_initialized
    enumerator cusparse_status_alloc_failed
    enumerator cusparse_status_invalid_value
    enumerator cusparse_status_arch_mismatch
    enumerator cusparse_status_mapping_error
    enumerator cusparse_status_execution_failed
    enumerator cusparse_status_internal_error
    enumerator cusparse_status_matrix_type_not_supported
  end enum

  enum, bind(c)
    enumerator cusparse_matrix_type_general 
    enumerator cusparse_matrix_type_symmetric     
    enumerator cusparse_matrix_type_hermitian 
    enumerator cusparse_matrix_type_triangular 
  end enum
  
  enum, bind(c)
    enumerator cusparse_fill_mode_lower 
    enumerator cusparse_fill_mode_upper
  end enum
  
  enum, bind(c)
    enumerator cusparse_diag_type_non_unit 
    enumerator cusparse_diag_type_unit
  end enum
  
  enum, bind(c)
    enumerator cusparse_index_base_zero 
    enumerator cusparse_index_base_one
  end enum
  
  enum, bind(c)
    enumerator cusparse_operation_non_transpose  
    enumerator cusparse_operation_transpose
    enumerator cusparse_operation_conjugate_transpose
  end enum
  
  enum, bind(c)
    enumerator cusparse_direction_row
    enumerator cusparse_direction_column
  end enum

  type(c_ptr), save :: cusparse_context = c_null_ptr

  type, bind(c) :: d_Cmat
    type(c_ptr) :: Mat = c_null_ptr
  end type d_Cmat

  type, bind(c) :: s_Cmat
    type(c_ptr) :: Mat = c_null_ptr
  end type s_Cmat

  type, bind(c) :: s_Hmat
     type(c_ptr) :: Mat = c_null_ptr
  end type s_Hmat

  type, bind(c) :: d_Hmat
     type(c_ptr) :: Mat = c_null_ptr
  end type d_Hmat

  type, bind(c) :: c_Hmat
     type(c_ptr) :: Mat = c_null_ptr
  end type s_Hmat

  type, bind(c) :: z_Hmat
     type(c_ptr) :: Mat = c_null_ptr
  end type d_Hmat

#if defined(HAVE_CUDA) && defined(HAVE_SPGPU)
  
  interface 
    function FcusparseCreate(context) &
         & bind(c,name="FcusparseCreate") result(res)
      use iso_c_binding
      type(c_ptr)    :: context
      integer(c_int) :: res
    end function FcusparseCreate
  end interface

  interface 
    function FcusparseDestroy(context) &
         & bind(c,name="FcusparseDestroy") result(res)
      use iso_c_binding
      type(c_ptr)    :: context
      integer(c_int) :: res
    end function FcusparseDestroy
  end interface

  interface CSRGDeviceFree
    function d_CSRGDeviceFree(Mat) &
         & bind(c,name="d_CSRGDeviceFree") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)   :: Mat
      integer(c_int) :: res
    end function d_CSRGDeviceFree
    function s_CSRGDeviceFree(Mat) &
         & bind(c,name="s_CSRGDeviceFree") result(res)
      use iso_c_binding
      import  s_Cmat
      type(s_Cmat)   :: Mat
      integer(c_int) :: res
    end function s_CSRGDeviceFree
  end interface

  interface CSRGDeviceSetMatType
     function s_CSRGDeviceSetMatType(Mat,type) &
         & bind(c,name="s_CSRGDeviceSetMatType") result(res)
      use iso_c_binding
      import  s_Cmat
      type(s_Cmat)          :: Mat
      integer(c_int),value  :: type
      integer(c_int)        :: res
    end function s_CSRGDeviceSetMatType
    function d_CSRGDeviceSetMatType(Mat,type) &
         & bind(c,name="d_CSRGDeviceSetMatType") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)          :: Mat
      integer(c_int),value  :: type
      integer(c_int)        :: res
    end function d_CSRGDeviceSetMatType
  end interface

  interface CSRGDeviceSetMatFillMode
     function s_CSRGDeviceSetMatFillMode(Mat,type) &
          & bind(c,name="s_CSRGDeviceSetMatFillMode") result(res)
       use iso_c_binding
      import  s_Cmat
      type(s_Cmat)          :: Mat
      integer(c_int),value  :: type
      integer(c_int)        :: res
    end function s_CSRGDeviceSetMatFillMode
    function d_CSRGDeviceSetMatFillMode(Mat,type) &
         & bind(c,name="d_CSRGDeviceSetMatFillMode") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)          :: Mat
      integer(c_int),value  :: type
      integer(c_int)        :: res
    end function d_CSRGDeviceSetMatFillMode
  end interface

  interface CSRGDeviceSetMatDiagType
     function s_CSRGDeviceSetMatDiagType(Mat,type) &
         & bind(c,name="s_CSRGDeviceSetMatDiagType") result(res)
      use iso_c_binding
      import  s_Cmat
      type(s_Cmat)          :: Mat
      integer(c_int),value  :: type
      integer(c_int)        :: res
    end function s_CSRGDeviceSetMatDiagType
    function d_CSRGDeviceSetMatDiagType(Mat,type) &
         & bind(c,name="d_CSRGDeviceSetMatDiagType") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)          :: Mat
      integer(c_int),value  :: type
      integer(c_int)        :: res
    end function d_CSRGDeviceSetMatDiagType
  end interface
    
  interface CSRGDeviceSetMatIndexBase
     function s_CSRGDeviceSetMatIndexBase(Mat,type) &
         & bind(c,name="s_CSRGDeviceSetMatIndexBase") result(res)
      use iso_c_binding
      import  s_Cmat
      type(s_Cmat)          :: Mat
      integer(c_int),value  :: type
      integer(c_int)        :: res
    end function s_CSRGDeviceSetMatIndexBase
    function d_CSRGDeviceSetMatIndexBase(Mat,type) &
         & bind(c,name="d_CSRGDeviceSetMatIndexBase") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)          :: Mat
      integer(c_int),value  :: type
      integer(c_int)        :: res
    end function d_CSRGDeviceSetMatIndexBase
  end interface
    
  interface CSRGDeviceCsrsmAnalysis
     function s_CSRGDeviceCsrsmAnalysis(Mat) &
         & bind(c,name="s_CSRGDeviceCsrsmAnalysis") result(res)
      use iso_c_binding
      import  s_Cmat
      type(s_Cmat)          :: Mat
      integer(c_int)        :: res
    end function s_CSRGDeviceCsrsmAnalysis
    function d_CSRGDeviceCsrsmAnalysis(Mat) &
         & bind(c,name="d_CSRGDeviceCsrsmAnalysis") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)          :: Mat
      integer(c_int)        :: res
    end function d_CSRGDeviceCsrsmAnalysis
  end interface

  interface CSRGDeviceAlloc
     function s_CSRGDeviceAlloc(Mat,nr,nc,nz) &
         & bind(c,name="s_CSRGDeviceAlloc") result(res)
      use iso_c_binding
      import  s_Cmat
      type(s_Cmat)          :: Mat
      integer(c_int), value :: nr, nc, nz
      integer(c_int)        :: res
    end function s_CSRGDeviceAlloc
    function d_CSRGDeviceAlloc(Mat,nr,nc,nz) &
         & bind(c,name="d_CSRGDeviceAlloc") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)          :: Mat
      integer(c_int), value :: nr, nc, nz
      integer(c_int)        :: res
    end function d_CSRGDeviceAlloc
  end interface
  
  interface spsvCSRGDevice
     function s_spsvCSRGDevice(Mat,alpha,x,beta,y) &
         &  bind(c,name="s_spsvCSRGDevice") result(res)
      use iso_c_binding
      import  s_Cmat
      type(s_Cmat)          :: Mat
      type(c_ptr), value    :: x
      type(c_ptr), value    :: y
      real(c_float), value :: alpha,beta
      integer(c_int)        :: res
    end function s_spsvCSRGDevice
    function d_spsvCSRGDevice(Mat,alpha,x,beta,y) &
         &  bind(c,name="d_spsvCSRGDevice") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)          :: Mat
      type(c_ptr), value    :: x
      type(c_ptr), value    :: y
      real(c_double), value :: alpha,beta
      integer(c_int)        :: res
    end function d_spsvCSRGDevice
  end interface
  
  interface spmvCSRGDevice
     function s_spmvCSRGDevice(Mat,alpha,x,beta,y) &
         &  bind(c,name="s_spmvCSRGDevice") result(res)
      use iso_c_binding
      import  s_Cmat
      type(s_Cmat)          :: Mat
      type(c_ptr), value    :: x
      type(c_ptr), value    :: y
      real(c_float), value :: alpha,beta
      integer(c_int)        :: res
    end function s_spmvCSRGDevice
    function d_spmvCSRGDevice(Mat,alpha,x,beta,y) &
         &  bind(c,name="d_spmvCSRGDevice") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)          :: Mat
      type(c_ptr), value    :: x
      type(c_ptr), value    :: y
      real(c_double), value :: alpha,beta
      integer(c_int)        :: res
    end function d_spmvCSRGDevice
  end interface

  interface CSRGHost2Device
     function s_CSRGHost2Device(Mat,m,n,nz,irp,ja,val) &
         &  bind(c,name="s_CSRGHost2Device") result(res)
      use iso_c_binding
      import  s_Cmat
      type(s_Cmat)          :: Mat
      integer(c_int), value :: m,n,nz
      integer(c_int)        :: irp(*), ja(*)
      real(c_float)         :: val(*)
      integer(c_int)        :: res
    end function s_CSRGHost2Device
    function d_CSRGHost2Device(Mat,m,n,nz,irp,ja,val) &
         &  bind(c,name="d_CSRGHost2Device") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)          :: Mat
      integer(c_int), value :: m,n,nz
      integer(c_int)        :: irp(*), ja(*)
      real(c_double)        :: val(*)
      integer(c_int)        :: res
    end function d_CSRGHost2Device
  end interface

  interface CSRGDevice2Host
     function s_CSRGDevice2Host(Mat,m,n,nz,irp,ja,val) &
         &  bind(c,name="s_CSRGDevice2Host") result(res)
      use iso_c_binding
      import  s_Cmat
      type(s_Cmat)          :: Mat
      integer(c_int), value :: m,n,nz
      integer(c_int)        :: irp(*), ja(*)
      real(c_float)         :: val(*)
      integer(c_int)        :: res
    end function s_CSRGDevice2Host
    function d_CSRGDevice2Host(Mat,m,n,nz,irp,ja,val) &
         &  bind(c,name="d_CSRGDevice2Host") result(res)
      use iso_c_binding
      import  d_Cmat
      type(d_Cmat)          :: Mat
      integer(c_int), value :: m,n,nz
      integer(c_int)        :: irp(*), ja(*)
      real(c_double)        :: val(*)
      integer(c_int)        :: res
    end function d_CSRGDevice2Host
  end interface
      
interface HYBGDeviceAlloc
   function s_HYBGDeviceAlloc(Mat,nr,nc,nz) &
          & bind(c,name="s_HYBGDeviceAlloc") result(res)
       use iso_c_binding
       import  s_hmat
       type(s_Hmat)          :: Mat
       integer(c_int), value :: nr, nc, nz
       integer(c_int)        :: res
     end function s_HYBGDeviceAlloc
     function d_HYBGDeviceAlloc(Mat,nr,nc,nz) &
          & bind(c,name="d_HYBGDeviceAlloc") result(res)
       use iso_c_binding
       import  d_hmat
       type(d_Hmat)          :: Mat
       integer(c_int), value :: nr, nc, nz
       integer(c_int)        :: res
     end function d_HYBGDeviceAlloc
  end interface HYBGDeviceAlloc
  
  interface HYBGDeviceFree
     function s_HYBGDeviceFree(Mat) &
          & bind(c,name="s_HYBGDeviceFree") result(res)
       use iso_c_binding
       import  s_Hmat
       type(s_Hmat)   :: Mat
       integer(c_int) :: res
     end function s_HYBGDeviceFree
     function d_HYBGDeviceFree(Mat) &
          & bind(c,name="d_HYBGDeviceFree") result(res)
       use iso_c_binding
       import  d_Hmat
       type(d_Hmat)   :: Mat
       integer(c_int) :: res
     end function d_HYBGDeviceFree
  end interface HYBGDeviceFree

  interface HYBGDeviceSetMatType 
     function s_HYBGDeviceSetMatType(Mat,type) &
          & bind(c,name="s_HYBGDeviceSetMatType") result(res)
       use iso_c_binding
       import  s_Hmat
       type(s_Hmat)          :: Mat
       integer(c_int),value  :: type
       integer(c_int)        :: res
     end function s_HYBGDeviceSetMatType
     function d_HYBGDeviceSetMatType(Mat,type) &
          & bind(c,name="d_HYBGDeviceSetMatType") result(res)
       use iso_c_binding
       import  d_Hmat
       type(d_Hmat)          :: Mat
       integer(c_int),value  :: type
       integer(c_int)        :: res
     end function d_HYBGDeviceSetMatType
  end interface HYBGDeviceSetMatType

  interface HYBGDeviceSetMatFillMode
     function s_HYBGDeviceSetMatFillMode(Mat,type) &
          & bind(c,name="s_HYBGDeviceSetMatFillMode") result(res)
       use iso_c_binding
       import  s_Hmat
       type(s_Hmat)          :: Mat
       integer(c_int),value  :: type
       integer(c_int)        :: res
     end function s_HYBGDeviceSetMatFillMode
     function d_HYBGDeviceSetMatFillMode(Mat,type) &
          & bind(c,name="d_HYBGDeviceSetMatFillMode") result(res)
       use iso_c_binding
       import  d_Hmat
       type(d_Hmat)          :: Mat
       integer(c_int),value  :: type
       integer(c_int)        :: res
     end function d_HYBGDeviceSetMatFillMode
  end interface HYBGDeviceSetMatFillMode

  interface HYBGDeviceSetMatDiagType
     function s_HYBGDeviceSetMatDiagType(Mat,type) &
          & bind(c,name="s_HYBGDeviceSetMatDiagType") result(res)
       use iso_c_binding
       import  s_Hmat
       type(s_Hmat)          :: Mat
       integer(c_int),value  :: type
       integer(c_int)        :: res
     end function s_HYBGDeviceSetMatDiagType
     function d_HYBGDeviceSetMatDiagType(Mat,type) &
          & bind(c,name="d_HYBGDeviceSetMatDiagType") result(res)
       use iso_c_binding
       import  d_Hmat
       type(d_Hmat)          :: Mat
       integer(c_int),value  :: type
       integer(c_int)        :: res
     end function d_HYBGDeviceSetMatDiagType
  end interface HYBGDeviceSetMatDiagType
    
  interface HYBGDeviceSetMatIndexBase
     function s_HYBGDeviceSetMatIndexBase(Mat,type) &
         & bind(c,name="s_HYBGDeviceSetMatIndexBase") result(res)
       use iso_c_binding
       import  s_Hmat
       type(s_Hmat)          :: Mat
       integer(c_int),value  :: type
       integer(c_int)        :: res
     end function s_HYBGDeviceSetMatIndexBase
     function d_HYBGDeviceSetMatIndexBase(Mat,type) &
         & bind(c,name="d_HYBGDeviceSetMatIndexBase") result(res)
       use iso_c_binding
       import  d_Hmat
       type(d_Hmat)          :: Mat
       integer(c_int),value  :: type
       integer(c_int)        :: res
     end function d_HYBGDeviceSetMatIndexBase
  end interface HYBGDeviceSetMatIndexBase
  
  interface HYBGDeviceHybsmAnalysis
    function s_HYBGDeviceHybsmAnalysis(Mat) &
         & bind(c,name="s_HYBGDeviceHybsmAnalysis") result(res)
      use iso_c_binding
      import  s_Hmat
      type(s_Hmat)          :: Mat
      integer(c_int)        :: res
    end function s_HYBGDeviceHybsmAnalysis
     function d_HYBGDeviceHybsmAnalysis(Mat) &
         & bind(c,name="d_HYBGDeviceHybsmAnalysis") result(res)
      use iso_c_binding
      import  d_Hmat
      type(d_Hmat)          :: Mat
      integer(c_int)        :: res
    end function d_HYBGDeviceHybsmAnalysis
 end interface HYBGDeviceHybsmAnalysis

 interface spsvHYBGDevice
    function s_spsvHYBGDevice(Mat,alpha,x,beta,y) &
         &  bind(c,name="s_spsvHYBGDevice") result(res)
      use iso_c_binding
      import  s_Hmat
      type(s_Hmat)          :: Mat
      type(c_ptr), value    :: x
      type(c_ptr), value    :: y
      real(c_float), value  :: alpha,beta
      integer(c_int)        :: res
    end function s_spsvHYBGDevice
    function d_spsvHYBGDevice(Mat,alpha,x,beta,y) &
         &  bind(c,name="d_spsvHYBGDevice") result(res)
      use iso_c_binding
      import  d_Hmat
      type(d_Hmat)          :: Mat
      type(c_ptr), value    :: x
      type(c_ptr), value    :: y
      real(c_double), value :: alpha,beta
      integer(c_int)        :: res
    end function d_spsvHYBGDevice
 end interface spsvHYBGDevice

 interface spmvHYBGDevice
    function s_spmvHYBGDevice(Mat,alpha,x,beta,y) &
         &  bind(c,name="s_spmvHYBGDevice") result(res)
      use iso_c_binding
      import  s_Hmat
      type(s_Hmat)          :: Mat
      type(c_ptr), value    :: x
      type(c_ptr), value    :: y
      real(c_float), value  :: alpha,beta
      integer(c_int)        :: res
    end function s_spmvHYBGDevice
    function d_spmvHYBGDevice(Mat,alpha,x,beta,y) &
         &  bind(c,name="d_spmvHYBGDevice") result(res)
      use iso_c_binding
      import  d_Hmat
      type(d_Hmat)          :: Mat
      type(c_ptr), value    :: x
      type(c_ptr), value    :: y
      real(c_double), value :: alpha,beta
      integer(c_int)        :: res
    end function d_spmvHYBGDevice
 end interface spmvHYBGDevice
 
  interface HYBGHost2Device
     function s_HYBGHost2Device(Mat,m,n,nz,irp,ja,val) &
          &  bind(c,name="s_HYBGHost2Device") result(res)
       use iso_c_binding
       import  s_Hmat
       type(s_Hmat)          :: Mat
       integer(c_int), value :: m,n,nz
       integer(c_int)        :: irp(*), ja(*)
       real(c_float)         :: val(*)
       integer(c_int)        :: res
     end function s_HYBGHost2Device
     function d_HYBGHost2Device(Mat,m,n,nz,irp,ja,val) &
          &  bind(c,name="d_HYBGHost2Device") result(res)
       use iso_c_binding
       import  d_Hmat
       type(d_Hmat)          :: Mat
       integer(c_int), value :: m,n,nz
       integer(c_int)        :: irp(*), ja(*)
       real(c_double)        :: val(*)
       integer(c_int)        :: res
     end function d_HYBGHost2Device
  end interface HYBGHost2Device

contains
  
  function initFcusparse() result(res)
    implicit none 
    integer(c_int) :: res
    if (c_associated(cusparse_context)) then 
      res = cusparse_status_success
    else
      res = FcusparseCreate(cusparse_context)
    end if
  end function initFcusparse

  function closeFcusparse() result(res)
    implicit none 
    integer(c_int) :: res
    if (c_associated(cusparse_context)) then 
      res = FcusparseDestroy(cusparse_context)
    else
      res = cusparse_status_success
    end if
  end function closeFcusparse

#endif

end module cusparse_mod
#endif
