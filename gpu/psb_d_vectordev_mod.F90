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
  

module psb_d_vectordev_mod

  use psb_base_vectordev_mod
 
#ifdef HAVE_SPGPU  
  
  interface registerMapped
    function registerMappedDouble(buf,d_p,n,dummy) &
         & result(res) bind(c,name='registerMappedDouble')
      use iso_c_binding
      integer(c_int) :: res
      type(c_ptr), value :: buf
      type(c_ptr) :: d_p
      integer(c_int),value :: n
      real(c_double), value :: dummy
    end function registerMappedDouble
  end interface

  interface writeMultiVecDevice 
    function writeMultiVecDeviceDouble(deviceVec,hostVec) &
         & result(res) bind(c,name='writeMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)             :: res
      type(c_ptr), value         :: deviceVec
      real(c_double)   :: hostVec(*)
    end function writeMultiVecDeviceDouble
    function writeMultiVecDeviceDoubleR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='writeMultiVecDeviceDoubleR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      real(c_double)      :: hostVec(ld,*)
    end function writeMultiVecDeviceDoubleR2
  end interface 

  interface readMultiVecDevice
    function readMultiVecDeviceDouble(deviceVec,hostVec) &
         & result(res) bind(c,name='readMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)           :: res
      type(c_ptr), value       :: deviceVec
      real(c_double) :: hostVec(*)
    end function readMultiVecDeviceDouble
    function readMultiVecDeviceDoubleR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='readMultiVecDeviceDoubleR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      real(c_double)      :: hostVec(ld,*)
    end function readMultiVecDeviceDoubleR2
  end interface 

! New gather functions

  interface 
    function igathMultiVecDeviceDouble(deviceVec, vectorId, first, n, idx, hostVec, indexBase) &
	& result(res) bind(c,name='igathMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int),value:: vectorId
      integer(c_int),value:: first, n
      type(c_ptr),value	  :: idx
      type(c_ptr),value   :: hostVec
      integer(c_int),value:: indexBase
    end function igathMultiVecDeviceDouble
  end interface


  interface 
    function iscatMultiVecDeviceDouble(deviceVec, vectorId, first, n, idx, hostVec, indexBase, beta) &
	& result(res) bind(c,name='iscatMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)         :: res
      type(c_ptr), value     :: deviceVec
      integer(c_int),value   :: vectorId
      integer(c_int),value   :: first, n
      type(c_ptr), value     :: idx
      type(c_ptr), value     :: hostVec
      integer(c_int),value   :: indexBase
      real(c_double),value :: beta
    end function iscatMultiVecDeviceDouble
  end interface

    
  interface nrm2MultiVecDevice
    function nrm2MultiVecDeviceDouble(res,n,deviceVecA) &
         & result(val) bind(c,name='nrm2MultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      real(c_double)         :: res
      type(c_ptr), value    :: deviceVecA
    end function nrm2MultiVecDeviceDouble
  end interface

  interface dotMultiVecDevice
    function dotMultiVecDeviceDouble(res, n,deviceVecA,deviceVecB) &
         & result(val) bind(c,name='dotMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      real(c_double) :: res
      type(c_ptr), value    :: deviceVecA, deviceVecB
    end function dotMultiVecDeviceDouble
  end interface
  
!!$  interface 
!!$    function geinsMultiVecDeviceDouble(n,deviceVecIrl,deviceVecVal,&
!!$         & dupl,indexbase,deviceVecX) &
!!$         & result(res) bind(c,name='geinsMultiVecDeviceDouble')
!!$      use iso_c_binding
!!$      integer(c_int)      :: res
!!$      integer(c_int), value :: n, dupl,indexbase
!!$      type(c_ptr), value  :: deviceVecIrl, deviceVecVal, deviceVecX
!!$    end function geinsMultiVecDeviceDouble
!!$  end interface

  interface axpbyMultiVecDevice
    function axpbyMultiVecDeviceDouble(n,alpha,deviceVecA,beta,deviceVecB) &
         & result(res) bind(c,name='axpbyMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)      :: res
      integer(c_int), value :: n
      real(c_double), value :: alpha, beta
      type(c_ptr), value  :: deviceVecA, deviceVecB
    end function axpbyMultiVecDeviceDouble
  end interface

  interface axyMultiVecDevice
    function axyMultiVecDeviceDouble(n,alpha,deviceVecA,deviceVecB) &
         & result(res) bind(c,name='axyMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)        :: res
      integer(c_int), value :: n
      real(c_double) :: alpha
      type(c_ptr), value       :: deviceVecA, deviceVecB
    end function axyMultiVecDeviceDouble
  end interface

  interface axybzMultiVecDevice
    function axybzMultiVecDeviceDouble(n,alpha,deviceVecA,deviceVecB,beta,deviceVecZ) &
         & result(res) bind(c,name='axybzMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)              :: res
      integer(c_int), value       :: n
      real(c_double), value     :: alpha, beta
      type(c_ptr), value          :: deviceVecA, deviceVecB,deviceVecZ
    end function axybzMultiVecDeviceDouble
  end interface

  interface inner_register
    module procedure inner_registerDouble
  end interface
  
  interface inner_unregister
    module procedure inner_unregisterDouble
  end interface

contains


  function inner_registerDouble(buffer,dval) result(res)
    real(c_double), allocatable, target :: buffer(:)
    type(c_ptr)            :: dval
    integer(c_int)         :: res
    real(c_double)         :: dummy
    res = registerMapped(c_loc(buffer),dval,size(buffer), dummy)        
  end function inner_registerDouble

  subroutine inner_unregisterDouble(buffer)
    real(c_double), allocatable, target :: buffer(:)

    call  unregisterMapped(c_loc(buffer))
  end subroutine inner_unregisterDouble

#endif  

end module psb_d_vectordev_mod
