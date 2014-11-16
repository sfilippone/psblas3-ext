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
  

module psb_i_vectordev_mod

  use psb_base_vectordev_mod
 
#ifdef HAVE_SPGPU  
  
!!$  interface registerMapped
!!$    function registerMappedInt(buf,d_p,n,dummy) &
!!$         & result(res) bind(c,name='registerMappedInt')
!!$      use iso_c_binding
!!$      integer(c_int) :: res
!!$      type(c_ptr), value :: buf
!!$      type(c_ptr) :: d_p
!!$      integer(c_int),value :: n
!!$      integer(c_int), value :: dummy
!!$    end function registerMappedInt
!!$  end interface
!!$
  interface writeMultiVecDevice 
    function writeMultiVecDeviceInt(deviceVec,hostVec) &
         & result(res) bind(c,name='writeMultiVecDeviceInt')
      use iso_c_binding
      integer(c_int)             :: res
      type(c_ptr), value         :: deviceVec
      integer(c_int)   :: hostVec(*)
    end function writeMultiVecDeviceInt
    function writeMultiVecDeviceIntR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='writeMultiVecDeviceIntR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      integer(c_int)      :: hostVec(ld,*)
    end function writeMultiVecDeviceIntR2
  end interface 

  interface readMultiVecDevice
    function readMultiVecDeviceInt(deviceVec,hostVec) &
         & result(res) bind(c,name='readMultiVecDeviceInt')
      use iso_c_binding
      integer(c_int)           :: res
      type(c_ptr), value       :: deviceVec
      integer(c_int) :: hostVec(*)
    end function readMultiVecDeviceInt
    function readMultiVecDeviceIntR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='readMultiVecDeviceIntR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      integer(c_int)      :: hostVec(ld,*)
    end function readMultiVecDeviceIntR2
  end interface 

! New gather functions

  interface 
    function igathMultiVecDeviceInt(deviceVec, vectorId, first, n, idx, hostVec, indexBase) &
	& result(res) bind(c,name='igathMultiVecDeviceInt')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int),value:: vectorId
      integer(c_int),value:: first, n
      type(c_ptr),value	  :: idx
      type(c_ptr),value   :: hostVec
      integer(c_int),value:: indexBase
    end function igathMultiVecDeviceInt
  end interface


  interface 
    function iscatMultiVecDeviceInt(deviceVec, vectorId, first, n, idx, hostVec, indexBase, beta) &
	& result(res) bind(c,name='iscatMultiVecDeviceInt')
      use iso_c_binding
      integer(c_int)         :: res
      type(c_ptr), value     :: deviceVec
      integer(c_int),value   :: vectorId
      integer(c_int),value   :: first, n
      type(c_ptr), value     :: idx
      type(c_ptr), value     :: hostVec
      integer(c_int),value   :: indexBase
      integer(c_int),value :: beta
    end function iscatMultiVecDeviceInt
  end interface

    
  interface nrm2MultiVecDevice
    function nrm2MultiVecDeviceInt(res,n,deviceVecA) &
         & result(val) bind(c,name='nrm2MultiVecDeviceInt')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      integer(c_int)         :: res
      type(c_ptr), value    :: deviceVecA
    end function nrm2MultiVecDeviceInt
  end interface

  interface dotMultiVecDevice
    function dotMultiVecDeviceInt(res, n,deviceVecA,deviceVecB) &
         & result(val) bind(c,name='dotMultiVecDeviceInt')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      integer(c_int) :: res
      type(c_ptr), value    :: deviceVecA, deviceVecB
    end function dotMultiVecDeviceInt
  end interface
  
  interface axpbyMultiVecDevice
    function axpbyMultiVecDeviceInt(n,alpha,deviceVecA,beta,deviceVecB) &
         & result(res) bind(c,name='axpbyMultiVecDeviceInt')
      use iso_c_binding
      integer(c_int)      :: res
      integer(c_int), value :: n
      integer(c_int), value :: alpha, beta
      type(c_ptr), value  :: deviceVecA, deviceVecB
    end function axpbyMultiVecDeviceInt
  end interface

  interface axyMultiVecDevice
    function axyMultiVecDeviceInt(n,alpha,deviceVecA,deviceVecB) &
         & result(res) bind(c,name='axyMultiVecDeviceInt')
      use iso_c_binding
      integer(c_int)        :: res
      integer(c_int), value :: n
      integer(c_int) :: alpha
      type(c_ptr), value       :: deviceVecA, deviceVecB
    end function axyMultiVecDeviceInt
  end interface

  interface axybzMultiVecDevice
    function axybzMultiVecDeviceInt(n,alpha,deviceVecA,deviceVecB,beta,deviceVecZ) &
         & result(res) bind(c,name='axybzMultiVecDeviceInt')
      use iso_c_binding
      integer(c_int)              :: res
      integer(c_int), value       :: n
      integer(c_int), value     :: alpha, beta
      type(c_ptr), value          :: deviceVecA, deviceVecB,deviceVecZ
    end function axybzMultiVecDeviceInt
  end interface

  interface inner_register
    module procedure inner_registerInt
  end interface
  
  interface inner_unregister
    module procedure inner_unregisterInt
  end interface

contains


  function inner_registerInt(buffer,dval) result(res)
    integer(c_int), allocatable, target :: buffer(:)
    type(c_ptr)            :: dval
    integer(c_int)         :: res
    integer(c_int)         :: dummy
    res = registerMapped(c_loc(buffer),dval,size(buffer), dummy)        
  end function inner_registerInt

  subroutine inner_unregisterInt(buffer)
    integer(c_int), allocatable, target :: buffer(:)

    call  unregisterMapped(c_loc(buffer))
  end subroutine inner_unregisterInt

#endif  

end module psb_i_vectordev_mod
