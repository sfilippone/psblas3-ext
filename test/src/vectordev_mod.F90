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
  

module vectordev_mod
  use iso_c_binding 
  use core_mod

  type, bind(c) :: multivec_dev_parms
    integer(c_int) :: count
    integer(c_int) :: element_type
    integer(c_int) :: pitch
    integer(c_int) :: size
  end type multivec_dev_parms
 
#ifdef HAVE_SPGPU  


  interface 
    function FallocMultiVecDevice(deviceVec,count,Size,elementType) &
         & result(res) bind(c,name='FallocMultiVecDevice')
      use iso_c_binding
      integer(c_int)        :: res
      integer(c_int), value :: count,Size,elementType
      type(c_ptr)           :: deviceVec
    end function FallocMultiVecDevice
  end interface

  interface 
    subroutine  freeMultiVecDevice(deviceVec) &
         & bind(c,name='freeMultiVecDevice')
      use iso_c_binding
      type(c_ptr), value  :: deviceVec
    end subroutine freeMultiVecDevice
  end interface

  interface writeMultiVecDevice 
    function writeMultiVecDeviceFloat(deviceVec,hostVec) &
         & result(res) bind(c,name='writeMultiVecDeviceFloat')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      real(c_float)       :: hostVec(*)
    end function writeMultiVecDeviceFloat
    function writeMultiVecDeviceDouble(deviceVec,hostVec) &
         & result(res) bind(c,name='writeMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      real(c_double)      :: hostVec(*)
    end function writeMultiVecDeviceDouble
    function writeMultiVecDeviceFloatComplex(deviceVec,hostVec) &
         & result(res) bind(c,name='writeMultiVecDeviceFloatComplex')
      use iso_c_binding
      integer(c_int)             :: res
      type(c_ptr), value         :: deviceVec
      complex(c_float_complex)   :: hostVec(*)
    end function writeMultiVecDeviceFloatComplex
    function writeMultiVecDeviceDoubleComplex(deviceVec,hostVec) &
         & result(res) bind(c,name='writeMultiVecDeviceDoubleComplex')
      use iso_c_binding
      integer(c_int)             :: res
      type(c_ptr), value         :: deviceVec
      complex(c_double_complex)  :: hostVec(*)
    end function writeMultiVecDeviceDoubleComplex
    function writeMultiVecDeviceFloatR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='writeMultiVecDeviceFloatR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      real(c_float)       :: hostVec(ld,*)
    end function writeMultiVecDeviceFloatR2
    function writeMultiVecDeviceDoubleR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='writeMultiVecDeviceDoubleR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      real(c_double)      :: hostVec(ld,*)
    end function writeMultiVecDeviceDoubleR2
    function writeMultiVecDeviceFloatComplexR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='writeMultiVecDeviceFloatComplexR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      complex(c_float_complex)      :: hostVec(ld,*)
    end function writeMultiVecDeviceFloatComplexR2
    function writeMultiVecDeviceDoubleComplexR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='writeMultiVecDeviceDoubleComplexR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      complex(c_double_complex)      :: hostVec(ld,*)
    end function writeMultiVecDeviceDoubleComplexR2
  end interface writeMultiVecDevice

  interface readMultiVecDevice
    function readMultiVecDeviceFloat(deviceVec,hostVec) &
         & result(res) bind(c,name='readMultiVecDeviceFloat')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      real(c_float)      :: hostVec(*)
    end function readMultiVecDeviceFloat
    function readMultiVecDeviceDouble(deviceVec,hostVec) &
         & result(res) bind(c,name='readMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      real(c_double)      :: hostVec(*)
    end function readMultiVecDeviceDouble
    function readMultiVecDeviceFloatComplex(deviceVec,hostVec) &
         & result(res) bind(c,name='readMultiVecDeviceFloatComplex')
      use iso_c_binding
      integer(c_int)           :: res
      type(c_ptr), value       :: deviceVec
      complex(c_float_complex) :: hostVec(*)
    end function readMultiVecDeviceFloatComplex
    function readMultiVecDeviceDoubleComplex(deviceVec,hostVec) &
         & result(res) bind(c,name='readMultiVecDeviceDoubleComplex')
      use iso_c_binding
      integer(c_int)           :: res
      type(c_ptr), value       :: deviceVec
      complex(c_double_complex) :: hostVec(*)
    end function readMultiVecDeviceDoubleComplex
    function readMultiVecDeviceFloatR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='readMultiVecDeviceFloatR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      real(c_float)       :: hostVec(ld,*)
    end function readMultiVecDeviceFloatR2
    function readMultiVecDeviceDoubleR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='readMultiVecDeviceDoubleR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      real(c_double)      :: hostVec(ld,*)
    end function readMultiVecDeviceDoubleR2
    function readMultiVecDeviceFloatComplexR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='readMultiVecDeviceFloatComplexR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      complex(c_float_complex)      :: hostVec(ld,*)
    end function readMultiVecDeviceFloatComplexR2
    function readMultiVecDeviceDoubleComplexR2(deviceVec,hostVec,ld) &
         & result(res) bind(c,name='readMultiVecDeviceDoubleComplexR2')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int), value :: ld
      complex(c_double_complex)      :: hostVec(ld,*)
    end function readMultiVecDeviceDoubleComplexR2
  end interface readMultiVecDevice

! New gather functions
  interface igathMultiVecDevice
    function igathMultiVecDeviceFloat(deviceVec, vectorId, n, idx, hostVec, firstIndex) &
	& result(res) bind(c,name='igathMultiVecDeviceFloat')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int),value:: vectorId
      integer(c_int),value:: n
      integer(c_int)	  :: idx(*)
      real(c_float)       :: hostVec(*)
      integer(c_int),value:: firstIndex
    end function igathMultiVecDeviceFloat
    function igathMultiVecDeviceDouble(deviceVec, vectorId, n, idx, hostVec, firstIndex) &
	& result(res) bind(c,name='igathMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int),value:: vectorId
      integer(c_int),value:: n
      integer(c_int)	  :: idx(*)
      real(c_double)      :: hostVec(*)
      integer(c_int),value:: firstIndex
    end function igathMultiVecDeviceDouble
    function igathMultiVecDeviceFloatComplex(deviceVec, vectorId, n, idx, hostVec, firstIndex) &
	& result(res) bind(c,name='igathMultiVecDeviceFloatComplex')
      use iso_c_binding
      integer(c_int)           :: res
      type(c_ptr), value       :: deviceVec
      integer(c_int),value     :: vectorId
      integer(c_int),value     :: n
      integer(c_int)	       :: idx(*)
      complex(c_float_complex) :: hostVec(*)
      integer(c_int),value     :: firstIndex
    end function igathMultiVecDeviceFloatComplex
    function igathMultiVecDeviceDoubleComplex(deviceVec, vectorId, n, idx, hostVec, firstIndex) &
	& result(res) bind(c,name='igathMultiVecDeviceDoubleComplex')
      use iso_c_binding
      integer(c_int)           :: res
      type(c_ptr), value       :: deviceVec
      integer(c_int),value     :: vectorId
      integer(c_int),value     :: n
      integer(c_int)	       :: idx(*)
      complex(c_double_complex) :: hostVec(*)
      integer(c_int),value     :: firstIndex
    end function igathMultiVecDeviceDoubleComplex
  end interface igathMultiVecDevice

  interface iscatMultiVecDevice
    function iscatMultiVecDeviceFloat(deviceVec, vectorId, n, idx, hostVec, firstIndex, beta) &
	& result(res) bind(c,name='iscatMultiVecDeviceFloat')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int),value:: vectorId
      integer(c_int),value:: n
      integer(c_int)	  :: idx(*)
      real(c_float)       :: hostVec(*)
      integer(c_int),value:: firstIndex
      real(c_float),value :: beta
    end function iscatMultiVecDeviceFloat
    function iscatMultiVecDeviceDouble(deviceVec, vectorId, n, idx, hostVec, firstIndex, beta) &
	& result(res) bind(c,name='iscatMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int),value:: vectorId
      integer(c_int),value:: n
      integer(c_int)	  :: idx(*)
      real(c_double)      :: hostVec(*)
      integer(c_int),value:: firstIndex
      real(c_double),value:: beta
    end function iscatMultiVecDeviceDouble
    function iscatMultiVecDeviceFloatComplex(deviceVec, vectorId, n, idx, hostVec, firstIndex, beta) &
	& result(res) bind(c,name='iscatMultiVecDeviceFloatComplex')
      use iso_c_binding
      integer(c_int)      :: res
      type(c_ptr), value  :: deviceVec
      integer(c_int),value:: vectorId
      integer(c_int),value:: n
      integer(c_int)	  :: idx(*)
      complex(c_float_complex)      :: hostVec(*)
      integer(c_int),value:: firstIndex
      complex(c_float_complex),value:: beta
    end function iscatMultiVecDeviceFloatComplex
    function iscatMultiVecDeviceDoubleComplex(deviceVec, vectorId, n, idx, hostVec, firstIndex, beta) &
	& result(res) bind(c,name='iscatMultiVecDeviceDoubleComplex')
      use iso_c_binding
      integer(c_int)                  :: res
      type(c_ptr), value              :: deviceVec
      integer(c_int),value            :: vectorId
      integer(c_int),value            :: n
      integer(c_int)	              :: idx(*)
      complex(c_double_complex)       :: hostVec(*)
      integer(c_int),value            :: firstIndex
      complex(c_double_complex),value :: beta
    end function iscatMultiVecDeviceDoubleComplex
  end interface iscatMultiVecDevice

    
  interface nrm2MultiVecDevice
    function nrm2MultiVecDeviceFloat(res,n,deviceVecA) &
         & result(val) bind(c,name='nrm2MultiVecDeviceFloat')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      real(c_float)         :: res
      type(c_ptr), value    :: deviceVecA
    end function nrm2MultiVecDeviceFloat
    function nrm2MultiVecDeviceDouble(res,n,deviceVecA) &
         & result(val) bind(c,name='nrm2MultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      real(c_double)        :: res
      type(c_ptr), value    :: deviceVecA
    end function nrm2MultiVecDeviceDouble
  end interface nrm2MultiVecDevice
  
  interface nrm2MultiVecDeviceComplex
    function nrm2MultiVecDeviceFloatComplex(res,n,deviceVecA) &
         & result(val) bind(c,name='nrm2MultiVecDeviceFloatComplex')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      real(c_float)         :: res
      type(c_ptr), value    :: deviceVecA
    end function nrm2MultiVecDeviceFloatComplex
    function nrm2MultiVecDeviceDoubleComplex(res,n,deviceVecA) &
         & result(val) bind(c,name='nrm2MultiVecDeviceDoubleComplex')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      real(c_double)        :: res
      type(c_ptr), value    :: deviceVecA
    end function nrm2MultiVecDeviceDoubleComplex
  end interface

  interface dotMultiVecDevice
    function dotMultiVecDeviceFloat(res, n,deviceVecA,deviceVecB) &
         & result(val) bind(c,name='dotMultiVecDeviceFloat')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      real(c_float)         :: res
      type(c_ptr), value    :: deviceVecA, deviceVecB
    end function dotMultiVecDeviceFloat
    function dotMultiVecDeviceDouble(res,n,deviceVecA,deviceVecB) &
         & result(val) bind(c,name='dotMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      real(c_double)        :: res
      type(c_ptr), value    :: deviceVecA, deviceVecB
    end function dotMultiVecDeviceDouble
    function dotMultiVecDeviceFloatComplex(res, n,deviceVecA,deviceVecB) &
         & result(val) bind(c,name='dotMultiVecDeviceFloatComplex')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      complex(c_float_complex) :: res
      type(c_ptr), value    :: deviceVecA, deviceVecB
    end function dotMultiVecDeviceFloatComplex
    function dotMultiVecDeviceDoubleComplex(res,n,deviceVecA,deviceVecB) &
         & result(val) bind(c,name='dotMultiVecDeviceDoubleComplex')
      use iso_c_binding
      integer(c_int)        :: val
      integer(c_int), value :: n
      complex(c_double_complex) :: res
      type(c_ptr), value    :: deviceVecA, deviceVecB
    end function dotMultiVecDeviceDoubleComplex
  end interface
  
  interface axpbyMultiVecDevice
    function axpbyMultiVecDeviceFloat(alpha,deviceVecA,beta,deviceVecB) &
         & result(res) bind(c,name='axpbyMultiVecDeviceFloat')
      use iso_c_binding
      integer(c_int)       :: res
      real(c_float), value :: alpha, beta
      type(c_ptr), value   :: deviceVecA, deviceVecB
    end function axpbyMultiVecDeviceFloat
    function axpbyMultiVecDeviceDouble(alpha,deviceVecA,beta,deviceVecB) &
         & result(res) bind(c,name='axpbyMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)        :: res
      real(c_double), value :: alpha,beta
      type(c_ptr), value    :: deviceVecA, deviceVecB
    end function axpbyMultiVecDeviceDouble
    function axpbyMultiVecDeviceFloatComplex(alpha,deviceVecA,beta,deviceVecB) &
         & result(res) bind(c,name='axpbyMultiVecDeviceFloatComplex')
      use iso_c_binding
      integer(c_int)      :: res
      complex(c_float_complex), value :: alpha, beta
      type(c_ptr), value  :: deviceVecA, deviceVecB
    end function axpbyMultiVecDeviceFloatComplex
    function axpbyMultiVecDeviceDoubleComplex(alpha,deviceVecA,beta,deviceVecB) &
         & result(res) bind(c,name='axpbyMultiVecDeviceDoubleComplex')
      use iso_c_binding
      integer(c_int)      :: res
      complex(c_double_complex), value :: alpha,beta
      type(c_ptr), value  :: deviceVecA, deviceVecB
    end function axpbyMultiVecDeviceDoubleComplex
  end interface axpbyMultiVecDevice

  interface axyMultiVecDevice
    function axyMultiVecDeviceFloat(alpha,deviceVecA,deviceVecB) &
         & result(res) bind(c,name='axyMultiVecDeviceFloat')
      use iso_c_binding
      integer(c_int)      :: res
      real(c_float)       :: alpha
      type(c_ptr), value  :: deviceVecA, deviceVecB
    end function axyMultiVecDeviceFloat
    function axyMultiVecDeviceDouble(alpha,deviceVecA,deviceVecB) &
         & result(res) bind(c,name='axyMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)      :: res
      real(c_double)      :: alpha
      type(c_ptr), value  :: deviceVecA, deviceVecB
    end function axyMultiVecDeviceDouble
    function axyMultiVecDeviceFloatComplex(alpha,deviceVecA,deviceVecB) &
         & result(res) bind(c,name='axyMultiVecDeviceFloatComplex')
      use iso_c_binding
      integer(c_int)           :: res
      complex(c_float_complex) :: alpha
      type(c_ptr), value       :: deviceVecA, deviceVecB
    end function axyMultiVecDeviceFloatComplex
    function axyMultiVecDeviceDoubleComplex(alpha,deviceVecA,deviceVecB) &
         & result(res) bind(c,name='axyMultiVecDeviceDoubleComplex')
      use iso_c_binding
      integer(c_int)            :: res
      complex(c_double_complex) :: alpha
      type(c_ptr), value        :: deviceVecA, deviceVecB
    end function axyMultiVecDeviceDoubleComplex
  end interface axyMultiVecDevice

  interface axybzMultiVecDevice
    function axybzMultiVecDeviceFloat(alpha,deviceVecA,deviceVecB,beta,deviceVecZ) &
         & result(res) bind(c,name='axybzMultiVecDeviceFloat')
      use iso_c_binding
      integer(c_int)      :: res
      real(c_float),value :: alpha, beta
      type(c_ptr), value  :: deviceVecA, deviceVecB,deviceVecZ
    end function axybzMultiVecDeviceFloat
    function axybzMultiVecDeviceDouble(alpha,deviceVecA,deviceVecB,beta,deviceVecZ) &
         & result(res) bind(c,name='axybzMultiVecDeviceDouble')
      use iso_c_binding
      integer(c_int)      :: res
      real(c_double),value:: alpha, beta
      type(c_ptr), value  :: deviceVecA, deviceVecB, deviceVecZ
    end function axybzMultiVecDeviceDouble
    function axybzMultiVecDeviceFloatComplex(alpha,deviceVecA,deviceVecB,beta,deviceVecZ) &
         & result(res) bind(c,name='axybzMultiVecDeviceFloatComplex')
      use iso_c_binding
      integer(c_int)                  :: res
      complex(c_float_complex), value :: alpha, beta
      type(c_ptr), value              :: deviceVecA, deviceVecB,deviceVecZ
    end function axybzMultiVecDeviceFloatComplex
    function axybzMultiVecDeviceDoubleComplex(alpha,deviceVecA,deviceVecB,beta,deviceVecZ) &
         & result(res) bind(c,name='axybzMultiVecDeviceDoubleComplex')
      use iso_c_binding
      integer(c_int)                   :: res
      complex(c_double_complex), value :: alpha, beta
      type(c_ptr), value               :: deviceVecA, deviceVecB, deviceVecZ
    end function axybzMultiVecDeviceDoubleComplex
  end interface axybzMultiVecDevice
  

  interface 
    function  getMultiVecDeviceSize(deviceVec) &
         & bind(c,name='getMultiVecDeviceSize') result(res)
      use iso_c_binding
      type(c_ptr), value  :: deviceVec
      integer(c_int)      :: res
    end function getMultiVecDeviceSize
  end interface

  interface 
    function  getMultiVecDeviceCount(deviceVec) &
         & bind(c,name='getMultiVecDeviceCount') result(res)
      use iso_c_binding
      type(c_ptr), value  :: deviceVec
      integer(c_int)      :: res
    end function getMultiVecDeviceCount
  end interface


#endif  


end module vectordev_mod
