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
  

subroutine psb_s_elg_csput_a(nz,ia,ja,val,a,imin,imax,jmin,jmax,info,gtl) 

  use psb_base_mod
  use iso_c_binding
#ifdef HAVE_SPGPU
  use elldev_mod
  use psb_s_elg_mat_mod, psb_protect_name => psb_s_elg_csput_a
#else 
  use psb_s_elg_mat_mod
#endif
  implicit none 

  class(psb_s_elg_sparse_mat), intent(inout) :: a
  real(psb_spk_), intent(in)               :: val(:)
  integer(psb_ipk_), intent(in)             :: nz, ia(:), ja(:), imin,imax,jmin,jmax
  integer(psb_ipk_), intent(out)            :: info
  integer(psb_ipk_), intent(in), optional   :: gtl(:)


  Integer(Psb_ipk_)  :: err_act
  character(len=20)  :: name='s_elg_csput_a'
  logical, parameter :: debug=.false.
  integer(psb_ipk_)  :: nza, i,j,k, nzl, isza, int_err(5), debug_level, debug_unit
  real(psb_dpk_)     :: t1,t2,t3
  type(c_ptr)        :: devIdxUpd

  call psb_erractionsave(err_act)
  info = psb_success_
  debug_unit  = psb_get_debug_unit()
  debug_level = psb_get_debug_level()

!!$  write(0,*) 'In ELG_csput_a'
  if (nz <= 0) then 
    info = psb_err_iarg_neg_
    int_err(1)=1
    call psb_errpush(info,name,i_err=int_err)
    goto 9999
  end if
  if (size(ia) < nz) then 
    info = psb_err_input_asize_invalid_i_
    int_err(1)=2
    call psb_errpush(info,name,i_err=int_err)
    goto 9999
  end if

  if (size(ja) < nz) then 
    info = psb_err_input_asize_invalid_i_
    int_err(1)=3
    call psb_errpush(info,name,i_err=int_err)
    goto 9999
  end if
  if (size(val) < nz) then 
    info = psb_err_input_asize_invalid_i_
    int_err(1)=4
    call psb_errpush(info,name,i_err=int_err)
    goto 9999
  end if

  if (nz == 0) return


  if (a%is_bld()) then 
    ! Build phase should only ever be in COO
    info = psb_err_invalid_mat_state_

  else  if (a%is_upd()) then 
#if 0
    if (present(gtl)) then 
#endif
      if (a%is_dev()) call a%sync()
      call a%psb_s_ell_sparse_mat%csput(nz,ia,ja,val,&
           &  imin,imax,jmin,jmax,info,gtl) 
      if (info /= psb_success_) then
        call psb_errpush(info,name)
        goto 9999
      end if
      call a%set_host()
#if 0 
    else
      t1=psb_wtime()
      if (a%is_host()) call a%sync()
      t2=psb_wtime()
      !!info = outerCsputEllDevicePinned(a%deviceMat,nz,ia,ja,val)      
      info = csputEllDevice(a%deviceMat,nz,ia,ja,val)
      t3=psb_wtime()
      if (.false.) write(0,*)name,' ',t2-t1,t3-t2
      if (info /= 0) then 
        info = psb_err_internal_error_
      end if
      call a%set_dev()

    end if
#endif
  else 
    ! State is wrong.
    info = psb_err_invalid_mat_state_
  end if
  if (info /= psb_success_) then
    call psb_errpush(info,name)
    goto 9999
  end if

  call psb_erractionrestore(err_act)
  return

9999 continue
  call psb_erractionrestore(err_act)

  if (err_act == psb_act_abort_) then
    call psb_error()
    return
  end if
  return

#if 0
contains
  function  outerCsputEllDevicePinned(deviceMat,nz,ia,ja,val) result(info)
    use iso_c_binding 
    use elldev_mod
    use psb_vectordev_mod
    use psb_s_elg_mat_mod, psb_protect_name => psb_s_elg_csput_a
    type(c_ptr)                             :: deviceMat
    real(psb_spk_), intent(in), target       :: val(:)
    integer(psb_ipk_), intent(in), target    ::  ia(:), ja(:)
    integer(psb_ipk_), intent(in)    ::  nz
    integer(psb_ipk_)            :: info
    type(c_ptr)   :: cval,cia,cja
    integer(c_int) :: idummy
    real(c_double) :: ddummy
    
!!$    info = registerMapped(c_loc(val),cval,size(val), ddummy)        
!!$    info = registerMapped(c_loc(ia),cia,size(ia), idummy)        
!!$    info = registerMapped(c_loc(ja),cja,size(ja), idummy)        
    cia  = c_loc(ia)
    cja  = c_loc(ja)
    cval = c_loc(val)
    info = csputEllDeviceDoublePinned(deviceMat,nz,cia,cja,cval)
!!$    call  unregisterMapped(c_loc(val))
!!$    call  unregisterMapped(c_loc(ia))
!!$    call  unregisterMapped(c_loc(ja))
  end function outerCsputEllDevicePinned
#endif
end subroutine psb_s_elg_csput_a



subroutine psb_s_elg_csput_v(nz,ia,ja,val,a,imin,imax,jmin,jmax,info,gtl) 

  use psb_base_mod
  use iso_c_binding
#ifdef HAVE_SPGPU
  use elldev_mod
  use psb_s_elg_mat_mod, psb_protect_name => psb_s_elg_csput_v
  use psb_s_gpu_vect_mod
#else 
  use psb_s_elg_mat_mod
#endif
  implicit none 

  class(psb_s_elg_sparse_mat), intent(inout) :: a
  class(psb_s_base_vect_type), intent(inout) :: val
  class(psb_i_base_vect_type), intent(inout) :: ia, ja
  integer(psb_ipk_), intent(in)             :: nz, imin,imax,jmin,jmax
  integer(psb_ipk_), intent(out)            :: info
  integer(psb_ipk_), intent(in), optional   :: gtl(:)


  Integer(Psb_ipk_)  :: err_act
  character(len=20)  :: name='s_elg_csput_v'
  logical, parameter :: debug=.false.
  integer(psb_ipk_)  :: nza, i,j,k, nzl, isza, int_err(5), debug_level, debug_unit, nrw
  logical            :: gpu_invoked       
  real(psb_dpk_)     :: t1,t2,t3
  type(c_ptr)        :: devIdxUpd
  integer(psb_ipk_), allocatable :: idxs(:)
  logical, parameter :: debug_idxs=.false., debug_vals=.false.


  call psb_erractionsave(err_act)
  info = psb_success_
  debug_unit  = psb_get_debug_unit()
  debug_level = psb_get_debug_level()

!  write(0,*) 'In ELG_csput_v'
  if (nz <= 0) then 
    info = psb_err_iarg_neg_
    int_err(1)=1
    call psb_errpush(info,name,i_err=int_err)
    goto 9999
  end if
  if (ia%get_nrows() < nz) then 
    info = psb_err_input_asize_invalid_i_
    int_err(1)=2
    call psb_errpush(info,name,i_err=int_err)
    goto 9999
  end if

  if (ja%get_nrows() < nz) then 
    info = psb_err_input_asize_invalid_i_
    int_err(1)=3
    call psb_errpush(info,name,i_err=int_err)
    goto 9999
  end if
  if (val%get_nrows() < nz) then 
    info = psb_err_input_asize_invalid_i_
    int_err(1)=4
    call psb_errpush(info,name,i_err=int_err)
    goto 9999
  end if

  if (nz == 0) return


  if (a%is_bld()) then 
    ! Build phase should only ever be in COO
    info = psb_err_invalid_mat_state_

  else  if (a%is_upd()) then 

    if (present(gtl)) then 
      if (a%is_dev()) call a%sync()
      call a%psb_s_ell_sparse_mat%csput(nz,ia,ja,val,&
           &  imin,imax,jmin,jmax,info,gtl) 
      if (info /= psb_success_) then
        call psb_errpush(info,name)
        goto 9999
      end if
      call a%set_host()
    else
      t1=psb_wtime()
      gpu_invoked = .false. 
#if 0 
      select type (ia)
      class is (psb_i_vect_gpu) 
        select type (ja)
        class is (psb_i_vect_gpu) 
          select type (val)
          class is (psb_s_vect_gpu) 
            !!$ write(0,*)'calling dev_csput',a%updcnt
           if (allocated(a%updidx%v)) then 
              if ((a%updcnt + nz) > a%updidx%get_nrows()) then 
                write(0,*) 'Reallocating in CSPUT'
                idxs = a%updidx%get_vect()
                nrw = size(idxs)
                call psb_realloc(nrw+4*nz,idxs,info,pad=-ione)
                if (info /= 0) then 
                  info = psb_err_internal_error_
                  goto 9999
                end if      
                call a%updidx%bld(idxs)
              end if
              devIdxUpd=a%updidx%deviceVect
            else
              devIdxUpd=c_null_ptr
            end if
            if (debug_idxs) then 
!!$              idxs= a%updidx%get_vect()
!!$              write(0,*) 'Before dev_csput',idxs(:)
              
            end if
!!$            devIdxUpd=c_null_ptr
              
            if (a%is_host()) call a%sync()
            if (val%is_host()) call val%sync()
            if (ia%is_host()) call ia%sync()
            if (ja%is_host()) call ja%sync()
!!$            info=dev_CsputEllDeviceDouble(a%deviceMat,nz,&
!!$                 & ia%deviceVect,ja%deviceVect,val%deviceVect,devIdxUpd,a%updcnt)
            info = csputEllDevice(a%deviceMat,nz,ia,ja,val)

!!$            call a%updidx%set_dev()
!!$            if (debug_idxs) then 
!!$              if (allocated(a%updidx%v)) then 
!!$                idxs = a%updidx%get_vect()
!!$                write(0,*) 'After dev_csput',idxs(a%updcnt+1:a%updcnt+nz)
!!$              end if
!!$                call ia%sync()
!!$                call ja%sync()
!!$                write(0,*) '     IA        ',ia%v(1:nz)
!!$                write(0,*) '     JA        ',ja%v(1:nz)
!!$              
!!$            end if
!!$            if (debug_vals) then 
!!$              call val%sync()
!!$              write(0,*) '     VALS        ',val%v(1:nz)
!!$            end if
!!$            a%updcnt = a%updcnt+nz
            call a%set_dev()
!!$            write(0,*)'After dev_csput',a%updcnt

            gpu_invoked=.true.
!!$            write(0,*)'done dev_csput'
          end select
        end select
      end select
#endif      
      if (.not.gpu_invoked) then 
!!$        write(0,*)'Not gpu_invoked '
        call a%psb_s_ell_sparse_mat%csput(nz,ia,ja,val,&
             &  imin,imax,jmin,jmax,info,gtl) 
        call a%set_host()
      end if
      
      if (info /= 0) then 
        info = psb_err_internal_error_
      end if

    end if
  else 
    ! State is wrong.
    info = psb_err_invalid_mat_state_
  end if
  if (info /= psb_success_) then
    call psb_errpush(info,name)
    goto 9999
  end if

  call psb_erractionrestore(err_act)
  return

9999 continue
  call psb_erractionrestore(err_act)

  if (err_act == psb_act_abort_) then
    call psb_error()
    return
  end if
  return


end subroutine psb_s_elg_csput_v
