!                Parallel Sparse BLAS   GPU plugin 
!      (C) Copyright 2013
!  
!                         Salvatore Filippone
!                         Alessandro Fanfarillo
!   
!    Redistribution and use in source and binary forms, with or without
!    modification, are permitted provided that the following conditions
!    are met:
!      1. Redistributions of source code must retain the above copyright
!         notice, this list of conditions and the following disclaimer.
!      2. Redistributions in binary form must reproduce the above copyright
!         notice, this list of conditions, and the following disclaimer in the
!         documentation and/or other materials provided with the distribution.
!      3. The name of the PSBLAS group or the names of its contributors may
!         not be used to endorse or promote products derived from this
!         software without specific written permission.
!   
!    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
!    ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
!    TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
!    PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE PSBLAS GROUP OR ITS CONTRIBUTORS
!    BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
!    CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
!    SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
!    INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
!    CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
!    ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
!    POSSIBILITY OF SUCH DAMAGE.
!   
  

subroutine psb_c_ell_csgetptn(imin,imax,a,nz,ia,ja,info,&
     & jmin,jmax,iren,append,nzin,rscale,cscale)
  use psb_base_mod
  use psb_c_ell_mat_mod, psb_protect_name => psb_c_ell_csgetptn
  implicit none

  class(psb_c_ell_sparse_mat), intent(in)       :: a
  integer(psb_ipk_), intent(in)                 :: imin,imax
  integer(psb_ipk_), intent(out)                :: nz
  integer(psb_ipk_), allocatable, intent(inout) :: ia(:), ja(:)
  integer(psb_ipk_),intent(out)                 :: info
  logical, intent(in), optional                 :: append
  integer(psb_ipk_), intent(in), optional       :: iren(:)
  integer(psb_ipk_), intent(in), optional       :: jmin,jmax, nzin
  logical, intent(in), optional                 :: rscale,cscale

  logical :: append_, rscale_, cscale_ 
  integer(psb_ipk_) :: nzin_, jmin_, jmax_, err_act, i
  character(len=20)  :: name='ell_getptn'
  logical, parameter :: debug=.false.

  call psb_erractionsave(err_act)
  info = psb_success_

  if (present(jmin)) then
    jmin_ = jmin
  else
    jmin_ = 1
  endif
  if (present(jmax)) then
    jmax_ = jmax
  else
    jmax_ = a%get_ncols()
  endif

  if ((imax<imin).or.(jmax_<jmin_)) then 
    nz = 0
    return
  end if

  if (present(append)) then
    append_=append
  else
    append_=.false.
  endif
  if ((append_).and.(present(nzin))) then 
    nzin_ = nzin
  else
    nzin_ = 0
  endif
  if (present(rscale)) then 
    rscale_ = rscale
  else
    rscale_ = .false.
  endif
  if (present(cscale)) then 
    cscale_ = cscale
  else
    cscale_ = .false.
  endif
  if ((rscale_.or.cscale_).and.(present(iren))) then 
    info = psb_err_many_optional_arg_
    call psb_errpush(info,name,a_err='iren (rscale.or.cscale)')
    goto 9999
  end if

  if (a%is_dev()) call a%sync()
  call ell_getptn(imin,imax,jmin_,jmax_,a,nz,ia,ja,nzin_,append_,info,iren)
  
  if (rscale_) then 
    do i=nzin_+1, nzin_+nz
      ia(i) = ia(i) - imin + 1
    end do
  end if
  if (cscale_) then 
    do i=nzin_+1, nzin_+nz
      ja(i) = ja(i) - jmin_ + 1
    end do
  end if

  if (info /= psb_success_) goto 9999

  call psb_erractionrestore(err_act)
  return

9999 call psb_error_handler(err_act)
  return

contains

  subroutine ell_getptn(imin,imax,jmin,jmax,a,nz,ia,ja,nzin,append,info,&
       & iren)
    implicit none
    class(psb_c_ell_sparse_mat), intent(in)     :: a
    integer(psb_ipk_)                             :: imin,imax,jmin,jmax
    integer(psb_ipk_), intent(out)                :: nz
    integer(psb_ipk_), allocatable, intent(inout) :: ia(:), ja(:)
    integer(psb_ipk_), intent(in)                 :: nzin
    logical, intent(in)                           :: append
    integer(psb_ipk_)                             :: info
    integer(psb_ipk_), optional                   :: iren(:)
    integer(psb_ipk_)  :: nzin_, nza, idx,i,j,k, nzt, irw, lrw
    integer(psb_ipk_)  :: debug_level, debug_unit
    character(len=20) :: name='ell_getptn'

    debug_unit  = psb_get_debug_unit()
    debug_level = psb_get_debug_level()

    nza = a%get_nzeros()
    irw = imin
    lrw = min(imax,a%get_nrows())
    if (irw<0) then 
      info = psb_err_pivot_too_small_
      return
    end if

    if (append) then 
      nzin_ = nzin
    else
      nzin_ = 0
    endif

    nzt = sum(a%irn(irw:lrw))
    nz  = 0 

    call psb_ensure_size(nzin_+nzt,ia,info)
    if (info == psb_success_) call psb_ensure_size(nzin_+nzt,ja,info)

    if (info /= psb_success_) return
    
    if (present(iren)) then 
      do i=irw, lrw
        do j=1,a%irn(i)
          if ((jmin <= a%ja(i,j)).and.(a%ja(i,j)<=jmax)) then 
            nzin_ = nzin_ + 1
            nz    = nz + 1
            ia(nzin_)  = iren(i)
            ja(nzin_)  = iren(a%ja(i,j))
          end if
        enddo
      end do
    else
      do i=irw, lrw
        do j=1,a%irn(i)
          if ((jmin <= a%ja(i,j)).and.(a%ja(i,j)<=jmax)) then 
            nzin_ = nzin_ + 1
            nz    = nz + 1
            ia(nzin_)  = (i)
            ja(nzin_)  = (a%ja(i,j))
          end if
        enddo
      end do
    end if

  end subroutine ell_getptn

end subroutine psb_c_ell_csgetptn
