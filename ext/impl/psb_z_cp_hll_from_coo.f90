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
  

subroutine psb_z_cp_hll_from_coo(a,b,info) 
  
  use psb_base_mod
  use psb_z_hll_mat_mod, psb_protect_name => psb_z_cp_hll_from_coo
  implicit none 

  class(psb_z_hll_sparse_mat), intent(inout) :: a
  class(psb_z_coo_sparse_mat), intent(in)    :: b
  integer(psb_ipk_), intent(out)             :: info

  !locals
  type(psb_z_coo_sparse_mat) :: tmp
  Integer(Psb_ipk_)   :: nza, nr, i,j,irw, idl,err_act, nc, isz,irs
  integer(psb_ipk_)   :: nzm, ir, ic, k, hksz, hk, mxrwl, noffs, kc
  integer(psb_ipk_)   :: debug_level, debug_unit
  character(len=20)   :: name='hll_from_coo'

  info = psb_success_
  debug_unit  = psb_get_debug_unit()
  debug_level = psb_get_debug_level()
  ! This is to have fix_coo called behind the scenes
  if (b%is_by_rows()) then 
    call inner_convert_from_coo(a,b,info)
  else
    call b%cp_to_coo(tmp,info)
    
    call tmp%fix(info)
    
    if (info /= psb_success_) return
    call inner_convert_from_coo(a,tmp,info)
    
    call tmp%free()
  end if

  return

9999 continue
  info = psb_err_alloc_dealloc_
  return

contains

  subroutine inner_convert_from_coo(a,tmp,info)
    implicit none 
    class(psb_z_hll_sparse_mat), intent(inout) :: a
    class(psb_z_coo_sparse_mat), intent(in)    :: tmp
    integer(psb_ipk_), intent(out)             :: info

    !locals
    Integer(Psb_ipk_)   :: nza, nr, i,j,irw, idl,err_act, nc, isz,irs
    integer(psb_ipk_)   :: nzm, ir, ic, k, hksz, hk, mxrwl, noffs, kc
    nr  = tmp%get_nrows()
    nc  = tmp%get_ncols()
    nza = tmp%get_nzeros()
    ! If it is sorted then we can lessen memory impact 
    a%psb_z_base_sparse_mat = tmp%psb_z_base_sparse_mat

    ! First compute the number of nonzeros in each row.
    call psb_realloc(nr,a%irn,info) 
    if (info /= 0) goto 9999
    a%irn = 0
    do i=1, nza
      a%irn(tmp%ia(i)) = a%irn(tmp%ia(i)) + 1
    end do

    a%nzt = nza
    ! Second. Figure out the block offsets. 
    call a%set_hksz(psb_hksz_def_)
    hksz  = a%get_hksz()
    noffs = (nr+hksz-1)/hksz
    call psb_realloc(noffs+1,a%hkoffs,info) 
    if (info /= 0) goto 9999
    a%hkoffs(1) = 0
    j=1
    do i=1,nr,hksz
      ir    = min(hksz,nr-i+1) 
      mxrwl = a%irn(i)
      do k=1,ir-1
        mxrwl = max(mxrwl,a%irn(i+k))
      end do
      a%hkoffs(j+1) = a%hkoffs(j) + mxrwl*hksz
      j = j + 1 
    end do

    !
    ! At this point a%hkoffs(noffs+1) contains the allocation
    ! size a%ja a%val. 
    ! 
    isz = a%hkoffs(noffs+1)
    call psb_realloc(nr,a%idiag,info) 
    if (info == 0) call psb_realloc(isz,a%ja,info) 
    if (info == 0) call psb_realloc(isz,a%val,info) 
    if (info /= 0) goto 9999
    ! Init last chunk of data
    nzm = a%hkoffs(noffs+1)-a%hkoffs(noffs)
    a%val(isz-(nzm-1):isz) = zzero
    a%ja(isz-(nzm-1):isz)  = nr
    !
    ! Now copy everything, noting the position of the diagonal. 
    !
    kc = 1 
    k  = 1
    do i=1, nr,hksz
      ir    = min(hksz,nr-i+1) 
      irs   = (i-1)/hksz
      hk    = irs + 1
      isz   = (a%hkoffs(hk+1)-a%hkoffs(hk))
      mxrwl = isz/hksz
      nza   = sum(a%irn(i:i+ir-1))
      call inner_copy(i,ir,mxrwl,tmp%ia(kc:kc+nza-1),&
           & tmp%ja(kc:kc+nza-1),tmp%val(kc:kc+nza-1),&
           & a%ja(k:k+isz-1),a%val(k:k+isz-1),a%irn(i:i+ir-1),&
           & a%idiag(i:i+ir-1),hksz)
      k  = k + isz
      kc = kc + nza

    enddo

    ! Third copy the other stuff
    if (info /= 0) goto 9999
    call a%set_sorted(.true.)

  end subroutine inner_convert_from_coo

  subroutine  inner_copy(i,ir,mxrwl,iac,&
       & jac,valc,ja,val,irn,diag,ld)
    implicit none 
    integer(psb_ipk_) :: i,ir,mxrwl,ld
    integer(psb_ipk_) :: iac(*),jac(*),ja(ld,*),irn(*),diag(*)
    complex(psb_dpk_)    :: valc(*), val(ld,*)
    
    integer(psb_ipk_) :: ii,jj,kk, kc,nc
    kc = 1
    do ii = 1, ir
      nc = irn(ii)
      do jj=1,nc
        if (iac(kc) /= i+ii-1) write(0,*) 'Copy mismatch',iac(kc),i,ii,i+ii-1
        if (jac(kc) == i+ii-1) a%idiag(ii) = jj
        ja(ii,jj)  = jac(kc) 
        val(ii,jj) = valc(kc) 
        kc = kc + 1
      end do
      do jj = nc+1,mxrwl
        ja(ii,jj)  = i+ii-1
        val(ii,jj) = zzero
      end do
    end do
  end subroutine inner_copy

end subroutine psb_z_cp_hll_from_coo
