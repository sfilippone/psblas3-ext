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
  

module psi_dia_util_mod

  use psb_base_mod, only : psb_ipk_, psb_dpk_

  interface psi_xtr_dia_from_coo
    module procedure psi_d_xtr_dia_from_coo
  end interface psi_xtr_dia_from_coo

contains
  
  subroutine psi_diacnt_from_coo(nr,nc,nz,ia,ja,nrd,nd,d,offset,info, initd) 
    use psb_base_mod, only : psb_ipk_, psb_success_
    
    implicit none 
    
    integer(psb_ipk_), intent(in)   :: nr, nc, nz, ia(:), ja(:)
    integer(psb_ipk_), intent(out)  :: nrd, nd, d(:), offset(:)
    integer(psb_ipk_), intent(out)  :: info
    logical, intent(in), optional   :: initd
    
    !locals
    integer(psb_ipk_)              :: k,i,j,ir,ic, ndiag
    logical                        :: initd_
    character(len=20)              :: name
    
    info = psb_success_
    initd_ = .true.
    if (present(initd)) initd_ = initd
    if (initd_) d(:) = 0 
    
    ndiag = nr+nc-1  
    nd    = 0
    nrd   = 0
    do i=1,nz
      k = nr+ja(i)-ia(i)
      if (d(k) == 0) nd = nd + 1 
      d(k) = d(k) + 1 
      nrd = max(nrd,d(k))
    enddo
    
    k = 1
    do i=1,ndiag
      if (d(i)/=0) then
        offset(k)=i-nr
        d(i) = k
        k    = k+1
      end if
    end do
    
    return
  end subroutine psi_diacnt_from_coo
  
  
  
  subroutine psi_d_xtr_dia_from_coo(nr,nz,ia,ja,val,d,data,info,initd)    
    use psb_base_mod, only : psb_ipk_, psb_success_, psb_dpk_, dzero
    
    implicit none 

    integer(psb_ipk_), intent(in)  :: nr, nz, ia(:), ja(:), d(:)
    real(psb_dpk_),    intent(in)  :: val(:)
    real(psb_dpk_),    intent(out) :: data(:,:)
    integer(psb_ipk_), intent(out) :: info
    logical, intent(in), optional  :: initd
    
    !locals
    logical                        :: initd_

    integer(psb_ipk_) :: i,ir,ic,k

    info = psb_success_
    info = psb_success_
    initd_ = .true.
    if (present(initd)) initd_ = initd
    if (initd_) data(:,:) = dzero
    
    do i=1,nz
      ir = ia(i)
      k  = ja(i) - ir
      ic = d(nr+k)
      data(ir,ic) = val(i)
    enddo
    
    
  end subroutine psi_d_xtr_dia_from_coo
  
  
end module psi_dia_util_mod
