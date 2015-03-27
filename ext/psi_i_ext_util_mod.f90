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
  

module psi_i_ext_util_mod

  use psb_base_mod, only : psb_ipk_


contains
  
  subroutine psi_diacnt_from_coo(nr,nc,nz,ia,ja,nrd,nd,d,info, initd) 
    use psb_base_mod, only : psb_ipk_, psb_success_
    
    implicit none 
    
    integer(psb_ipk_), intent(in)    :: nr, nc, nz, ia(:), ja(:)
    integer(psb_ipk_), intent(out)   :: nrd, nd
    integer(psb_ipk_), intent(inout) :: d(:)
    integer(psb_ipk_), intent(out)   :: info
    logical, intent(in), optional    :: initd
    
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
    
    return
  end subroutine psi_diacnt_from_coo
  
  subroutine psi_offset_from_d(nr,nc,d,offset,info) 
    use psb_base_mod, only : psb_ipk_, psb_success_
    
    implicit none 
    
    integer(psb_ipk_), intent(in)    :: nr, nc
    integer(psb_ipk_), intent(inout) :: d(:)
    integer(psb_ipk_), intent(out)   :: offset(:)
    integer(psb_ipk_), intent(out)   :: info
    
    !locals
    integer(psb_ipk_)              :: k,i,j,ir,ic, ndiag
    character(len=20)              :: name
    
    info = psb_success_
    
    ndiag = nr+nc-1  
    k = 1
    do i=1,ndiag
      if (d(i)/=0) then
        offset(k)=i-nr
        d(i) = k
        k    = k+1
      end if
    end do
    
    return
  end subroutine psi_offset_from_d
  

  
  subroutine psi_dia_offset_from_coo(nr,nc,nz,ia,ja,nrd,nd,d,offset,info,initd,cleard) 
    use psb_base_mod
    
    implicit none 
    
    integer(psb_ipk_), intent(in)   :: nr, nc, nz, ia(:), ja(:)
    integer(psb_ipk_), intent(inout) :: d(:)
    integer(psb_ipk_), intent(out)   :: offset(:)
    integer(psb_ipk_), intent(out)  :: nrd, nd
    integer(psb_ipk_), intent(out)  :: info
    logical, intent(in), optional   :: initd,cleard
    
    type(psb_int_heap)             :: heap
    integer(psb_ipk_)              :: k,i,j,ir,ic, ndiag, id
    logical                        :: initd_, cleard_
    character(len=20)              :: name
    
    info = psb_success_
    initd_ = .true.
    if (present(initd)) initd_ = initd
    cleard_ = .false.
    if (present(cleard)) cleard_ = cleard

    if (initd_) d(:) = 0 
    
    ndiag = nr+nc-1  
    if (size(d)<ndiag) then 
      info = -8
      return
    end if
    nrd   = 0
    call psb_init_heap(heap,info)
    if (info /= psb_success_) return

    do i=1,nz
      k = nr+ja(i)-ia(i)
      if (d(k) == 0) call psb_insert_heap(k,heap,info)
      d(k) = d(k) + 1 
      nrd = max(nrd,d(k))
    enddo
    nd  = psb_howmany_heap(heap)
    if (size(offset)<nd) then 
      info = -9 
      return
    end if
    if (cleard_) then 
      do i=1, nd
        call psb_heap_get_first(id,heap,info)
        offset(i) = id - nr
        d(id) = 0
      end do
    else
      do i=1, nd
        call psb_heap_get_first(id,heap,info)
        offset(i) = id - nr
        d(id) = i 
      end do
    end if
    
  end subroutine psi_dia_offset_from_coo
  
end module psi_i_ext_util_mod
