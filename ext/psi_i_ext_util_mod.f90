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
  integer(psb_ipk_), parameter     :: psb_hksz_def_ = 32
  integer(psb_ipk_), private, save :: psb_hksz      = psb_hksz_def_ 

contains

  function psi_get_hksz() result(res)
    implicit none 
    integer(psb_ipk_) :: res
    res = psb_hksz
  end function psi_get_hksz

  subroutine  psi_set_hksz(size) 
    implicit none 
    integer(psb_ipk_), intent(in) :: size
    if (size > 0) psb_hksz = size
  end subroutine psi_set_hksz
  

  !
  ! Compute offsets and allocation for DIAgonal storage.
  ! Input:
  !   nr,nc,nz,ia,ja
  ! Output:
  !     nd: number of nonzero diagonals
  !      d: d(k) contains the index inside offset of diagonal k,
  !         which is, if A(I,J) /= 0 then K=NR+J-I, or (optionally) 0.
  ! offset: for each of the ND nonzero diagonals, its offset J-I
  ! 
  ! 1. Optionally init D vector to zeros
  ! 2. Walk through the NZ pairs (I,J):
  !    a. if it is a new diagonal add to a heap;
  !    b. increase its population count stored in D(J-I+NR)
  !    c. Keep track of maximum population count.
  ! 3. Go through the ND diagonals, getting them K out of the heap in order:
  !    a. Set offset(i) to K-NR == J-I
  !    b. Set D(K) = i  or 0 (depending on cleard)
  !
  ! Setting to 0 allows to reuse this function in a loop in a dry run
  ! to estimate the allocation size for HDIA; without settng to 0 we
  ! would  need to zero the whole vector, resulting in a quadratic overall cost. 
  !
  !
  subroutine psi_dia_offset_from_coo(nr,nc,nz,ia,ja,nd,d,offset,info,&
       & initd,cleard) 
    use psb_base_mod
    
    implicit none 
    
    integer(psb_ipk_), intent(in)    :: nr, nc, nz, ia(:), ja(:)
    integer(psb_ipk_), intent(inout) :: d(:)
    integer(psb_ipk_), intent(out)   :: offset(:)
    integer(psb_ipk_), intent(out)   :: nd
    integer(psb_ipk_), intent(out)   :: info
    logical, intent(in), optional    :: initd,cleard
    
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
    call psb_init_heap(heap,info)
    if (info /= psb_success_) return

    do i=1,nz
      k = nr+ja(i)-ia(i)
      if (d(k) == 0) call psb_insert_heap(k,heap,info)
      d(k) = d(k) + 1 
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
