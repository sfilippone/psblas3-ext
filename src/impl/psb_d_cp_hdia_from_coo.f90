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
  

subroutine psb_d_cp_hdia_from_coo(a,b,info) 
  
  use psb_base_mod
  use psb_d_hdia_mat_mod, psb_protect_name => psb_d_cp_hdia_from_coo
  implicit none 

  class(psb_d_hdia_sparse_mat), intent(inout) :: a
  class(psb_d_coo_sparse_mat), intent(in)    :: b
  integer(psb_ipk_), intent(out)             :: info

  !locals
  type(psb_d_coo_sparse_mat) :: tmp
  integer(psb_ipk_)              :: ndiag,mi,mj,dm,nd,bi
  integer(psb_ipk_),allocatable  :: d(:,:),pres(:,:) 
  integer(psb_ipk_)              :: k,i,j,nc,nr,nza, nzd,h
  integer(psb_ipk_)              :: debug_level, debug_unit
  character(len=20)              :: name

  info = psb_success_
  ! This is to have fix_coo called behind the scenes
  call b%cp_to_coo(tmp,info)

  call tmp%fix(info)
  if (info /= psb_success_) return

  nr  = tmp%get_nrows()
  nc  = tmp%get_ncols()
  nza = tmp%get_nzeros()
  ! If it is sorted then we can lessen memory impact 
  a%psb_d_base_sparse_mat = tmp%psb_d_base_sparse_mat

  mi = maxval(tmp%ia)
  mj = maxval(tmp%ja)

  a%nblocks = ceiling(nr/real(a%hack))

  ndiag = a%hack+nc-1
  allocate(d(a%nblocks,ndiag),pres(a%nblocks,ndiag), a%hdia(a%nblocks))

  d=0
  pres=0

  do i=1,nza
     k = a%hack + tmp%ja(i)
     if(mod(tmp%ia(i),a%hack)==0) then
        k = k - a%hack
     else
        k = k - 1
     endif
     d(ceiling(tmp%ia(i)/real(a%hack)),k) = d(ceiling(tmp%ia(i)/real(a%hack)),k) + 1
 enddo

  dm = nr
  nd=0
  nzd = 0

  a%nzeros = nza

  do i=1,a%nblocks
    nzd = max(nzd,maxval(d(i,:)))
    nd = 0
    do j=1,ndiag
        if (d(i,j)>0) then
            pres(i,j)=1
            nd = nd + 1
        endif
    enddo

    call psb_realloc(a%hack,nd,a%hdia(i)%data,info)
    if (info /= 0) goto 9999
    a%hdia(i)%data = dzero
   
    call psb_realloc(nd,a%hdia(i)%offset,info)
    if (info /= 0) goto 9999
    a%hdia(i)%offset = dzero
    
    k=1
    
    do h=1,ndiag
       if(d(i,h)/=0) then
          a%hdia(i)%offset(k)=h-a%hack
          k=k+1
       endif
    enddo
    
    do h=1,nd
       a%hdia(i)%offset(h) = a%hdia(i)%offset(h) - (i-1)*a%hack
    enddo

    nzd = 0
    
 enddo

 do h=1,size(tmp%ia)
   bi = ceiling(tmp%ia(h)/real(a%hack))
   if(tmp%ia(h)>tmp%ja(h)) then
      nc = tmp%ja(h) - tmp%ia(h)
      do k=1,size(a%hdia(bi)%offset)
         if (a%hdia(bi)%offset(k)==nc) then
            nc = k
            exit
         endif
      enddo
      nr = mod(tmp%ia(h),a%hack)
      if (nr==0) nr = a%hack
   elseif(tmp%ia(h)<tmp%ja(h)) then
      nc = tmp%ja(h)-tmp%ia(h)
      do k=1,size(a%hdia(bi)%offset)
         if(a%hdia(bi)%offset(k)==nc) then
            nc = k!h
            exit
         endif
      enddo
      nr = mod(tmp%ia(h),a%hack)
      if (nr==0) nr = a%hack
   else
      nc = 0
      do k=1,size(a%hdia(bi)%offset)
         if(a%hdia(bi)%offset(k)==nc) then
            nc = k
            exit
         endif
      enddo
      nr = mod(tmp%ia(h),a%hack);
      if (nr==0) nr = a%hack
   endif
   a%hdia(bi)%data(nr,nc) = tmp%val(h)
  enddo

  deallocate(d,pres)

  call tmp%free

  a%dim = a%sizeof()

!!$  write(0,*) 'End of cp_hdia_from_coo', size(a%offset)
!!$  write(0,*) '   HDIAGS', (a%offset)
!!$  write(0,*)   '   DATA'
!!$  do i=1,nr
!!$    write(0,*) '       ', a%data(i,:) 
!!$  end do

  return

9999 continue
  info = psb_err_alloc_dealloc_
  return

end subroutine psb_d_cp_hdia_from_coo
