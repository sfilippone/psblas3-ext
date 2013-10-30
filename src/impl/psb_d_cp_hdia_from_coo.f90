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
  integer(psb_ipk_)              :: ndiag,mi,mj,dm,nd,bi,w
  integer(psb_ipk_),allocatable  :: d(:)
  integer(psb_ipk_)              :: k,i,j,nc,nr,nza, nzd,h,hack,nblocks
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
  nblocks = a%nblocks
  hack = a%hack

  ndiag = hack+nc-1
  allocate(d(ndiag), a%hdia(nblocks),a%offset(nblocks))

  d=0
  nzd=0
  ! dm represents the block number
  dm = ceiling(tmp%ia(1)/real(hack))

  do i=1,nza
     k = hack + tmp%ja(i)
     if(mod(tmp%ia(i),hack)==0) then
        k = k - hack
     else
        !k = k - 1
        k = k - mod(tmp%ia(i),hack)
     endif

     if(dm/=ceiling(tmp%ia(i)/real(hack))) then
        ! Manage old d, build the structs and set to 0 the new d
        nzd = max(nzd,maxval(d(:)))
        nd = 0
        do j=1,ndiag
           if (d(j)>0) then
              nd = nd + 1
           endif
        enddo
        
        !call psb_realloc(a%hack,nd,a%hdia(i)%data,info)
        allocate(a%hdia(dm)%data(hack,nd))
        if (info /= 0) goto 9999
        a%hdia(dm)%data = dzero
        
        !call psb_realloc(nd,a%offset(i)%off,info)
        allocate(a%offset(dm)%off(nd))
        if (info /= 0) goto 9999
        a%offset(dm)%off = dzero
        
        w=1
        
        do h=1,ndiag
           if(d(h)/=0) then
              a%offset(dm)%off(w)=h-hack
              w=w+1
           endif
        enddo
        
        do h=1,nd
           a%offset(dm)%off(h) = a%offset(dm)%off(h) - (dm-1)*hack
        enddo
        
        nzd = 0
        dm = ceiling(tmp%ia(i)/real(hack))
        d=0
     endif

     d(k) = d(k) + 1

 enddo

 nzd = max(nzd,maxval(d(:)))
 nd = 0
 do j=1,ndiag
    if (d(j)>0) then
       nd = nd + 1
    endif
 enddo
 
 !call psb_realloc(a%hack,nd,a%hdia(i)%data,info)
 allocate(a%hdia(dm)%data(hack,nd))
 if (info /= 0) goto 9999
 a%hdia(dm)%data = dzero
 
 !call psb_realloc(nd,a%offset(i)%off,info)
 allocate(a%offset(dm)%off(nd))
 if (info /= 0) goto 9999
 a%offset(dm)%off = dzero
 
 w=1
 
 do h=1,ndiag
    if(d(h)/=0) then
       a%offset(dm)%off(w)=h-hack
       w=w+1
    endif
 enddo
 
 do h=1,nd
    a%offset(dm)%off(h) = a%offset(dm)%off(h) - (dm-1)*hack
 enddo
 
 nzd = 0
 dm = ceiling(tmp%ia(i)/real(hack))
 d=0
 
 dm = nr
 nd=0
 nzd = 0
 
 a%nzeros = nza
 
 do h=1,size(tmp%ia)
   bi = ceiling(tmp%ia(h)/real(hack))
   if(tmp%ia(h)>tmp%ja(h)) then
      nc = tmp%ja(h) - tmp%ia(h)
      do k=1,size(a%offset(bi)%off)
         if (a%offset(bi)%off(k)==nc) then
            nc = k
            exit
         endif
      enddo
      nr = mod(tmp%ia(h),hack)
      if (nr==0) nr = hack
   elseif(tmp%ia(h)<tmp%ja(h)) then
      nc = tmp%ja(h)-tmp%ia(h)
      do k=1,size(a%offset(bi)%off)
         if(a%offset(bi)%off(k)==nc) then
            nc = k!h
            exit
         endif
      enddo
      nr = mod(tmp%ia(h),hack)
      if (nr==0) nr = hack
   else
      nc = 0
      do k=1,size(a%offset(bi)%off)
         if(a%offset(bi)%off(k)==nc) then
            nc = k
            exit
         endif
      enddo
      nr = mod(tmp%ia(h),hack);
      if (nr==0) nr = hack
   endif
   a%hdia(bi)%data(nr,nc) = tmp%val(h)
  enddo

  deallocate(d)

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
