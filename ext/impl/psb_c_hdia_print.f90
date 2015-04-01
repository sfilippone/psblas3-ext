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
subroutine psb_c_hdia_print(iout,a,iv,head,ivr,ivc)
  
  use psb_base_mod
  use psb_c_hdia_mat_mod, psb_protect_name => psb_c_hdia_print
  use psi_ext_util_mod
  implicit none 

  integer(psb_ipk_), intent(in)           :: iout
  class(psb_c_hdia_sparse_mat), intent(in) :: a   
  integer(psb_ipk_), intent(in), optional :: iv(:)
  character(len=*), optional              :: head
  integer(psb_ipk_), intent(in), optional :: ivr(:), ivc(:)

  Integer(Psb_ipk_)  :: err_act
  character(len=20)  :: name='hdia_print'
  character(len=*), parameter  :: datatype='complex'
  logical, parameter :: debug=.false.

  class(psb_c_coo_sparse_mat),allocatable :: acoo

  character(len=80)  :: frmtv 
  integer(psb_ipk_)  :: irs,ics,i,j, nmx, ni, nr, nc, nz
  integer(psb_ipk_)  :: nhacks, hacksize,maxnzhack, k, ncd,ib, nzhack, info,&
       & hackfirst, hacknext
  integer(psb_ipk_), allocatable :: ia(:), ja(:)
  complex(psb_spk_), allocatable    :: val(:) 

  if (present(head)) then 
    write(iout,'(a)') '%%MatrixMarket matrix coordinate complex general'
    write(iout,'(a,a)') '% ',head 
    write(iout,'(a)') '%'    
    write(iout,'(a,a)') '% HDIA'
  endif

  nr  = a%get_nrows()
  nc  = a%get_ncols()
  nz  = a%get_nzeros()
  nmx = max(nr,nc,1)
  ni  = floor(log10(1.0*nmx)) + 1

  nhacks   = a%nhacks
  hacksize = a%hacksize
  maxnzhack = 0
  do k=1, nhacks
    maxnzhack = max(maxnzhack,(a%hackoffsets(k+1)-a%hackoffsets(k)))
  end do
  maxnzhack = hacksize*maxnzhack
  allocate(ia(maxnzhack),ja(maxnzhack),val(maxnzhack),stat=info)
  if (info /= 0) return 

  if (datatype=='complex') then 
    write(frmtv,'(a,i3.3,a,i3.3,a)') '(2(i',ni,',1x),es26.18,1x,2(i',ni,',1x))'
  else 
    write(frmtv,'(a,i3.3,a,i3.3,a)') '(2(i',ni,',1x),2(es26.18,1x),2(i',ni,',1x))'
  end if

  write(iout,*) nr, nc, nz 
  do k=1, nhacks
    i = (k-1)*hacksize + 1
    ib = min(hacksize,nr-i+1) 
    hackfirst = a%hackoffsets(k)
    hacknext  = a%hackoffsets(k+1)
    ncd = hacknext-hackfirst
    
    call psi_c_xtr_coo_from_dia(nr,nc,&
           & ia, ja, val, nzhack,&
           & hacksize,ncd,&
           & a%val((hacksize*hackfirst)+1:hacksize*hacknext),&
           & a%diaOffsets(hackfirst+1:hacknext),info,rdisp=(i-1))
    !nzhack = sum(ib - abs(a%diaOffsets(hackfirst+1:hacknext)))
    
    if(present(iv)) then 
      do j=1,nzhack
        write(iout,frmtv) iv(ia(j)),iv(ja(j)),val(j)
      enddo
    else      
      if (present(ivr).and..not.present(ivc)) then 
        do j=1,nzhack
          write(iout,frmtv) ivr(ia(j)),ja(j),val(j)
        enddo
      else if (present(ivr).and.present(ivc)) then 
        do j=1,nzhack
          write(iout,frmtv) ivr(ia(j)),ivc(ja(j)),val(j)
        enddo
      else if (.not.present(ivr).and.present(ivc)) then 
        do j=1,nzhack
          write(iout,frmtv) ia(j),ivc(ja(j)),val(j)
        enddo
      else if (.not.present(ivr).and..not.present(ivc)) then 
        do j=1,nzhack
          write(iout,frmtv) ia(j),ja(j),val(j)
        enddo
      endif
    end if
      
  end do
  
end subroutine psb_c_hdia_print
