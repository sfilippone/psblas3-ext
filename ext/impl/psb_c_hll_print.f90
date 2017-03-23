!                Parallel Sparse BLAS   GPU plugin 
!      (C) Copyright 2013
!  
!                         Salvatore Filippone    University of Rome Tor Vergata
!                         Alessandro Fanfarillo  University of Rome Tor Vergata
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
  

subroutine psb_c_hll_print(iout,a,iv,head,ivr,ivc)
  
  use psb_base_mod
  use psb_c_hll_mat_mod, psb_protect_name => psb_c_hll_print
  implicit none 

  integer(psb_ipk_), intent(in)           :: iout
  class(psb_c_hll_sparse_mat), intent(in) :: a   
  integer(psb_ipk_), intent(in), optional :: iv(:)
  character(len=*), optional              :: head
  integer(psb_ipk_), intent(in), optional :: ivr(:), ivc(:)

  Integer(Psb_ipk_)  :: err_act
  character(len=20)  :: name='c_hll_print'
  character(len=*), parameter  :: datatype='complex'
  logical, parameter :: debug=.false.

  character(len=80)  :: frmtv 
  integer(psb_ipk_)  :: irs,ics,i,j, nmx, ni, nr, nc, nz, k, hksz, hk, mxrwl,ir, ix
  
  
  write(iout,'(a)') '%%MatrixMarket matrix coordinate real general'
  if (present(head)) write(iout,'(a,a)') '% ',head 
  write(iout,'(a)') '%'    
  write(iout,'(a,a)') '% COO'

  if (a%is_dev()) call a%sync()

  nr  = a%get_nrows()
  nc  = a%get_ncols()
  nz  = a%get_nzeros()
  nmx = max(nr,nc,1)
  ni  = floor(log10(1.0*nmx)) + 1
  hksz = a%get_hksz()

  if (datatype=='real') then 
    write(frmtv,'(a,i3.3,a,i3.3,a)') '(2(i',ni,',1x),es26.18,1x,2(i',ni,',1x))'
  else 
    write(frmtv,'(a,i3.3,a,i3.3,a)') '(2(i',ni,',1x),2(es26.18,1x),2(i',ni,',1x))'
  end if
  write(iout,*) nr, nc, nz 
  if(present(iv)) then 
    do i=1, nr
      irs   = (i-1)/hksz
      hk    = irs + 1
      mxrwl = (a%hkoffs(hk+1)-a%hkoffs(hk))/hksz
      k     = a%hkoffs(hk)
      k     = k + (i-(irs*hksz))
      do j=1,a%irn(i)
        write(iout,frmtv) iv(i),iv(a%ja(k)),a%val(k)
        k          = k + hksz
      end do
    enddo
  else      
    if (present(ivr).and..not.present(ivc)) then 
      do i=1, nr
        irs   = (i-1)/hksz
        hk    = irs + 1
        mxrwl = (a%hkoffs(hk+1)-a%hkoffs(hk))/hksz
        k     = a%hkoffs(hk)
        k     = k + (i-(irs*hksz))
        do j=1,a%irn(i)
          write(iout,frmtv) ivr(i),(a%ja(k)),a%val(k)
          k          = k + hksz
        end do
      enddo
    else if (present(ivr).and.present(ivc)) then 
      do i=1, nr
        irs   = (i-1)/hksz
        hk    = irs + 1
        mxrwl = (a%hkoffs(hk+1)-a%hkoffs(hk))/hksz
        k     = a%hkoffs(hk)
        k     = k + (i-(irs*hksz))
        do j=1,a%irn(i)
          write(iout,frmtv) ivr(i),ivc(a%ja(k)),a%val(k)
          k          = k + hksz
        end do
      enddo
    else if (.not.present(ivr).and.present(ivc)) then 
      do i=1, nr
        irs   = (i-1)/hksz
        hk    = irs + 1
        mxrwl = (a%hkoffs(hk+1)-a%hkoffs(hk))/hksz
        k     = a%hkoffs(hk)
        k     = k + (i-(irs*hksz))
        do j=1,a%irn(i)
          write(iout,frmtv) (i),ivc(a%ja(k)),a%val(k)
          k          = k + hksz
        end do
      enddo

    else if (.not.present(ivr).and..not.present(ivc)) then 

      do i=1, nr
        irs   = (i-1)/hksz
        hk    = irs + 1
        mxrwl = (a%hkoffs(hk+1)-a%hkoffs(hk))/hksz
        k     = a%hkoffs(hk)
        k     = k + (i-(irs*hksz))
        do j=1,a%irn(i)
          write(iout,frmtv) (i),(a%ja(k)),a%val(k)
          k          = k + hksz
        end do
      enddo
    endif
  endif
  
end subroutine psb_c_hll_print
