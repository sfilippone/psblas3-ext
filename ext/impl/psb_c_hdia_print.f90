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
  

subroutine psb_d_hdia_print(iout,a,iv,head,ivr,ivc)
  
  use psb_base_mod
  use psb_d_hdia_mat_mod, psb_protect_name => psb_d_hdia_print
  implicit none 

  integer(psb_ipk_), intent(in)           :: iout
  class(psb_d_hdia_sparse_mat), intent(in) :: a   
  integer(psb_ipk_), intent(in), optional :: iv(:)
  character(len=*), optional              :: head
  integer(psb_ipk_), intent(in), optional :: ivr(:), ivc(:)

  Integer(Psb_ipk_)  :: err_act
  character(len=20)  :: name='d_hdia_print'
  character(len=*), parameter  :: datatype='real'
  logical, parameter :: debug=.false.

  class(psb_d_coo_sparse_mat),allocatable :: acoo

  character(len=80)  :: frmtv 
  integer(psb_ipk_)  :: irs,ics,i,j, nmx, ni, nr, nc, nz

  if (present(head)) then 
    write(iout,'(a)') '%%MatrixMarket matrix coordinate real general'
    write(iout,'(a,a)') '% ',head 
    write(iout,'(a)') '%'    
    write(iout,'(a,a)') '% COO'
  endif

  nr  = a%get_nrows()
  nc  = a%get_ncols()
  nz  = a%get_nzeros()
  nmx = max(nr,nc,1)
  ni  = floor(log10(1.0*nmx)) + 1


  if (datatype=='real') then 
    write(frmtv,'(a,i3.3,a,i3.3,a)') '(2(i',ni,',1x),es26.18,1x,2(i',ni,',1x))'
  else 
    write(frmtv,'(a,i3.3,a,i3.3,a)') '(2(i',ni,',1x),2(es26.18,1x),2(i',ni,',1x))'
  end if
  write(iout,*) nr, nc, nz 
  ! if(present(iv)) then 
  !   do i=1, nr
  !     do j=1,a%irn(i)
  !       write(iout,frmtv) iv(i),iv(a%ja(i,j)),a%val(i,j)
  !     end do
  !   enddo
  ! else      
  !   if (present(ivr).and..not.present(ivc)) then 
  !     do i=1, nr
  !       do j=1,a%irn(i)
  !         write(iout,frmtv) ivr(i),(a%ja(i,j)),a%val(i,j)
  !       end do
  !     enddo
  !   else if (present(ivr).and.present(ivc)) then 
  !     do i=1, nr
  !       do j=1,a%irn(i)
  !         write(iout,frmtv) ivr(i),ivc(a%ja(i,j)),a%val(i,j)
  !       end do
  !     enddo
  !   else if (.not.present(ivr).and.present(ivc)) then 
  !     do i=1, nr
  !       do j=1,a%irn(i)
  !         write(iout,frmtv) (i),ivc(a%ja(i,j)),a%val(i,j)
  !       end do
  !     enddo
  !   else if (.not.present(ivr).and..not.present(ivc)) then 

  !   endif
  ! endif
!!$
!!$  do i=1, nz
!!$     write(iout,frmtv) a%ia(i),a%ja(i),a%val(i)
!!$  enddo

  !call acoo%free()

  ! do i=1, size(a%data,1)
  !    do j=1,size(acoo%data,2)
  !       write(iout,frmtv) (i),(a%data(i,j))
  !    end do
  ! enddo
  
end subroutine psb_d_hdia_print
