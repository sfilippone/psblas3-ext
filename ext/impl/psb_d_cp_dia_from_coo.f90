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

subroutine psb_d_cp_dia_from_coo(a,b,info) 

  use psb_base_mod
  use psb_d_dia_mat_mod, psb_protect_name => psb_d_cp_dia_from_coo
  implicit none 

  class(psb_d_dia_sparse_mat), intent(inout) :: a
  class(psb_d_coo_sparse_mat), intent(in)    :: b
  integer(psb_ipk_), intent(out)             :: info

  !locals
  type(psb_d_coo_sparse_mat) :: tmp
  integer(psb_ipk_)              :: ndiag,nd
  integer(psb_ipk_),allocatable  :: d(:)
  integer(psb_ipk_)              :: k,i,j,nc,nr,nza,nrd,ir,ic
  integer(psb_ipk_)              :: debug_level, debug_unit
  character(len=20)              :: name

  info = psb_success_
  ! This is to have fix_coo called behind the scenes
  call b%cp_to_coo(tmp,info)
  if (info /= psb_success_) return
  if (.not.tmp%is_sorted()) call tmp%fix(info)
  if (info /= psb_success_) return

  nr  = tmp%get_nrows()
  nc  = tmp%get_ncols()
  nza = tmp%get_nzeros()
  ! If it is sorted then we can lessen memory impact 
  a%psb_d_base_sparse_mat = tmp%psb_d_base_sparse_mat

  ndiag = nr+nc-1
  allocate(d(ndiag),stat=info)
  if (info /= 0) goto 9999
  call psb_realloc(ndiag,a%offset,info)
  if (info /= 0) goto 9999

  call psi_diacnt_from_coo(nr,nc,nza,tmp%ia,tmp%ja,nrd,nd,d,info,initd=.true.)
  call psi_offset_from_d(nr,nc,d,a%offset,info)

  call psb_realloc(nd,a%offset,info)
  if (info /= 0) goto 9999
  call psb_realloc(nrd,nd,a%data,info) 
  if (info /= 0) goto 9999
  a%nzeros = nza

  call psi_xtr_dia_from_coo(nr,nza,tmp%ia,tmp%ja,tmp%val,d,a%data,info,initd=.true.)

  do i=1,nd
    k=a%offset(i)+nr
    d(k) = 0
  end do
  if (any(d(1:ndiag)/=0)) then
    write(*,*) 'Check from dia_from_coo: some entries in D not zeroed on exitr'
  end if
  deallocate(d,stat=info)
  if (info /= 0) goto 9999

  call tmp%free()


  return

9999 continue
  info = psb_err_alloc_dealloc_
  return

end subroutine psb_d_cp_dia_from_coo
