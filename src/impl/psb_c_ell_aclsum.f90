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
  

subroutine psb_c_ell_aclsum(d,a) 
  
  use psb_base_mod
  use psb_c_ell_mat_mod, psb_protect_name => psb_c_ell_aclsum
  implicit none 
  class(psb_c_ell_sparse_mat), intent(in) :: a
  real(psb_spk_), intent(out)              :: d(:)

  integer(psb_ipk_) :: i,j,k,m,n, nnz, ir, jc, nc
  logical           :: tra
  Integer(Psb_ipk_) :: err_act, info, int_err(5)
  character(len=20) :: name='aclsum'
  logical, parameter :: debug=.false.

  call psb_erractionsave(err_act)

  m = a%get_nrows()
  n = a%get_ncols()
  if (size(d) < n) then 
    info=psb_err_input_asize_small_i_
    int_err(1) = 1
    int_err(2) = size(d)
    int_err(3) = n
    call psb_errpush(info,name,i_err=int_err)
    goto 9999
  end if

  if (a%is_unit()) then 
    d = sone
  else
    d = szero
  end if

  do i=1, m
    do j=1,a%irn(i)
      k = a%ja(i,j)
      d(k) = d(k) + abs(a%val(i,j))
    end do
  end do



  call psb_erractionrestore(err_act)
  return  

9999 continue
  call psb_erractionrestore(err_act)

  if (err_act == psb_act_abort_) then
    call psb_error()
    return
  end if
  return

end subroutine psb_c_ell_aclsum
