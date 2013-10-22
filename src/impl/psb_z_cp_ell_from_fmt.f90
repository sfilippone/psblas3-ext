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
  

subroutine psb_z_cp_ell_from_fmt(a,b,info) 
  
  use psb_base_mod
  use psb_z_ell_mat_mod, psb_protect_name => psb_z_cp_ell_from_fmt
  implicit none 

  class(psb_z_ell_sparse_mat), intent(inout) :: a
  class(psb_z_base_sparse_mat), intent(in)   :: b
  integer(psb_ipk_), intent(out)             :: info

  !locals
  type(psb_z_coo_sparse_mat) :: tmp

  info = psb_success_

  select type (b)
  type is (psb_z_coo_sparse_mat) 
    call a%cp_from_coo(b,info)

  type is (psb_z_ell_sparse_mat) 
    a%psb_z_base_sparse_mat = b%psb_z_base_sparse_mat
    if (info == 0) call psb_safe_cpy( b%irn,   a%irn ,  info)
    if (info == 0) call psb_safe_cpy( b%idiag, a%idiag, info)
    if (info == 0) call psb_safe_cpy( b%ja ,   a%ja  ,  info)
    if (info == 0) call psb_safe_cpy( b%val,   a%val ,  info)

  class default
    call b%cp_to_coo(tmp,info)
    if (info == psb_success_) call a%mv_from_coo(tmp,info)
  end select
end subroutine psb_z_cp_ell_from_fmt
