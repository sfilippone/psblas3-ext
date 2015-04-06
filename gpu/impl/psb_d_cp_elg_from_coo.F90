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
  

subroutine psb_d_cp_elg_from_coo(a,b,info) 
  
  use psb_base_mod
#ifdef HAVE_SPGPU
  use elldev_mod
  use psb_vectordev_mod
  use psb_d_elg_mat_mod, psb_protect_name => psb_d_cp_elg_from_coo
  use psi_ext_util_mod
  use psb_gpu_env_mod
#else 
  use psb_d_elg_mat_mod
#endif
  implicit none 

  class(psb_d_elg_sparse_mat), intent(inout) :: a
  class(psb_d_coo_sparse_mat), intent(in)    :: b
  integer(psb_ipk_), intent(out)             :: info

  !locals
  Integer(Psb_ipk_)   :: nza, nr, i,j,k, idl,err_act, nc, nzm, ir, ic, ld, hacksize
  integer(psb_ipk_)   :: debug_level, debug_unit
  character(len=20)   :: name
  type(psb_d_coo_sparse_mat)  :: tmp
#ifdef HAVE_SPGPU
  type(elldev_parms) :: gpu_parms
#endif

  info = psb_success_
#ifdef HAVE_SPGPU
  hacksize = psb_gpu_WarpSize()
#else
  hacksize = 1
#endif
  if (b%is_by_rows()) then
    call psi_d_convert_ell_from_coo(a,b,info,hacksize=hacksize)
  else
    call b%cp_to_coo(tmp,info)
    if (info == psb_success_)  call psi_d_convert_ell_from_coo(a,tmp,&
         & info,hacksize=hacksize) 
    if (info == psb_success_)  call tmp%free()
  end if
  if (info /= psb_success_) goto 9999

#ifdef HAVE_SPGPU
  call a%to_gpu(info)  
  if (info /= psb_success_) goto 9999
#endif  

  
  return

9999 continue
  info = psb_err_alloc_dealloc_
  return


end subroutine psb_d_cp_elg_from_coo
