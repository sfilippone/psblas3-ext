!                Parallel Sparse BLAS   GPU plugin 
!      (C) Copyright 2013
!  
!                         Salvatore Filippone
!                         Alessandro Fanfarillo
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
  

subroutine  psb_d_csrg_reallocate_nz(nz,a) 
  
  use psb_base_mod
#ifdef HAVE_SPGPU
  use cusparse_mod
  use psb_d_csrg_mat_mod, psb_protect_name => psb_d_csrg_reallocate_nz
#else 
  use psb_d_csrg_mat_mod
#endif
  implicit none 
  integer(psb_ipk_), intent(in) :: nz
  class(psb_d_csrg_sparse_mat), intent(inout) :: a
  integer(psb_ipk_) :: m, nzrm,ld
  Integer(Psb_ipk_) :: err_act, info
  character(len=20)  :: name='d_csrg_reallocate_nz'
  logical, parameter :: debug=.false.

  call psb_erractionsave(err_act)
  info = psb_success_

  !
  ! What should this really do??? 
  ! 
  call a%psb_d_csr_sparse_mat%reallocate(nz)

#ifdef HAVE_SPGPU
  call a%to_gpu(info,nzrm=nz)
  if (info /= 0) goto 9999
#endif

  call psb_erractionrestore(err_act)
  return

9999 call psb_error_handler(err_act)

  return

end subroutine psb_d_csrg_reallocate_nz
