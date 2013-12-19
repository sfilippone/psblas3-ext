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
  

subroutine psb_z_cp_elg_from_coo(a,b,info) 
  
  use psb_base_mod
#ifdef HAVE_SPGPU
  use elldev_mod
  use psb_vectordev_mod
  use psb_z_elg_mat_mod, psb_protect_name => psb_z_cp_elg_from_coo
#else 
  use psb_z_elg_mat_mod
#endif
  implicit none 

  class(psb_z_elg_sparse_mat), intent(inout) :: a
  class(psb_z_coo_sparse_mat), intent(in)    :: b
  integer(psb_ipk_), intent(out)             :: info

  !locals
  Integer(Psb_ipk_)   :: nza, nr, i,j,k, idl,err_act, nc, nzm, ir, ic, ld
  integer(psb_ipk_)   :: debug_level, debug_unit
  character(len=20)   :: name
  type(psb_z_coo_sparse_mat)  :: tmp
#ifdef HAVE_SPGPU
  type(elldev_parms) :: gpu_parms
#endif

  info = psb_success_

  call b%cp_to_coo(tmp, info)
  call tmp%fix(info)
  nr  = tmp%get_nrows()
  nc  = tmp%get_ncols()
  nza = tmp%get_nzeros()
  ! If it is sorted then we can lessen memory impact 
  a%psb_z_base_sparse_mat = tmp%psb_z_base_sparse_mat

  ! First compute the number of nonzeros in each row.
  call psb_realloc(nr,a%irn,info) 
  if (info /= 0) goto 9999
  a%irn = 0
  do i=1, nza
    a%irn(tmp%ia(i)) = a%irn(tmp%ia(i)) + 1
  end do
  nzm = 0 
  do i=1, nr
    nzm = max(nzm,a%irn(i))
    a%irn(i) = 0
  end do
  ! Second: copy the column indices.
  call psb_realloc(nr,a%idiag,info) 
#ifdef HAVE_SPGPU
  gpu_parms = FgetEllDeviceParams(nr,nzm,nc,spgpu_type_double,1)
  ld  = gpu_parms%pitch
  nzm = gpu_parms%maxRowSize
#else
  ld = nr
#endif

  if (info == 0) call psb_realloc(ld,nzm,a%ja,info) 
  if (info /= 0) goto 9999
  do i=1, nza
    ir = tmp%ia(i)
    ic = tmp%ja(i)
    j  = a%irn(ir) + 1 
    a%ja(ir,j) = ic
    a%irn(ir)  = j
  end do
  ! Third copy the other stuff
  deallocate(tmp%ia,tmp%ja,stat=info) 
  if (info == 0) call psb_realloc(nr,a%idiag,info)
  if (info == 0) call psb_realloc(ld,nzm,a%val,info)
  if (info /= 0) goto 9999
  k = 0 
  do i=1, nr
    a%idiag(i) = 0 
    do j=1, a%irn(i)
      k = k + 1 
      a%val(i,j) = tmp%val(k)
      if (i==a%ja(i,j)) a%idiag(i)=j
    end do
    do j=a%irn(i)+1, nzm
      a%ja(i,j) = i
      a%val(i,j) = zzero
    end do
  end do

  call tmp%free()

#ifdef HAVE_SPGPU
  call a%to_gpu(info)
  if (info /= 0) goto 9999
#endif

  return

9999 continue
  info = psb_err_alloc_dealloc_
  return


end subroutine psb_z_cp_elg_from_coo
