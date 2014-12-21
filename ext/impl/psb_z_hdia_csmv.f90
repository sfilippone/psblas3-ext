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
  

subroutine psb_d_hdia_csmv(alpha,a,x,beta,y,info) 
  
  use psb_base_mod
  use psb_d_hdia_mat_mod, psb_protect_name => psb_d_hdia_csmv
  implicit none 
  class(psb_d_hdia_sparse_mat), intent(in) :: a
  real(psb_dpk_), intent(in)        :: alpha, beta, x(:)
  real(psb_dpk_), intent(inout)     :: y(:)
  integer(psb_ipk_), intent(out)     :: info
  !character, optional, intent(in)    :: trans

  character :: trans_
  integer(psb_ipk_)  :: i,j,k,m,n, nnz, ir, jc,nr,nc
  real(psb_dpk_)    :: acc
  logical            :: tra, ctra
  integer(psb_ipk_)  :: err_act
  character(len=20)  :: name='d_hdia_csmv'
  logical, parameter :: debug=.false.
  real :: start, finish
  call psb_erractionsave(err_act)
  info = psb_success_

  if (.not.a%is_asb()) then 
    info = psb_err_invalid_mat_state_
    call psb_errpush(info,name)
    goto 9999
  endif

    n = a%get_ncols()
    m = a%get_nrows()

  if (size(x,1)<n) then 
    info = 36
    call psb_errpush(info,name,i_err=(/3*ione,n,izero,izero,izero/))
    goto 9999
  end if

  if (size(y,1)<m) then 
    info = 36
    call psb_errpush(info,name,i_err=(/5*ione,m,izero,izero,izero/))
    goto 9999
  end if

  if (beta == dzero) then
     do i = 1, m
        y(i) = dzero
     enddo
  else
     do  i = 1, m
        y(i) = beta*y(i)
     end do
  endif

  do i=1,a%nblocks
     call psb_d_hdia_csmv_inner(m,n,alpha,size(a%hdia(i)%data,1),&
          & size(a%hdia(i)%data,2),a%hdia(i)%data,a%offset(i)%off,x,beta,y,i)
  enddo
  
  call psb_erractionrestore(err_act)
  return

9999 call psb_error_handler(err_act)
  return

contains

  subroutine psb_d_hdia_csmv_inner(m,n,alpha,nr,nc,data,off,&
       &x,beta,y,bl) 
    integer(psb_ipk_), intent(in)   :: m,n,nr,nc,off(:),bl
    real(psb_dpk_), intent(in)     :: alpha, beta, x(:),data(:,:)
    real(psb_dpk_), intent(inout)  :: y(:)

    integer(psb_ipk_) :: i,j,k, ir, jc, m4, ir1, ir2,jump
    real(psb_dpk_)   :: acc(4) 
    
    jump = a%hack*(bl-1)

    do j=1,nc
       if (off(j) > 0) then 

          ir1 = 1
          ir2 = n - off(j) - jump

          if(ir2 > a%hack) then
             ir2 = a%hack
          endif
       else
          ir1 = 1 - off(j) - jump
          ir2 = a%hack
                
          if(ir2+jump>m) then
             ir2 = m - jump
          endif
          
          if(ir1<=0)then
             ir1=1
          endif

       end if

       do i=ir1,ir2
             y(i+jump) = y(i+jump) + alpha*data(i,j)*x((i+jump)+off(j))
       enddo

    enddo
    
  end subroutine psb_d_hdia_csmv_inner

end subroutine psb_d_hdia_csmv
