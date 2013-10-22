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
  

subroutine psb_s_hll_csmv(alpha,a,x,beta,y,info,trans) 
  
  use psb_base_mod
  use psb_s_hll_mat_mod, psb_protect_name => psb_s_hll_csmv
  implicit none 
  class(psb_s_hll_sparse_mat), intent(in) :: a
  real(psb_spk_), intent(in)             :: alpha, beta, x(:)
  real(psb_spk_), intent(inout)          :: y(:)
  integer(psb_ipk_), intent(out)          :: info
  character, optional, intent(in)         :: trans

  character          :: trans_
  integer(psb_ipk_)  :: i,j,k,m,n, nnz, ir, jc, ic, hksz, hk, mxrwl, noffs, kc
  real(psb_spk_)    :: acc
  logical            :: tra, ctra
  Integer(Psb_ipk_)  :: err_act
  character(len=20)  :: name='s_hll_csmv'
  logical, parameter :: debug=.false.

  call psb_erractionsave(err_act)
  info = psb_success_

  if (present(trans)) then
    trans_ = trans
  else
    trans_ = 'N'
  end if

  if (.not.a%is_asb()) then 
    info = psb_err_invalid_mat_state_
    call psb_errpush(info,name)
    goto 9999
  endif

  tra  = (psb_toupper(trans_) == 'T')
  ctra = (psb_toupper(trans_) == 'C')

  if (tra.or.ctra) then 

    m = a%get_ncols()
    n = a%get_nrows()
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

    if (beta == szero) then
      do i = 1, m
        y(i) = szero
      enddo
    else
      do  i = 1, m
        y(i) = beta*y(i)
      end do
    endif

    hksz = a%get_hksz()
    j=1
    do i=1,n,hksz
      ir    = min(hksz,n-i+1) 
      mxrwl = (a%hkoffs(j+1) - a%hkoffs(j))/hksz
      k     = a%hkoffs(j) + 1
      call psb_s_hll_csmv_inner(i,ir,mxrwl,a%irn(i),&
           & alpha,a%ja(k),hksz,a%val(k),hksz,&
           & a%is_triangle(),a%is_unit(),&
           & x,sone,y,tra,ctra,info) 
      if (info /= psb_success_) goto 9999
      j = j + 1 
    end do


  else if (.not.(tra.or.ctra)) then 

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


    hksz = a%get_hksz()
    j=1
    do i=1,m,hksz
      ir    = min(hksz,m-i+1) 
      mxrwl = (a%hkoffs(j+1) - a%hkoffs(j))/hksz
      k     = a%hkoffs(j) + 1
      call psb_s_hll_csmv_inner(i,ir,mxrwl,a%irn(i),&
           & alpha,a%ja(k),hksz,a%val(k),hksz,&
           & a%is_triangle(),a%is_unit(),&
           & x,beta,y,tra,ctra,info) 
      if (info /= psb_success_) goto 9999
      j = j + 1 
    end do

  end if

  call psb_erractionrestore(err_act)
  return


9999 continue
  call psb_erractionrestore(err_act)

  if (err_act == psb_act_abort_) then
    call psb_error()
    return
  end if
  return

contains

  subroutine psb_s_hll_csmv_inner(ir,m,n,irn,alpha,ja,ldj,val,ldv,&
       & is_triangle,is_unit, x,beta,y,tra,ctra,info) 
    integer(psb_ipk_), intent(in)    :: ir,m,n,ldj,ldv,ja(ldj,*),irn(*)
    real(psb_spk_), intent(in)      :: alpha, beta, x(*),val(ldv,*)
    real(psb_spk_), intent(inout)   :: y(*)
    logical, intent(in)              :: is_triangle,is_unit,tra,ctra
    integer(psb_ipk_), intent(out)   :: info

    integer(psb_ipk_) :: i,j,k, m4, jc
    real(psb_spk_)   :: acc(4), tmp

    info = psb_success_
    if (tra) then 

      if (beta == sone) then 
        do i=1,m
          do j=1, irn(i)
            jc = ja(i,j)
            y(jc) = y(jc) + alpha*val(i,j)*x(ir+i-1)
          end do
        end do
      else
        info = -10

      end if

    else if (ctra) then 

      if (beta == sone) then 
        do i=1,m
          do j=1, irn(i)
            jc = ja(i,j)
            y(jc) = y(jc) + alpha*(val(i,j))*x(ir+i-1)
          end do
        end do
      else
        info = -10

      end if

    else if (.not.(tra.or.ctra)) then 

      if (alpha == szero) then 
        if (beta == szero) then 
          do i=1,m
            y(ir+i-1) = szero
          end do
        else
          do i=1,m
            y(ir+i-1) =  beta*y(ir+i-1) 
          end do
        end if

      else
        if (beta == szero) then 
          do i=1,m
            tmp = szero
            do j=1, irn(i)
              tmp = tmp + val(i,j)*x(ja(i,j))
            end do
            y(ir+i-1) = alpha*tmp 
          end do
        else
          do i=1,m
            tmp = szero
            do j=1, irn(i)
              tmp = tmp + val(i,j)*x(ja(i,j))
            end do
            y(ir+i-1) = alpha*tmp + beta*y(ir+i-1)
          end do
        endif
      end if
    end if

    if (is_unit) then 
      do i=1, min(m,n)
        y(i) = y(i) + alpha*x(i)
      end do
    end if

  end subroutine psb_s_hll_csmv_inner
end subroutine psb_s_hll_csmv
