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
  

subroutine psb_d_ell_csmv(alpha,a,x,beta,y,info,trans) 
  
  use psb_base_mod
  use psb_d_ell_mat_mod, psb_protect_name => psb_d_ell_csmv
  implicit none 
  class(psb_d_ell_sparse_mat), intent(in) :: a
  real(psb_dpk_), intent(in)        :: alpha, beta, x(:)
  real(psb_dpk_), intent(inout)     :: y(:)
  integer(psb_ipk_), intent(out)     :: info
  character, optional, intent(in)    :: trans

  character :: trans_
  integer(psb_ipk_)  :: i,j,k,m,n, nnz, ir, jc
  real(psb_dpk_)    :: acc
  logical            :: tra, ctra
  Integer(Psb_ipk_)  :: err_act
  character(len=20)  :: name='d_ell_csmv'
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
  else
    n = a%get_ncols()
    m = a%get_nrows()
  end if

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


  call psb_d_ell_csmv_inner(m,n,alpha,size(a%ja,2),&
       & a%ja,size(a%ja,1),a%val,size(a%val,1),&
       & a%is_triangle(),a%is_unit(),&
       & x,beta,y,tra,ctra) 

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

  subroutine psb_d_ell_csmv_inner(m,n,alpha,nc,ja,ldj,val,ldv,&
       & is_triangle,is_unit, x,beta,y,tra,ctra) 
    integer(psb_ipk_), intent(in)   :: m,n,nc,ldj,ldv,ja(ldj,*)
    real(psb_dpk_), intent(in)     :: alpha, beta, x(*),val(ldv,*)
    real(psb_dpk_), intent(inout)  :: y(*)
    logical, intent(in)             :: is_triangle,is_unit,tra,ctra


    integer(psb_ipk_) :: i,j,k, ir, jc, m4
    real(psb_dpk_)   :: acc(4) 


    if (alpha == dzero) then
      if (beta == dzero) then
        do i = 1, m
          y(i) = dzero
        enddo
      else
        do  i = 1, m
          y(i) = beta*y(i)
        end do
      endif
      return
    end if


    if (.not.(tra.or.ctra)) then 

      if (beta == dzero) then 

        m4 = mod(m,4) 
        do i=1,m4
          acc(1) = dzero 
          do j=1,nc
            acc(1)  = acc(1) + val(i,j) * x(ja(i,j))          
          enddo
          y(i) = alpha*acc(1)
        end do


        if (alpha == done) then 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4)  = acc(1:4) + val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = acc(1:4)
          end do

        else if (alpha == -done) then 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4) = acc(1:4) - val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = acc(1:4)
          end do

        else 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4)  = acc(1:4) + val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = alpha * acc(1:4)
          end do

        end if


      else if (beta == done) then 
        

        m4 = mod(m,4) 
        do i=1,m4
          acc(1) = dzero 
          do j=1,nc
            acc(1)  = acc(1) + val(i,j) * x(ja(i,j))          
          enddo
          y(i) = y(i) + alpha*acc(1)
        end do

        if (alpha == done) then 
          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4) = acc(1:4) + val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = y(i:i+3) + acc(1:4)
          end do

        else if (alpha == -done) then 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4) = acc(1:4) - val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = y(i:i+3) + acc(1:4)
          end do

        else 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4) = acc(1:4) + val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = y(i:i+3) + alpha*acc(1:4)
          end do

        end if

      else if (beta == -done) then 

        m4 = mod(m,4) 
        do i=1,m4
          acc(1) = dzero 
          do j=1,nc
            acc(1)  = acc(1) + val(i,j) * x(ja(i,j))          
          enddo
          y(i) = - y(i) + alpha*acc(1)
        end do

        if (alpha == done) then 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4) = acc(1:4) + val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = -y(i:i+3) + acc(1:4)
          end do

        else if (alpha == -done) then 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4) = acc(1:4) - val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = -y(i:i+3) + acc(1:4)
          end do

        else 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4) = acc(1:4) + val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = -y(i:i+3) + alpha*acc(1:4)
          end do

        end if

      else 

        m4 = mod(m,4) 
        do i=1,m4
          acc(1) = dzero 
          do j=1,nc
            acc(1)  = acc(1) + val(i,j) * x(ja(i,j))          
          enddo
          y(i) = beta*y(i) + alpha*acc(1)
        end do

        if (alpha == done) then 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4) = acc(1:4) + val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = beta*y(i:i+3) + acc(1:4)
          end do

        else if (alpha == -done) then 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4) = acc(1:4) - val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = beta*y(i:i+3) + acc(1:4)
          end do

        else 

          do i=m4+1,m,4
            acc  = dzero
            do j=1,nc
              acc(1:4) = acc(1:4) + val(i:i+3,j) * x(ja(i:i+3,j))          
            enddo
            y(i:i+3) = beta*y(i:i+3) + alpha*acc(1:4)
          end do

        end if

      end if

    else if (tra) then 

      if (beta == dzero) then 
        do i=1, m
          y(i) = dzero
        end do
      else if (beta == done) then 
        ! Do nothing
      else if (beta == -done) then 
        do i=1, m
          y(i) = -y(i) 
        end do
      else
        do i=1, m
          y(i) = beta*y(i) 
        end do
      end if

      !
      ! Need to think about this.
      ! Transpose does not mix well with ELLPACK.
      ! 
      if (alpha == done) then

        do i=1,n
          do j=1,nc
            ir = ja(i,j)
            y(ir) = y(ir) +  val(i,j)*x(i)
          end do
        enddo

      else if (alpha == -done) then

        do i=1,n
          do j=1,nc
            ir = ja(i,j)
            y(ir) = y(ir) -  val(i,j)*x(i)
          end do
        enddo

      else                    

        do i=1,n
          do j=1,nc
            ir = ja(i,j)
            y(ir) = y(ir) +  alpha*val(i,j)*x(i)
          end do
        enddo

      end if

    else if (ctra) then 

      if (beta == dzero) then 
        do i=1, m
          y(i) = dzero
        end do
      else if (beta == done) then 
        ! Do nothing
      else if (beta == -done) then 
        do i=1, m
          y(i) = -y(i) 
        end do
      else
        do i=1, m
          y(i) = beta*y(i) 
        end do
      end if

      !
      ! Need to think about this.
      ! Transpose does not mix well with ELLPACK.
      ! 
      if (alpha == done) then

        do i=1,n
          do j=1,nc
            ir = ja(i,j)
            y(ir) = y(ir) +  (val(i,j))*x(i)
          end do
        enddo

      else if (alpha == -done) then

        do i=1,n
          do j=1,nc
            ir = ja(i,j)
            y(ir) = y(ir) -  (val(i,j))*x(i)
          end do
        enddo

      else                    

        do i=1,n
          do j=1,nc
            ir = ja(i,j)
            y(ir) = y(ir) +  alpha*(val(i,j))*x(i)
          end do
        enddo

      end if

    endif

    if (is_unit) then 
      do i=1, min(m,n)
        y(i) = y(i) + alpha*x(i)
      end do
    end if


  end subroutine psb_d_ell_csmv_inner

end subroutine psb_d_ell_csmv
