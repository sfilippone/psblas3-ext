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
  

subroutine psb_s_hll_cssv(alpha,a,x,beta,y,info,trans) 
  
  use psb_base_mod
  use psb_s_hll_mat_mod, psb_protect_name => psb_s_hll_cssv
  implicit none 
  class(psb_s_hll_sparse_mat), intent(in) :: a
  real(psb_spk_), intent(in)              :: alpha, beta, x(:)
  real(psb_spk_), intent(inout)           :: y(:)
  integer(psb_ipk_), intent(out)          :: info
  character, optional, intent(in)         :: trans

  character           :: trans_
  integer(psb_ipk_)   :: i,j,k,m,n, nnz, ir, jc, ic, hksz, hk, mxrwl, noffs, kc, mk
  real(psb_spk_)      :: acc
  real(psb_spk_), allocatable :: tmp(:)
  logical            :: tra, ctra
  Integer(Psb_ipk_)  :: err_act
  character(len=20)  :: name='s_hll_cssv'
  logical, parameter :: debug=.false.

  info = psb_success_
  call psb_erractionsave(err_act)
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

  if (a%is_dev()) call a%sync()
  tra  = (psb_toupper(trans_) == 'T')
  ctra = (psb_toupper(trans_) == 'C')
  m = a%get_nrows()

  if (.not. (a%is_triangle().and.a%is_sorted())) then 
    info = psb_err_invalid_mat_state_
    call psb_errpush(info,name)
    goto 9999
  end if

  if (size(x)<m) then 
    info = 36
    call psb_errpush(info,name,i_err=(/3*ione,m/))
    goto 9999
  end if

  if (size(y)<m) then 
    info = 36
    call psb_errpush(info,name,i_err=(/5*ione,m/))
    goto 9999
  end if

  if (alpha == szero) then
    if (beta == szero) then
      do i = 1, m
        y(i) = szero
      enddo
    else
      do  i = 1, m
        y(i) = beta*y(i)
      end do
    endif
    return
  end if

  hksz = a%get_hksz()

  if (beta == szero) then 

    if (.not.(tra.or.ctra)) then 

      if (a%is_lower()) then 
        do i=1,m,hksz
          ir   = min(hksz,m-i+1) 
          j     = (i-1)/hksz + 1
          mxrwl = (a%hkoffs(j+1) - a%hkoffs(j))/hksz
          k     = a%hkoffs(j) + 1
          call inner_hllsv(tra,ctra,a%is_lower(),a%is_unit(),&
               & i,ir,mxrwl,a%irn(i),a%idiag(i),&
               & a%ja(k),hksz,a%val(k),hksz,&
               & x,y,info) 
          if (info /= 0) goto 9999
        end do
      else

        k = mod(m,hksz)
        if (k==0) k=hksz
        do i=m-k+1,1,-hksz
          ir   = min(hksz,m-i+1) 
          j     = (i-1)/hksz + 1
          mxrwl = (a%hkoffs(j+1) - a%hkoffs(j))/hksz
          k     = a%hkoffs(j) + 1
          call inner_hllsv(tra,ctra,a%is_lower(),a%is_unit(),&
               & i,ir,mxrwl,a%irn(i),a%idiag(i),&
               & a%ja(k),hksz,a%val(k),hksz,&
               & x,y,info) 
          if (info /= 0) goto 9999
        end do
      end if

    else if (tra.or.ctra) then 

      do i=1, m
        y(i) = x(i)
      end do


      if (a%is_lower()) then 

        mk = mod(m,hksz)
        if (k==0) k=hksz
        do i=m-mk+1,1,-hksz
          ir   = min(hksz,m-i+1) 
          j     = (i-1)/hksz + 1
          mxrwl = (a%hkoffs(j+1) - a%hkoffs(j))/hksz
          k     = a%hkoffs(j) + 1
          call inner_hllsv(tra,ctra,a%is_lower(),a%is_unit(),&
               & i,ir,mxrwl,a%irn(i),a%idiag(i),&
               & a%ja(k),hksz,a%val(k),hksz,&
               & x,y,info) 
          if (info /= 0) goto 9999
        end do

      else

        do i=1,m,hksz
          ir   = min(hksz,m-i+1) 
          j     = (i-1)/hksz + 1
          mxrwl = (a%hkoffs(j+1) - a%hkoffs(j))/hksz
          k     = a%hkoffs(j) + 1
          call inner_hllsv(tra,ctra,a%is_lower(),a%is_unit(),&
               & i,ir,mxrwl,a%irn(i),a%idiag(i),&
               & a%ja(k),hksz,a%val(k),hksz,&
               & x,y,info) 
          if (info /= 0) goto 9999
        end do

      end if

    end if

    if (info /= 0) then 
      info = psb_err_invalid_mat_state_
      call psb_errpush(info,name)
      goto 9999
    endif

    if (alpha == sone) then 
      ! do nothing
    else if (alpha == -sone) then 
      do  i = 1, m
        y(i) = -y(i)
      end do
    else
      do  i = 1, m
        y(i) = alpha*y(i)
      end do
    end if


  else 

    allocate(tmp(m), stat=info) 
    if (info /= psb_success_) then 
      info = psb_err_alloc_dealloc_
      call psb_errpush(info,name)
      goto 9999
    endif

    if (.not.(tra.or.ctra)) then 

      if (a%is_lower()) then 
        do i=1,m,hksz
          ir   = min(hksz,m-i+1) 
          j     = (i-1)/hksz + 1
          mxrwl = (a%hkoffs(j+1) - a%hkoffs(j))/hksz
          k     = a%hkoffs(j) + 1
          call inner_hllsv(tra,ctra,a%is_lower(),a%is_unit(),&
               & i,ir,mxrwl,a%irn(i),a%idiag(i),&
               & a%ja(k),hksz,a%val(k),hksz,&
               & x,tmp,info) 
          if (info /= 0) goto 9999
        end do
      else

        mk = mod(m,hksz)
        if (k==0) k=hksz
        do i=m-mk+1,1,-hksz
          ir   = min(hksz,m-i+1) 
          j     = (i-1)/hksz + 1
          mxrwl = (a%hkoffs(j+1) - a%hkoffs(j))/hksz
          k     = a%hkoffs(j) + 1
          call inner_hllsv(tra,ctra,a%is_lower(),a%is_unit(),&
               & i,ir,mxrwl,a%irn(i),a%idiag(i),&
               & a%ja(k),hksz,a%val(k),hksz,&
               & x,tmp,info) 
          if (info /= 0) goto 9999
        end do
      end if

    else if (tra.or.ctra) then 

      do i=1, m
        tmp(i) = x(i)
      end do

      if (a%is_lower()) then 

        mk = mod(m,hksz)
        if (k==0) k=hksz
        do i=m-mk+1,1,-hksz
          ir   = min(hksz,m-i+1) 
          j     = (i-1)/hksz + 1
          mxrwl = (a%hkoffs(j+1) - a%hkoffs(j))/hksz
          k     = a%hkoffs(j) + 1
          call inner_hllsv(tra,ctra,a%is_lower(),a%is_unit(),&
               & i,ir,mxrwl,a%irn(i),a%idiag(i),&
               & a%ja(k),hksz,a%val(k),hksz,&
               & x,tmp,info) 
          if (info /= 0) goto 9999
        end do

      else

        do i=1,m,hksz
          ir   = min(hksz,m-i+1) 
          j     = (i-1)/hksz + 1
          mxrwl = (a%hkoffs(j+1) - a%hkoffs(j))/hksz
          k     = a%hkoffs(j) + 1
          call inner_hllsv(tra,ctra,a%is_lower(),a%is_unit(),&
               & i,ir,mxrwl,a%irn(i),a%idiag(i),&
               & a%ja(k),hksz,a%val(k),hksz,&
               & x,tmp,info) 
          if (info /= 0) goto 9999
        end do

      end if
    end if

    if (info == 0) & 
         & call psb_geaxpby(m,alpha,tmp,beta,y,info)

    if (info /= 0) then 
      info = psb_err_invalid_mat_state_
      call psb_errpush(info,name)
      goto 9999
    endif

  end if

  call psb_erractionrestore(err_act)
  return

9999 call psb_error_handler(err_act)
  return

contains 

  subroutine inner_hllsv(tra,ctra,lower,unit,ie,n,nc,irn,idiag,ja,ldj,val,ldv,x,y,info) 
    implicit none 
    logical, intent(in)             :: tra,ctra,lower,unit
    integer(psb_ipk_), intent(in)   :: ie,n,nc,ldj,ldv
    integer(psb_ipk_), intent(in)   :: irn(*),idiag(*), ja(ldj,*)
    real(psb_spk_), intent(in)      :: val(ldv,*)
    real(psb_spk_), intent(in)      :: x(*)
    real(psb_spk_), intent(out)     :: y(*)
    integer(psb_ipk_), intent(out)  :: info

    integer(psb_ipk_) :: i,j,k,m, ir, jc
    real(psb_spk_)    :: acc

    !
    ! The only error condition here is if
    ! the matrix is non-unit and some idiag value is illegal.
    !
    info = 0 
    if (.not.(tra.or.ctra)) then 

      if (lower) then 

        if (unit) then 
          do i=1,n
            acc = szero 
            do j=1,irn(i)
              acc = acc + val(i,j)*y(ja(i,j))
            end do
            y(ie+i-1) = x(ie+i-1) - acc
          end do
        else if (.not.unit) then 
          do i=1, n
            acc = szero 
            do j=1,idiag(i)-1
              acc = acc + val(i,j)*y(ja(i,j))
            end do
            if (idiag(i) <= 0) then 
              info = -1 
              return
            endif
            y(ie+i-1) = (x(ie+i-1) - acc)/val(i,idiag(i))
          end do
        end if

      else if (.not.lower) then 

        if (unit) then 

          do i=n, 1, -1 
            acc = szero 
            do j=1,irn(i)
              acc = acc + val(i,j)*y(ja(i,j))
            end do
            y(ie+i-1) = x(ie+i-1) - acc
          end do

        else if (.not.unit) then 

          do i=n, 1, -1 
            acc = szero 
            do j=idiag(i)+1, irn(i)
              acc = acc + val(i,j)*y(ja(i,j))
            end do
            if (idiag(i) <= 0) then 
              info = -1 
              return
            endif
            y(ie+i-1) = (x(ie+i-1) - acc)/val(i,idiag(i))
          end do

        end if

      end if

    else if (tra) then 

      if (lower) then 

        if (unit) then 

          do i=n, 1, -1
            acc = y(ie+i-1) 

            do j=1,irn(i)
              jc    = ja(i,j)
              y(jc) = y(jc) - val(i,j)*acc 
            end do

          end do

        else if (.not.unit) then 

          do i=n, 1, -1
            if (idiag(i) <= 0) then 
              info = -1 
              return
            endif
            y(ie+i-1) = y(ie+i-1)/val(i,idiag(i))
            acc  = y(ie+i-1) 
            do j=1,idiag(i) -1
              jc    = ja(i,j)
              y(jc) = y(jc) - val(i,j)*acc 
            end do
          end do

        end if

      else if (.not.lower) then 

        if (unit) then 

          do i=1, n
            acc  = y(ie+i-1) 
            do j=1, irn(i)
              jc    = ja(i,j)
              y(jc) = y(jc) - val(i,j)*acc 
            end do
          end do

        else if (.not.unit) then 

          do i=1, n
            if (idiag(i) <= 0) then 
              info = -1 
              return
            endif
            y(ie+i-1) = y(ie+i-1)/val(i,idiag(i))
            acc  = y(ie+i-1) 
            do j=idiag(i)+1, irn(i) 
              jc    = ja(i,j)
              y(jc) = y(jc) - val(i,j)*acc 
            end do
          end do

        end if

      end if

    else if (ctra) then 
          
      if (lower) then 

        if (unit) then 

          do i=n, 1, -1
            acc = y(ie+i-1) 

            do j=1,irn(i)
              jc    = ja(i,j)
              y(jc) = y(jc) - (val(i,j))*acc 
            end do

          end do

        else if (.not.unit) then 

          do i=n, 1, -1
            if (idiag(i) <= 0) then 
              info = -1 
              return
            endif
            y(ie+i-1) = y(ie+i-1)/(val(i,idiag(i)))
            acc  = y(ie+i-1) 
            do j=1,idiag(i) -1
              jc    = ja(i,j)
              y(jc) = y(jc) - (val(i,j))*acc 
            end do
          end do

        end if

      else if (.not.lower) then 

        if (unit) then 

          do i=1, n
            acc  = y(ie+i-1) 
            do j=1, irn(i)
              jc    = ja(i,j)
              y(jc) = y(jc) - (val(i,j))*acc 
            end do
          end do

        else if (.not.unit) then 

          do i=1, n
            if (idiag(i) <= 0) then 
              info = -1 
              return
            endif
            y(ie+i-1) = y(ie+i-1)/(val(i,idiag(i)))
            acc  = y(ie+i-1) 
            do j=idiag(i)+1, irn(i) 
              jc    = ja(i,j)
              y(jc) = y(jc) - (val(i,j))*acc 
            end do
          end do

        end if

      end if
    end if
  end subroutine inner_hllsv
end subroutine psb_s_hll_cssv
