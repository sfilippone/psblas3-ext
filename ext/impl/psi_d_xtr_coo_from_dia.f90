subroutine psi_d_xtr_coo_from_dia(nr,ia,ja,val,nrd,ncd,data,offsets,info,rdisp)    
  use psb_base_mod, only : psb_ipk_, psb_success_, psb_dpk_, dzero

  implicit none 

  integer(psb_ipk_), intent(in)    :: nr, nrd,ncd, offsets(:) 
  integer(psb_ipk_), intent(inout) :: ia(:), ja(:)
  real(psb_dpk_),    intent(inout) :: val(:)
  real(psb_dpk_),    intent(in)    :: data(nrd,ncd)
  integer(psb_ipk_), intent(out)   :: info
  integer(psb_ipk_), intent(in), optional :: rdisp

  !locals
  integer(psb_ipk_) :: rdisp_
  integer(psb_ipk_) :: i,j,ir1, ir2, ir, ic,k
  logical, parameter :: debug=.false.

  info = psb_success_
  rdisp_ = 0
  if (present(rdisp)) rdisp_ = rdisp

  if (debug) write(0,*) 'Start xtr_coo_from_dia',nr,nrd,ncd, rdisp_

  k = 1
  do j=1, ncd
    if (offsets(j)>=0) then 
      ir1 = 1
      ir2 = nr - offsets(j)
    else
      ir1 = 1 - offsets(j)
      ir2 = nr 
    end if
    do i=ir1,ir2 
      ir = i + rdisp_
      ic = i + rdisp_ + offsets(j)
      ia(k) = ir
      ja(k) = ic 
      val(k) = data(i,j)
      k = k + 1 
    enddo
  end do

end subroutine psi_d_xtr_coo_from_dia

