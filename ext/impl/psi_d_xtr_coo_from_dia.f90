subroutine psi_d_xtr_coo_from_dia(nr,nc,ia,ja,val,nz,nrd,ncd,data,offsets,info,rdisp)
  use psb_base_mod, only : psb_ipk_, psb_success_, psb_dpk_, dzero

  implicit none 

  integer(psb_ipk_), intent(in)    :: nr,nc, nrd,ncd, offsets(:) 
  integer(psb_ipk_), intent(inout) :: ia(:), ja(:),nz
  real(psb_dpk_),    intent(inout) :: val(:)
  real(psb_dpk_),    intent(in)    :: data(nrd,ncd)
  integer(psb_ipk_), intent(out)   :: info
  integer(psb_ipk_), intent(in), optional :: rdisp

  !locals
  integer(psb_ipk_) :: rdisp_, nrcmdisp, rdisp1
  integer(psb_ipk_) :: i,j,ir1, ir2, ir, ic,k
  logical, parameter :: debug=.false.

  info = psb_success_
  rdisp_ = 0
  if (present(rdisp)) rdisp_ = rdisp

  if (debug) write(0,*) 'Start xtr_coo_from_dia',nr,nc,nrd,ncd, rdisp_
  nrcmdisp = min(nr-rdisp_,nc-rdisp_) 
  rdisp1   = 1-rdisp_
  nz = 0 
  do j=1, ncd
    if (offsets(j)>=0) then 
      ir1 = 1
      ! ir2 = min(nrd,nr - offsets(j) - rdisp_,nc-offsets(j)-rdisp_)
      ir2 = min(nrd, nrcmdisp - offsets(j))
    else
      ! ir1 = max(1,1-offsets(j)-rdisp_) 
      ir1 = max(1, rdisp1 - offsets(j))
      ir2 = min(nrd, nrcmdisp) 
    end if
    if (debug) write(0,*) ' Loop  J',j,ir1,ir2, offsets(j)      
    do i=ir1,ir2 
      ir = i + rdisp_
      ic = i + rdisp_ + offsets(j)
      if (debug) write(0,*) ' Loop  I',i,ir,ic
      nz = nz + 1
      ia(nz) = ir
      ja(nz) = ic 
      val(nz) = data(i,j)
    enddo
  end do

end subroutine psi_d_xtr_coo_from_dia

