module pde2d_base_mod
  use psb_base_mod, only : psb_dpk_
  real(psb_dpk_), save, private :: epsilon=1.d0/80
contains
  subroutine pde2d_set_parm(dat)
    real(psb_dpk_), intent(in) :: dat
    epsilon = dat
  end subroutine pde2d_set_parm
  !
  ! functions parametrizing the differential equation 
  !  
  function b1(x,y)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) :: b1
    real(psb_dpk_), intent(in) :: x,y
    b1 = 1.d0/1.414d0
  end function b1
  function b2(x,y)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) ::  b2
    real(psb_dpk_), intent(in) :: x,y
    b2 = 1.d0/1.414d0
  end function b2
  function c(x,y)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) ::  c
    real(psb_dpk_), intent(in) :: x,y      
    c = 0.d0 
  end function c
  function a1(x,y)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) ::  a1   
    real(psb_dpk_), intent(in) :: x,y
    a1=1.d0*epsilon
  end function a1
  function a2(x,y)
    use psb_base_mod, only : psb_dpk_
    real(psb_dpk_) ::  a2
    real(psb_dpk_), intent(in) :: x,y
    a2=1.d0*epsilon
  end function a2
  function g(x,y)
    use psb_base_mod, only : psb_dpk_, done, dzero
    real(psb_dpk_) ::  g
    real(psb_dpk_), intent(in) :: x,y
    g = dzero
    if (x == done) then
      g = done
    else if (x == dzero) then 
      g = done
    end if
  end function g
end module pde2d_base_mod

