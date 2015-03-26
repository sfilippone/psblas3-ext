
!!$  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
!!$  POSSIBILITY OF SUCH DAMAGE.
!!$ 
  

subroutine psi_d_convert_dia_from_coo(a,tmp,info)
  use psb_base_mod
  use psb_d_dia_mat_mod, psb_protect_name => psi_d_convert_dia_from_coo
  use psi_ext_util_mod
  implicit none 
  class(psb_d_dia_sparse_mat), intent(inout) :: a
  class(psb_d_coo_sparse_mat), intent(in)    :: tmp
  integer(psb_ipk_), intent(out)             :: info
  
  !locals
  integer(psb_ipk_)              :: ndiag,nd
  integer(psb_ipk_),allocatable  :: d(:)
  integer(psb_ipk_)              :: k,i,j,nc,nr,nza,nrd,ir,ic
  
  info = psb_success_
  nr  = tmp%get_nrows()
  nc  = tmp%get_ncols()
  nza = tmp%get_nzeros()
  ! If it is sorted then we can lessen memory impact 
  a%psb_d_base_sparse_mat = tmp%psb_d_base_sparse_mat
  
  ndiag = nr+nc-1
  allocate(d(ndiag),stat=info)
  if (info /= 0) return
  call psb_realloc(ndiag,a%offset,info)
  if (info /= 0) return
  
  call psi_diacnt_from_coo(nr,nc,nza,tmp%ia,tmp%ja,nrd,nd,d,info,initd=.true.)
  call psi_offset_from_d(nr,nc,d,a%offset,info)
  
  call psb_realloc(nd,a%offset,info)
  if (info /= 0) return
  call psb_realloc(nrd,nd,a%data,info) 
  if (info /= 0) return
  a%nzeros = nza
  
  call psi_xtr_dia_from_coo(nr,nza,tmp%ia,tmp%ja,tmp%val,d,a%data,info,initd=.true.)
  
  do i=1,nd
    k=a%offset(i)+nr
    d(k) = 0
  end do
  if (any(d(1:ndiag)/=0)) then
    write(*,*) 'Check from dia_from_coo: some entries in D not zeroed on exitr'
  end if
  deallocate(d,stat=info)
  if (info /= 0) return
  
end subroutine psi_d_convert_dia_from_coo
