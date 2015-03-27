
!!$  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
!!$  POSSIBILITY OF SUCH DAMAGE.
!!$ 
  

subroutine psi_s_convert_dia_from_coo(a,tmp,info)
  use psb_base_mod
  use psb_s_dia_mat_mod, psb_protect_name => psi_s_convert_dia_from_coo
  use psi_ext_util_mod
  implicit none 
  class(psb_s_dia_sparse_mat), intent(inout) :: a
  class(psb_s_coo_sparse_mat), intent(in)    :: tmp
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
  a%psb_s_base_sparse_mat = tmp%psb_s_base_sparse_mat
  
  ndiag = nr+nc-1
  allocate(d(ndiag),stat=info)
  if (info /= 0) return
  call psb_realloc(ndiag,a%offset,info)
  if (info /= 0) return
  
  call psi_dia_offset_from_coo(nr,nc,nza,tmp%ia,tmp%ja, &
       & nrd,nd,d,a%offset,info,initd=.true.,cleard=.false.) 
  
  call psb_realloc(nd,a%offset,info)
  if (info /= 0) return
  call psb_realloc(nrd,nd,a%data,info) 
  if (info /= 0) return
  a%nzeros = nza
  
  call psi_xtr_dia_from_coo(nr,nza,tmp%ia,tmp%ja,tmp%val,&
       & d,nrd,nd,a%data,info,initdata=.true.)
  
  deallocate(d,stat=info)
  if (info /= 0) return
  
end subroutine psi_s_convert_dia_from_coo
