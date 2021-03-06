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
  

module psb_ext_mod
  use psb_const_mod
  use psi_ext_util_mod

  use psb_s_dns_mat_mod
  use psb_d_dns_mat_mod
  use psb_c_dns_mat_mod
  use psb_z_dns_mat_mod
  
  use psb_d_ell_mat_mod
  use psb_s_ell_mat_mod
  use psb_z_ell_mat_mod
  use psb_c_ell_mat_mod

  use psb_s_hll_mat_mod
  use psb_d_hll_mat_mod
  use psb_c_hll_mat_mod
  use psb_z_hll_mat_mod
  
  use psb_s_dia_mat_mod
  use psb_d_dia_mat_mod
  use psb_c_dia_mat_mod
  use psb_z_dia_mat_mod

  use psb_s_hdia_mat_mod
  use psb_d_hdia_mat_mod
  use psb_c_hdia_mat_mod
  use psb_z_hdia_mat_mod

#ifdef HAVE_RSB
  use psb_d_rsb_mat_mod
#endif
end module psb_ext_mod
