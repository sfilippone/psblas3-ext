  /*             Parallel Sparse BLAS   GPU plugin  */
  /*   (C) Copyright 2013 */

  /*                      Salvatore Filippone    University of Rome Tor Vergata */
  /*                      Alessandro Fanfarillo  University of Rome Tor Vergata */
 
  /* Redistribution and use in source and binary forms, with or without */
  /* modification, are permitted provided that the following conditions */
  /* are met: */
  /*   1. Redistributions of source code must retain the above copyright */
  /*      notice, this list of conditions and the following disclaimer. */
  /*   2. Redistributions in binary form must reproduce the above copyright */
  /*      notice, this list of conditions, and the following disclaimer in the */
  /*      documentation and/or other materials provided with the distribution. */
  /*   3. The name of the PSBLAS group or the names of its contributors may */
  /*      not be used to endorse or promote products derived from this */
  /*      software without specific written permission. */
 
  /* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
  /* ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED */
  /* TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR */
  /* PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE PSBLAS GROUP OR ITS CONTRIBUTORS */
  /* BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR */
  /* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF */
  /* SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS */
  /* INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN */
  /* CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) */
  /* ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE */
  /* POSSIBILITY OF SUCH DAMAGE. */
 
  

#include <stdio.h>
#include <stdlib.h>

#ifdef HAVE_SPGPU
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cintrf.h"
#include "fcusparse.h"

static   cusparseHandle_t *cusparse_handle=NULL;


void setHandle(cusparseHandle_t);

int FcusparseCreate() 
{
  int ret=CUSPARSE_STATUS_SUCCESS;
  cusparseHandle_t *handle;
  if (cusparse_handle == NULL) {
    if ((handle = (cusparseHandle_t *)malloc(sizeof(cusparseHandle_t)))==NULL) 
      return((int) CUSPARSE_STATUS_ALLOC_FAILED);
    ret = (int)cusparseCreate(handle);
    if (ret == CUSPARSE_STATUS_SUCCESS)
      cusparse_handle = handle;
  }
  return (ret);
}

int FcusparseDestroy() 
{
  int val;
  val = (int) cusparseDestroy(*cusparse_handle);
  free(cusparse_handle);
  cusparse_handle=NULL;
  return(val);
}
cusparseHandle_t *getHandle()
{
  if (cusparse_handle == NULL)
    FcusparseCreate();
  return(cusparse_handle);
}



/*    Single precision real   */ 
#define TYPE 			       float                             
#define T_CSRGDeviceMat		       s_CSRGDeviceMat
#define T_Cmat			       s_Cmat
#define T_HYBGDeviceMat		       s_HYBGDeviceMat
#define T_Hmat			       s_Hmat
#define T_spmvCSRGDevice	       s_spmvCSRGDevice
#define T_spsvCSRGDevice	       s_spsvCSRGDevice
#define T_CSRGDeviceAlloc	       s_CSRGDeviceAlloc
#define T_CSRGDeviceGetParms	       s_CSRGDeviceGetParms
#define T_CSRGDeviceFree	       s_CSRGDeviceFree
#define T_CSRGDeviceSetMatType	       s_CSRGDeviceSetMatType
#define T_CSRGDeviceSetMatFillMode     s_CSRGDeviceSetMatFillMode
#define T_CSRGDeviceSetMatDiagType     s_CSRGDeviceSetMatDiagType
#define T_CSRGDeviceSetMatIndexBase    s_CSRGDeviceSetMatIndexBase
#define T_CSRGDeviceCsrsmAnalysis      s_CSRGDeviceCsrsmAnalysis
#define T_CSRGHost2Device	       s_CSRGHost2Device
#define T_CSRGDevice2Host	       s_CSRGDevice2Host
#define T_HYBGDeviceFree	       s_HYBGDeviceFree
#define T_spmvHYBGDevice	       s_spmvHYBGDevice
#define T_HYBGDeviceAlloc	       s_HYBGDeviceAlloc
#define T_HYBGDeviceSetMatDiagType     s_HYBGDeviceSetMatDiagType
#define T_HYBGDeviceSetMatIndexBase    s_HYBGDeviceSetMatIndexBase
#define T_HYBGDeviceSetMatType	       s_HYBGDeviceSetMatType
#define T_HYBGDeviceSetMatFillMode     s_HYBGDeviceSetMatFillMode
#define T_HYBGDeviceHybsmAnalysis      s_HYBGDeviceHybsmAnalysis
#define T_spsvHYBGDevice	       s_spsvHYBGDevice
#define T_HYBGHost2Device	       s_HYBGHost2Device
#define cusparseTcsrmv		       cusparseScsrmv
#define cusparseTcsrsv_solve	       cusparseScsrsv_solve
#define cusparseTcsrsv_analysis	       cusparseScsrsv_analysis
#define cusparseThybmv		       cusparseShybmv
#define cusparseThybsv_solve	       cusparseShybsv_solve
#define cusparseThybsv_analysis	       cusparseShybsv_analysis
#define cusparseTcsr2hyb               cusparseScsr2hyb               

#include "fcusparse_fct.h"



/*    Double precision real   */ 
#define TYPE 			       double
#define T_CSRGDeviceMat		       d_CSRGDeviceMat
#define T_Cmat			       d_Cmat
#define T_HYBGDeviceMat		       d_HYBGDeviceMat
#define T_Hmat			       d_Hmat
#define T_spmvCSRGDevice	       d_spmvCSRGDevice
#define T_spsvCSRGDevice	       d_spsvCSRGDevice
#define T_CSRGDeviceGetParms	       d_CSRGDeviceGetParms
#define T_CSRGDeviceAlloc	       d_CSRGDeviceAlloc
#define T_CSRGDeviceFree	       d_CSRGDeviceFree
#define T_CSRGDeviceSetMatType	       d_CSRGDeviceSetMatType
#define T_CSRGDeviceSetMatFillMode     d_CSRGDeviceSetMatFillMode
#define T_CSRGDeviceSetMatDiagType     d_CSRGDeviceSetMatDiagType
#define T_CSRGDeviceSetMatIndexBase    d_CSRGDeviceSetMatIndexBase
#define T_CSRGDeviceCsrsmAnalysis      d_CSRGDeviceCsrsmAnalysis
#define T_CSRGHost2Device	       d_CSRGHost2Device
#define T_CSRGDevice2Host	       d_CSRGDevice2Host
#define T_HYBGDeviceFree	       d_HYBGDeviceFree
#define T_spmvHYBGDevice	       d_spmvHYBGDevice
#define T_HYBGDeviceAlloc	       d_HYBGDeviceAlloc
#define T_HYBGDeviceSetMatDiagType     d_HYBGDeviceSetMatDiagType
#define T_HYBGDeviceSetMatIndexBase    d_HYBGDeviceSetMatIndexBase
#define T_HYBGDeviceSetMatType	       d_HYBGDeviceSetMatType
#define T_HYBGDeviceSetMatFillMode     d_HYBGDeviceSetMatFillMode
#define T_HYBGDeviceHybsmAnalysis      d_HYBGDeviceHybsmAnalysis
#define T_spsvHYBGDevice	       d_spsvHYBGDevice
#define T_HYBGHost2Device	       d_HYBGHost2Device
#define cusparseTcsrmv		       cusparseDcsrmv
#define cusparseTcsrsv_solve	       cusparseDcsrsv_solve
#define cusparseTcsrsv_analysis	       cusparseDcsrsv_analysis
#define cusparseThybmv		       cusparseDhybmv
#define cusparseThybsv_solve	       cusparseDhybsv_solve
#define cusparseThybsv_analysis	       cusparseDhybsv_analysis
#define cusparseTcsr2hyb               cusparseDcsr2hyb               

#include "fcusparse_fct.h"


/*    Single precision complex   */ 
#define TYPE 			       float complex                     
#define T_CSRGDeviceMat		       c_CSRGDeviceMat
#define T_Cmat			       c_Cmat
#define T_HYBGDeviceMat		       c_HYBGDeviceMat
#define T_Hmat			       c_Hmat
#define T_spmvCSRGDevice	       c_spmvCSRGDevice
#define T_spsvCSRGDevice	       c_spsvCSRGDevice
#define T_CSRGDeviceGetParms	       c_CSRGDeviceGetParms
#define T_CSRGDeviceAlloc	       c_CSRGDeviceAlloc
#define T_CSRGDeviceFree	       c_CSRGDeviceFree
#define T_CSRGDeviceSetMatType	       c_CSRGDeviceSetMatType
#define T_CSRGDeviceSetMatFillMode     c_CSRGDeviceSetMatFillMode
#define T_CSRGDeviceSetMatDiagType     c_CSRGDeviceSetMatDiagType
#define T_CSRGDeviceSetMatIndexBase    c_CSRGDeviceSetMatIndexBase
#define T_CSRGDeviceCsrsmAnalysis      c_CSRGDeviceCsrsmAnalysis
#define T_CSRGHost2Device	       c_CSRGHost2Device
#define T_CSRGDevice2Host	       c_CSRGDevice2Host
#define T_HYBGDeviceFree	       c_HYBGDeviceFree
#define T_spmvHYBGDevice	       c_spmvHYBGDevice
#define T_HYBGDeviceAlloc	       c_HYBGDeviceAlloc
#define T_HYBGDeviceSetMatDiagType     c_HYBGDeviceSetMatDiagType
#define T_HYBGDeviceSetMatIndexBase    c_HYBGDeviceSetMatIndexBase
#define T_HYBGDeviceSetMatType	       c_HYBGDeviceSetMatType
#define T_HYBGDeviceSetMatFillMode     c_HYBGDeviceSetMatFillMode
#define T_HYBGDeviceHybsmAnalysis      c_HYBGDeviceHybsmAnalysis
#define T_spsvHYBGDevice	       c_spsvHYBGDevice
#define T_HYBGHost2Device	       c_HYBGHost2Device
#define cusparseTcsrmv		       cusparseCcsrmv
#define cusparseTcsrsv_solve	       cusparseCcsrsv_solve
#define cusparseTcsrsv_analysis	       cusparseCcsrsv_analysis
#define cusparseThybmv		       cusparseChybmv
#define cusparseThybsv_solve	       cusparseChybsv_solve
#define cusparseThybsv_analysis	       cusparseChybsv_analysis
#define cusparseTcsr2hyb               cusparseCcsr2hyb               

#include "fcusparse_fct.h"


/*    Double precision complex   */ 
#define TYPE 			       double complex                    
#define T_CSRGDeviceMat		       z_CSRGDeviceMat
#define T_Cmat			       z_Cmat
#define T_HYBGDeviceMat		       z_HYBGDeviceMat
#define T_Hmat			       z_Hmat
#define T_spmvCSRGDevice	       z_spmvCSRGDevice
#define T_spsvCSRGDevice	       z_spsvCSRGDevice
#define T_CSRGDeviceGetParms	       z_CSRGDeviceGetParms
#define T_CSRGDeviceAlloc	       z_CSRGDeviceAlloc
#define T_CSRGDeviceFree	       z_CSRGDeviceFree
#define T_CSRGDeviceSetMatType	       z_CSRGDeviceSetMatType
#define T_CSRGDeviceSetMatFillMode     z_CSRGDeviceSetMatFillMode
#define T_CSRGDeviceSetMatDiagType     z_CSRGDeviceSetMatDiagType
#define T_CSRGDeviceSetMatIndexBase    z_CSRGDeviceSetMatIndexBase
#define T_CSRGDeviceCsrsmAnalysis      z_CSRGDeviceCsrsmAnalysis
#define T_CSRGHost2Device	       z_CSRGHost2Device
#define T_CSRGDevice2Host	       z_CSRGDevice2Host
#define T_HYBGDeviceFree	       z_HYBGDeviceFree
#define T_spmvHYBGDevice	       z_spmvHYBGDevice
#define T_HYBGDeviceAlloc	       z_HYBGDeviceAlloc
#define T_HYBGDeviceSetMatDiagType     z_HYBGDeviceSetMatDiagType
#define T_HYBGDeviceSetMatIndexBase    z_HYBGDeviceSetMatIndexBase
#define T_HYBGDeviceSetMatType	       z_HYBGDeviceSetMatType
#define T_HYBGDeviceSetMatFillMode     z_HYBGDeviceSetMatFillMode
#define T_HYBGDeviceHybsmAnalysis      z_HYBGDeviceHybsmAnalysis
#define T_spsvHYBGDevice	       z_spsvHYBGDevice
#define T_HYBGHost2Device	       z_HYBGHost2Device
#define cusparseTcsrmv		       cusparseZcsrmv
#define cusparseTcsrsv_solve	       cusparseZcsrsv_solve
#define cusparseTcsrsv_analysis	       cusparseZcsrsv_analysis
#define cusparseThybmv		       cusparseZhybmv
#define cusparseThybsv_solve	       cusparseZhybsv_solve
#define cusparseThybsv_analysis	       cusparseZhybsv_analysis
#define cusparseTcsr2hyb               cusparseZcsr2hyb               

#include "fcusparse_fct.h"


#endif 
