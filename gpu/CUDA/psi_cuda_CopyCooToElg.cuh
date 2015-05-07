#include <stdlib.h>
#include <stdio.h>

#include "cintrf.h"
#include "vectordev.h"
#include "psi_cuda_common.cuh"


#undef GEN_PSI_FUNC_NAME
#define GEN_PSI_FUNC_NAME(x) CONCAT(CONCAT(psi_cuda_,x),_CopyCooToElg)

#define THREAD_BLOCK 256

#ifdef __cplusplus
extern "C" {
#endif


void GEN_PSI_FUNC_NAME(TYPE_SYMBOL)(spgpuHandle_t handle, int nr, int nc, int nza, int hacksz, int ldv, int nzm,
		   int *rS,int *devIdisp, int *devJa, VALUE_TYPE *devVal,
		   int *rP, VALUE_TYPE *cM);


#ifdef __cplusplus
}
#endif



 

__global__ void   CONCAT(GEN_PSI_FUNC_NAME(TYPE_SYMBOL),_krn)(int ii, int nrws, int nr, int nza, int hacksz, int ldv, int nzm,
			  int *rS, int *devIdisp, int *devJa, VALUE_TYPE *devVal,  int *rP, VALUE_TYPE *cM)
{
  int ir, k, ipnt, rsz;
  int ki = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
  int i=ii+ki; 

  if (ki >= nrws) return; 
  if (i >= nr) return; 

  ipnt=devIdisp[i];
  rsz=rS[i];
  ir = i;
  for (k=0; k<rsz; k++) {
    rP[ir] = devJa[ipnt];
    cM[ir] = devVal[ipnt];
    ir += ldv;
    ipnt++;
  }
  for (k=rsz; k<nzm; k++) {
    rP[ir] = i;
    cM[ir] = CONCAT(zero_,VALUE_TYPE)();
    ir += ldv;
  }
    
}    



void  CONCAT(GEN_PSI_FUNC_NAME(TYPE_SYMBOL),_)(spgpuHandle_t handle, int nrws, int i, int nr, int nza, int hacksz, int ldv, int nzm,
			  int *rS,int *devIdisp, int *devJa, VALUE_TYPE *devVal,  int *rP, VALUE_TYPE *cM)
{
  dim3 block (THREAD_BLOCK, 1);
  dim3 grid ((nrws + THREAD_BLOCK - 1) / THREAD_BLOCK);

  CONCAT(GEN_PSI_FUNC_NAME(TYPE_SYMBOL),_krn) 
    <<< grid, block, 0, handle->currentStream >>>(i,nrws, nr, nza, hacksz, ldv, nzm,
						  rS,devIdisp,devJa,devVal,rP,cM);

}




void 
GEN_PSI_FUNC_NAME(TYPE_SYMBOL)
(spgpuHandle_t handle, int nr, int nc, int nza, int hacksz, int ldv, int nzm,
		   int *rS,int *devIdisp, int *devJa, VALUE_TYPE *devVal,
		   int *rP, VALUE_TYPE *cM)
{ int i,j,k, nrws;
  //int maxNForACall = THREAD_BLOCK*handle->maxGridSizeX;
  int maxNForACall = max(handle->maxGridSizeX, THREAD_BLOCK*handle->maxGridSizeX);
  

  //fprintf(stderr,"Loop on j: %d\n",j); 
  for (i=0; i<nr; i+=nrws) {
    nrws = MIN(maxNForACall, nr - i);
    //fprintf(stderr,"ifirst: %d i : %d nrws: %d i + ifirst + (nrws -1) -1 %d \n",ifirst,i,nrws,i + ifirst + (nrws -1) -1);
    CONCAT(GEN_PSI_FUNC_NAME(TYPE_SYMBOL),_)(handle,nrws,i, nr, nza, hacksz, ldv, nzm,
			  rS,devIdisp, devJa, devVal,  rP, cM);
  }
}
