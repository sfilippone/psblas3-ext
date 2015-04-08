#include <stdlib.h>
#include <stdio.h>

#include "cintrf.h"
#include "vectordev.h"

#define THREAD_BLOCK 256
#define MIN(A,B) ( (A)<(B) ? (A) : (B) )
#define SQUARE(x) ((x)*(x))
#define GET_ADDR(a,ix,iy,nc) a[(nc)*(ix)+(iy)]
#define GET_VAL(a,ix,iy,nc) (GET_ADDR(a,ix,iy,nc))

#ifdef __cplusplus
extern "C" {
#endif


void w_cuda_CopyCooToHlg(spgpuHandle_t handle, int nr, int nc, int nza, int hacksz, int noffs, int isz, 
			 int *rS, int *hackOffs, int *devIdisp, int *devJa, double *devVal,
			 int *rP, double *cM);


/* void _w_Cuda_coeff_upd_a_krn(int ifirst, int ii, int j, int nrws, int *idxs, int dim1,int dim2, */
/* 			  int *vi, int *vj, double *va, int *idxdiag, double *vdiag, */
/* 			  int nr, int nc, int *lidxs, double *vu,  */
/* 			     double c1, double c6); */

#ifdef __cplusplus
}
#endif


__global__ void _w_Cuda_cpy_coo_2_hlg_krn(int ii, int nrws, int nr, int nza,
					  int hacksz, int noffs, int isz,
					  int *rS, int *hackOffs, int *devIdisp, 
					  int *devJa, double *devVal, 
					  int *rP, double *cM)
{
  int ir, k, ipnt, rsz;
  int ki = threadIdx.x + blockIdx.x * (THREAD_BLOCK);
  int i=ii+ki; 

  if (ki >= nrws) return; 
  

  if (i<nr) {
    int hackId = i / hacksz;
    int hackLaneId = i % hacksz;
    int hackOffset = hackOffs[hackId] + hackLaneId;
    int nzm = (hackOffs[hackId+1]-hackOffs[hackId])/hacksz;
    rsz  = rS[i];
    ipnt = devIdisp[i];
    ir   = hackOffset;
    for (k=0; k<rsz; k++) {
#if 1
      rP[ir] = devJa[ipnt];
      cM[ir] = devVal[ipnt];
#else 
      rP[ir] = ipnt;
      cM[ir] = 0.0; 
#endif      
      ir += hacksz;
      ipnt++;
    }
    for (k=rsz; k<nzm; k++) {
      rP[ir] = i;
      cM[ir] = 0.0;
      ir += hacksz;
    }
  }
    
}    
  




void _w_Cuda_cpy_coo_2_hlg(spgpuHandle_t handle, int nrws, int i, int nr, int nza,
			   int hacksz, int noffs, int isz,
			   int *rS, int *hackOffs, int *devIdisp, int *devJa,
			   double *devVal,  int *rP, double *cM)
{
  dim3 block (THREAD_BLOCK, 1);
  dim3 grid ((nrws + THREAD_BLOCK - 1) / THREAD_BLOCK);
  
  _w_Cuda_cpy_coo_2_hlg_krn 
    <<< grid, block, 0, handle->currentStream >>>(i,nrws,nr, nza, hacksz, noffs, isz,
						  rS,hackOffs,devIdisp,devJa,devVal,rP,cM);

}


void w_cuda_CopyCooToHlg(spgpuHandle_t handle, int nr, int nc, int nza, int hacksz, int noffs, int isz, 
			 int *rS, int *hackOffs, int *devIdisp, int *devJa, double *devVal,
			 int *rP, double *cM)
{ int i, nrws;
  //int maxNForACall = THREAD_BLOCK*handle->maxGridSizeX;
  int maxNForACall = max(handle->maxGridSizeX, THREAD_BLOCK*handle->maxGridSizeX);
  
  //fprintf(stderr,"Loop on j: %d\n",j); 
  for (i=0; i<nr; i+=nrws) {
    nrws = MIN(maxNForACall, nr - i);
    //fprintf(stderr,"cpy_coo_2_hlg:  i : %d nrws: %d \n", i,nrws);
    _w_Cuda_cpy_coo_2_hlg(handle,nrws,i, nr, nza, hacksz, noffs, isz,
			  rS, hackOffs, devIdisp, devJa, devVal,  rP, cM);
  }

}
