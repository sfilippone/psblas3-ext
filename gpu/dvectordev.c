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
#include <complex.h>
#if defined(HAVE_SPGPU)
//#include "utils.h"
//#include "common.h"
#include "dvectordev.h"


int registerMappedDouble(void  *buff, void **d_p, int n, double dummy)
{
  return registerMappedMemory(buff,d_p,n*sizeof(double));
}

int writeMultiVecDeviceDouble(void* deviceVec, double* hostVec)
{ int i;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // Ex updateFromHost vector function
  i = writeRemoteBuffer((void*) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(double));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceDouble",i);
  }*/
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","FallocMultiVecDevice",i);
  }

  //  cudaSync();
  return(i);
}

int writeMultiVecDeviceDoubleR2(void* deviceVec, double* hostVec, int ld)
{ int i;
  i = writeMultiVecDeviceDouble(deviceVec, (void *) hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceDoubleR2",i);
  }
  //  cudaSync();
  return(i);
}

int readMultiVecDeviceDouble(void* deviceVec, double* hostVec)
{ int i,j;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  i = readRemoteBuffer((void *) hostVec, (void *)devVec->v_, 
		       devVec->pitch_*devVec->count_*sizeof(double));
#if 0
  for (j=0;j<devVec->size_; j++) {
    fprintf(stderr,"readDouble:  %d  %lf \n",j,hostVec[j]);
  }
#endif
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceDouble",i);
  }
  //  cudaSync();
  return(i);
}

int readMultiVecDeviceGatherDouble(void* deviceVec, double* hostVec, int * idx, int *n)
{int i;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  //i = readMultiVecDeviceGather(deviceVec, (void *) hostVec, (void *) idx, (void *)n);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceGatherDouble",i);
  }*/
  return(i);
}

int readMultiVecDeviceDoubleR2(void* deviceVec, double* hostVec, int ld)
{ int i;
  //i = readMultiVecDevice(deviceVec, (void *) hostVec);
  i = readMultiVecDeviceDouble(deviceVec, hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceDoubleR2",i);
  }
  return(i);
}

int nrm2MultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA)
{ int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuDmnrm2(handle, y_res, n,(double *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
}

int amaxMultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA)
{ int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuDmamax(handle, y_res, n,(double *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
}

int asumMultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA)
{ int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuDmasum(handle, y_res, n,(double *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
}

int dotMultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA, void* devMultiVecB)
{int i=0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) devMultiVecB;
  spgpuHandle_t handle=psb_gpuGetHandle();

  spgpuDmdot(handle, y_res, n, (double*)devVecA->v_, (double*)devVecB->v_,devVecA->count_,devVecB->pitch_);
  return(i);
}


#if 0 
int geinsMultiVecDeviceDouble(int n, void* devMultiVecIrl, void* devMultiVecVal, 
			      int dupl, int indexBase, void* devMultiVecX)
{ int j=0, i=0,nmin=0,nmax=0;
  int pitch = 0;
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecIrl = (struct MultiVectDevice *) devMultiVecIrl;
  struct MultiVectDevice *devVecVal = (struct MultiVectDevice *) devMultiVecVal;
  spgpuHandle_t handle=psb_gpuGetHandle();
  pitch = devVecIrl->pitch_;
  if ((n > devVecIrl->size_) || (n>devVecVal->size_ )) 
    return SPGPU_UNSUPPORTED;

  //fprintf(stderr,"geins: %d %d  %p %p %p\n",dupl,n,devVecIrl->v_,devVecVal->v_,devVecX->v_);

#if DEBUG_GEINS
  int *hidx=(int *) malloc(n*sizeof(int));
  double *hval=(double *) malloc(n*sizeof(double));
  int nx=devVecX->size_;
  double *hx=(double *) malloc(nx*sizeof(double));
  i = readRemoteBuffer((void *) hidx, (void *)devVecIrl->v_, 
		       n*sizeof(int));
  i = readRemoteBuffer((void *) hval, (void *)devVecVal->v_, 
		       n*sizeof(double));
  
  i = readRemoteBuffer((void *) hx, (void *)devVecX->v_, 
		       nx*sizeof(double));
  if (n<nx) {
    for (j=0; j<n; j++) {
      fprintf(stderr,"before: %d  %d %16.12lf %16.12lf \n",j,hidx[j],hval[j],hx[j]);
    }
    for (j=n; j<nx; j++) {
      fprintf(stderr,"%d  %lf \n",j,hx[j]);
    }
  } else {
    for (j=0; j<nx; j++) {
      fprintf(stderr,"before: %d  %d %16.12lf %16.12lf \n",j,hidx[j],hval[j],hx[j]);
    }
    for (j=nx; j<n; j++) {
      fprintf(stderr,"%d   %lf %lf \n",j,hidx[j],hval[j]);
    }
  }


#endif
  
  spgpuDgeins(handle,n, (int*)devVecIrl->v_, 
  	      (double*)devVecVal->v_, dupl, indexBase,(double*) devVecX->v_);

#if DEBUG_GEINS
  i = readRemoteBuffer((void *) hidx, (void *)devVecIrl->v_, 
		       n*sizeof(int));
  i = readRemoteBuffer((void *) hval, (void *)devVecVal->v_, 
		       n*sizeof(double));
  
  i = readRemoteBuffer((void *) hx, (void *)devVecX->v_, 
		       nx*sizeof(double));
  if (n<nx) {
    for (j=0; j<n; j++) {
      fprintf(stderr,"after: %d  %d %16.12lf %16.12lf \n",j,hidx[j],hval[j],hx[j]);
    }
    for (j=n; j<nx; j++) {
      fprintf(stderr,"%d  %lf \n",j,hx[j]);
    }
  } else {
    for (j=0; j<nx; j++) {
      fprintf(stderr,"after: %d  %d %16.12lf %16.12lf \n",j,hidx[j],hval[j],hx[j]);
    }
    for (j=nx; j<n; j++) {
      fprintf(stderr,"%d   %lf %lf \n",j,hidx[j],hval[j]);
    }
  }


#endif
  //  fprintf(stderr,"geins: %d\n",i);
  

  return(i);
}
#endif

int axpbyMultiVecDeviceDouble(int n,double alpha, void* devMultiVecX, 
			      double beta, void* devMultiVecY)
{ int j=0, i=0;
  int pitch = 0;
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  spgpuHandle_t handle=psb_gpuGetHandle();
  pitch = devVecY->pitch_;
  if ((n > devVecY->size_) || (n>devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;

  for(j=0;j<devVecY->count_;j++)
    spgpuDaxpby(handle,(double*)devVecY->v_+pitch*j, n, beta, 
		(double*)devVecY->v_+pitch*j, alpha,(double*) devVecX->v_+pitch*j);
  return(i);
}

int axyMultiVecDeviceDouble(int n, double alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0; 
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  spgpuHandle_t handle=psb_gpuGetHandle();
  if ((n > devVecA->size_) || (n>devVecB->size_ )) 
    return SPGPU_UNSUPPORTED;

  spgpuDmaxy(handle, (double*)devVecB->v_, n, alpha, (double*)devVecA->v_, 
	     (double*)devVecB->v_, devVecA->count_, devVecA->pitch_);
  //i = axyMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, deviceVecB);
  return(i);
}

int axybzMultiVecDeviceDouble(int n, double alpha, void *deviceVecA,
			      void *deviceVecB, double beta, void *deviceVecZ)
{ int i=0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) deviceVecZ;
  spgpuHandle_t handle=psb_gpuGetHandle();

  if ((n > devVecA->size_) || (n>devVecB->size_ ) || (n>devVecZ->size_ )) 
    return SPGPU_UNSUPPORTED;
  spgpuDmaxypbz(handle, (double*)devVecZ->v_, n, beta, (double*)devVecZ->v_, 
	       alpha, (double*) devVecA->v_, (double*) devVecB->v_,
	       devVecB->count_, devVecB->pitch_);
  return(i);
}


int igathMultiVecDeviceDoubleVecIdx(void* deviceVec, int vectorId, int first,
			      int n, void* deviceIdx, void* host_values, int indexBase)
{
  int i, *idx;
  struct MultiVectDevice *devIdx = (struct MultiVectDevice *) deviceIdx;
  i= igathMultiVecDeviceDouble(deviceVec, vectorId, first,
			       n, (void*) devIdx->v_,  host_values, indexBase);
  return(i);
}

int igathMultiVecDeviceDouble(void* deviceVec, int vectorId, int first,
			      int n, void* indexes, void* host_values, int indexBase)
{
  int i, *idx;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  i=0; 
  idx = (int *) indexes;
  idx = &(idx[first-indexBase]);
  spgpuDgath(handle,(double *)host_values, n, idx,
  	     indexBase, (double *) devVec->v_+vectorId*devVec->pitch_);
  return(i);
}

int iscatMultiVecDeviceDoubleVecIdx(void* deviceVec, int vectorId, int first, int n, void *deviceIdx,
			      void* host_values, int indexBase, double beta)
{  
  int i, *idx;
  struct MultiVectDevice *devIdx = (struct MultiVectDevice *) deviceIdx;
  i= iscatMultiVecDeviceDouble(deviceVec, vectorId, first, n, 
			       (void*) devIdx->v_,  host_values, indexBase, beta);
  return(i);
}

int iscatMultiVecDeviceDouble(void* deviceVec, int vectorId, int first, int n, void *indexes,
			     void* host_values, int indexBase, double beta)
{ int i=0;
  int *idx=(int *) indexes;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  idx = &(idx[first-indexBase]);
  spgpuDscat(handle, (double *) devVec->v_, n, (double *) host_values, idx, indexBase, beta);
  return SPGPU_SUCCESS;
  
}

#endif

