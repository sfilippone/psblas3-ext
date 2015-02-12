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
#include "svectordev.h"


int registerMappedFloat(void  *buff, void **d_p, int n, float dummy)
{
  return registerMappedMemory(buff,d_p,n*sizeof(float));
}

int writeMultiVecDeviceFloat(void* deviceVec, float* hostVec)
{ int i;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // Ex updateFromHost vector function
  i = writeRemoteBuffer((void*) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(float));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceFloat",i);
  }*/
  //  cudaSync();
  return(i);
}

int writeMultiVecDeviceFloatR2(void* deviceVec, float* hostVec, int ld)
{ int i;
  i = writeMultiVecDeviceFloat(deviceVec, (void *) hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceFloatR2",i);
  }
  //  cudaSync();
  return(i);
}

int readMultiVecDeviceFloat(void* deviceVec, float* hostVec)
{ int i;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  i = readRemoteBuffer((void *) hostVec, (void *)devVec->v_, 
		       devVec->pitch_*devVec->count_*sizeof(float));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceFloat",i);
  }*/
  //  cudaSync();
  return(i);
}

int readMultiVecDeviceFloatR2(void* deviceVec, float* hostVec, int ld)
{ int i;
  //i = readMultiVecDevice(deviceVec, (void *) hostVec);
  i = readMultiVecDeviceFloat(deviceVec, hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceFloatR2",i);
  }
  return(i);
}

int nrm2MultiVecDeviceFloat(float* y_res, int n, void* devMultiVecA)
{ int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  __assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuSmnrm2(handle,y_res,n,(float *)devVecA->v_,devVecA->count_,devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","nrm2MultiVecDeviceFloat",i);
  }*/
  return(i);
}

int amaxMultiVecDeviceFloat(float* y_res, int n, void* devMultiVecA)
{ int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuSmamax(handle, y_res, n,(float *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
}

int asumMultiVecDeviceFloat(float* y_res, int n, void* devMultiVecA)
{ int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuSmasum(handle, y_res, n,(float *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
}


int dotMultiVecDeviceFloat(float* y_res, int n, void* devMultiVecA, void* devMultiVecB)
{ int i = 0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) devMultiVecB;
  spgpuHandle_t handle=psb_gpuGetHandle();

  spgpuSmdot(handle, y_res, n, (float*)devVecA->v_, (float*)devVecB->v_,devVecA->count_,devVecB->pitch_);
  return(i);
}

int axpbyMultiVecDeviceFloat(int n, float alpha, void* devMultiVecX, float beta, void* devMultiVecY)
{ int i=0, j=0, pitch=0;
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  spgpuHandle_t handle=psb_gpuGetHandle();
  pitch = devVecY->pitch_;
  if ((n > devVecY->size_) || (n>devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  for(j=0;j<devVecY->count_;j++)
    spgpuSaxpby(handle,(float*)devVecY->v_+pitch*j, n, beta,
		(float*)devVecY->v_+pitch*j, alpha,(float*) devVecX->v_+pitch*j);
  return(i); 
}

int axyMultiVecDeviceFloat(int n, float alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  spgpuHandle_t handle=psb_gpuGetHandle();
  if ((n > devVecA->size_) || (n>devVecB->size_ )) 
    return SPGPU_UNSUPPORTED;

  spgpuSmaxy(handle, (float*)devVecB->v_, n, alpha,
	     (float*)devVecA->v_,(float *)devVecB->v_,
	     devVecA->count_, devVecA->pitch_);
    //i = axyMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, deviceVecB);
  return(i);
}

int axybzMultiVecDeviceFloat(int n, float alpha, void *deviceVecA, 
			     void *deviceVecB, float beta, void *deviceVecZ)
{ int i = 0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) deviceVecZ;
  spgpuHandle_t handle=psb_gpuGetHandle();

  if ((n > devVecA->size_) || (n>devVecB->size_ ) || (n>devVecZ->size_ )) 
    return SPGPU_UNSUPPORTED;

  spgpuSmaxypbz(handle, (float*)devVecZ->v_, n, beta, (float*)devVecZ->v_, alpha,
	       (float*) devVecA->v_, (float*) devVecB->v_, devVecB->count_,
	       devVecB->pitch_);
  //i = axybzMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, (float *) beta, deviceVecZ);
  return(i);
}

//New gather functions single and double precision

int igathMultiVecDeviceFloatVecIdx(void* deviceVec, int vectorId, int first,
			      int n, void* deviceIdx, void* host_values, int indexBase)
{
  int i, *idx;
  struct MultiVectDevice *devIdx = (struct MultiVectDevice *) deviceIdx;
  i= igathMultiVecDeviceFloat(deviceVec, vectorId, first,
			       n, (void*) devIdx->v_,  host_values, indexBase);
  return(i);
}

int igathMultiVecDeviceFloat(void* deviceVec, int vectorId, int first,
			      int n, void* indexes, void* host_values, int indexBase)
{
  int i, *idx;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  i=0; 
  idx = (int *) indexes;
  idx = &(idx[first-indexBase]);
  spgpuSgath(handle,(float *)host_values, n, idx,
  	     indexBase, (float *) devVec->v_+vectorId*devVec->pitch_);
  return(i);
}

int iscatMultiVecDeviceFloatVecIdx(void* deviceVec, int vectorId, int first, int n, void *deviceIdx,
			      void* host_values, int indexBase, float beta)
{  
  int i, *idx;
  struct MultiVectDevice *devIdx = (struct MultiVectDevice *) deviceIdx;
  i= iscatMultiVecDeviceFloat(deviceVec, vectorId, first, n, 
			       (void*) devIdx->v_,  host_values, indexBase, beta);
  return(i);
}

int iscatMultiVecDeviceFloat(void* deviceVec, int vectorId, int first, int n, void *indexes,
			     void* host_values, int indexBase, float beta)
{ int i=0;
  int *idx=(int *) indexes;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  idx = &(idx[first-indexBase]);
  spgpuSscat(handle, (float *) devVec->v_, n, (float *) host_values, idx, indexBase, beta);
  return SPGPU_SUCCESS;
  
}

#endif

