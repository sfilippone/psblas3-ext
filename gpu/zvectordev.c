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
#include "zvectordev.h"
//#include "utils.h"
//#include "common.h"


int registerMappedDoubleComplex(void  *buff, void **d_p, int n, cuDoubleComplex dummy)
{
  return registerMappedMemory(buff,d_p,n*sizeof(cuDoubleComplex));
}

int writeMultiVecDeviceDoubleComplex(void* deviceVec, cuDoubleComplex* hostVec)
{ int i;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // Ex updateFromHost vector function
  i = writeRemoteBuffer((void*) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(cuDoubleComplex));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceDouble",i);
  }*/
  //  cudaSync();
  return(i);
}

int writeMultiVecDeviceDoubleComplexR2(void* deviceVec, cuDoubleComplex* hostVec, int ld)
{ int i;
  i = writeMultiVecDeviceDoubleComplex(deviceVec, (void *) hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceDoubleR2",i);
  }
  //  cudaSync();
  return(i);
}

int readMultiVecDeviceDoubleComplex(void* deviceVec, double complex* hostVec)
{ int i;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  i = readRemoteBuffer((void *) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(cuDoubleComplex));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceDouble",i);
  }*/
  return(i);
}

int readMultiVecDeviceDoubleComplexR2(void* deviceVec, double complex* hostVec, int ld)
{ int i;
  //i = readMultiVecDevice(deviceVec, (void *) hostVec);
  i = readMultiVecDeviceDoubleComplex(deviceVec, hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceDoubleR2",i);
  }
  return(i);
}

int nrm2MultiVecDeviceDoubleComplex(double* y_res, int n, void* devMultiVecA)
{ int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuZmnrm2(handle, y_res, n,(cuDoubleComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
}

int amaxMultiVecDeviceDoubleComplex(double* y_res, int n, void* devMultiVecA)
{ int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuZmamax(handle, y_res, n,(cuDoubleComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
}

int asumMultiVecDeviceDoubleComplex(double* y_res, int n, void* devMultiVecA)
{ int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuZmasum(handle, y_res, n,(cuDoubleComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
}

int dotMultiVecDeviceDoubleComplex(double complex* y_res, int n, void* devMultiVecA, void* devMultiVecB)
{int i=0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) devMultiVecB;
  spgpuHandle_t handle=psb_gpuGetHandle();
  spgpuZmdot(handle, (cuDoubleComplex*)y_res, n, (cuDoubleComplex*)devVecA->v_,
	     (cuDoubleComplex*)devVecB->v_,devVecA->count_,devVecB->pitch_);
  return(0);
}

int axpbyMultiVecDeviceDoubleComplex(int n, double complex alpha, void* devMultiVecX, 
				     double complex beta, void* devMultiVecY)
{ int i = 0, j=0;
  int pitch = 0;
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  spgpuHandle_t handle=psb_gpuGetHandle();
  cuDoubleComplex a, b;
  a = make_cuDoubleComplex(crealf(alpha),cimagf(alpha));
  b = make_cuDoubleComplex(crealf(beta),cimagf(beta));
  pitch = devVecY->pitch_;
  if ((n > devVecY->size_) || (n>devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  for(j=0;j<devVecY->count_;j++)
    spgpuZaxpby(handle,(cuDoubleComplex*)devVecY->v_+pitch*j, n, b, 
		(cuDoubleComplex*)devVecY->v_+pitch*j, a,(cuDoubleComplex*) devVecX->v_+pitch*j);
  return(i);
}

int axyMultiVecDeviceDoubleComplex(int n, double complex alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  spgpuHandle_t handle=psb_gpuGetHandle();
  if ((n > devVecA->size_) || (n>devVecB->size_ )) 
    return SPGPU_UNSUPPORTED;

  cuDoubleComplex a = make_cuDoubleComplex(creal(alpha),cimag(alpha));
  spgpuZmaxy(handle, (cuDoubleComplex *)devVecB->v_, n, a,
	     (cuDoubleComplex *)devVecA->v_, (cuDoubleComplex *)devVecB->v_,
	     devVecA->count_, devVecA->pitch_);
  //i = axyMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, deviceVecB);
  return(i);
}

int axybzMultiVecDeviceDoubleComplex(int n, double complex alpha, void *deviceVecA,
				     void *deviceVecB, double complex beta, 
				     void *deviceVecZ)
{ int i=0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) deviceVecZ;
  spgpuHandle_t handle=psb_gpuGetHandle();

  if ((n > devVecA->size_) || (n>devVecB->size_ ) || (n>devVecZ->size_ )) 
    return SPGPU_UNSUPPORTED;

  cuDoubleComplex a = make_cuDoubleComplex(creal(alpha),cimag(alpha));
  cuDoubleComplex b = make_cuDoubleComplex(creal(beta),cimag(beta));
  spgpuZmaxypbz(handle, (cuDoubleComplex *)devVecZ->v_, n, b, (cuDoubleComplex *)devVecZ->v_, 
	       a, (cuDoubleComplex *) devVecA->v_, (cuDoubleComplex *) devVecB->v_,
	       devVecB->count_, devVecB->pitch_);
  return(i);
}

//New gather functions single and double precision

int igathMultiVecDeviceDoubleComplexVecIdx(void* deviceVec, int vectorId, int first,
			     int n, void* deviceIdx, void* host_values, int indexBase)
{
  int i, *idx;
  struct MultiVectDevice *devIdx = (struct MultiVectDevice *) deviceIdx;
  i= igathMultiVecDeviceDoubleComplex(deviceVec, vectorId, first,
			       n, (void*) devIdx->v_,  host_values, indexBase);
  return(i);
}

int igathMultiVecDeviceDoubleComplex(void* deviceVec, int vectorId, int first,
			     int n, void* indexes, void* host_values, int indexBase)
{
  int i, *idx;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  i=0; 
  idx = (int *) indexes;
  idx = &(idx[first-indexBase]);
  spgpuZgath(handle,(cuDoubleComplex *)host_values, n, idx,
  	     indexBase, (cuDoubleComplex *) devVec->v_+vectorId*devVec->pitch_);
  return(i);
}


int iscatMultiVecDeviceDoubleComplexVecIdx(void* deviceVec, int vectorId, int first,
			     int n, void* deviceIdx, void* host_values, int indexBase, double complex beta)
{
  int i, *idx;
  struct MultiVectDevice *devIdx = (struct MultiVectDevice *) deviceIdx;
  i= iscatMultiVecDeviceDoubleComplex(deviceVec, vectorId, first,
				     n, (void*) devIdx->v_,  host_values, indexBase, beta);
  return(i);
}

int iscatMultiVecDeviceDoubleComplex(void* deviceVec, int vectorId, int first, int n, void *indexes,
				    void* host_values, int indexBase, double complex beta)
{ int i=0;
  int *idx=(int *) indexes;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();
  cuDoubleComplex cuBeta;

  cuBeta = make_cuDoubleComplex(crealf(beta),cimagf(beta));
  idx = &(idx[first-indexBase]);
  spgpuZscat(handle, (cuDoubleComplex *) devVec->v_, n, (cuDoubleComplex *) host_values, 
	     idx, indexBase, cuBeta);
  return SPGPU_SUCCESS;
  
}

#endif

