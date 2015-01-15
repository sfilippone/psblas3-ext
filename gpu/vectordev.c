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
#include "cuComplex.h"
#include "vectordev.h"
#include "cuda_runtime.h"
#include "core.h"

//new
MultiVectorDeviceParams getMultiVectorDeviceParams(unsigned int count, unsigned int size, unsigned int elementType)
{
  struct MultiVectorDeviceParams params;

  if (count == 1)
    params.pitch = size;
  else
    if (elementType == SPGPU_TYPE_INT)
      {
	//fprintf(stderr,"Getting parms for  a DOUBLE vector\n");
	params.pitch = (((size*sizeof(int) + 255)/256)*256)/sizeof(int);
      }
    else if (elementType == SPGPU_TYPE_DOUBLE)
      {
	//fprintf(stderr,"Getting parms for  a DOUBLE vector\n");
	params.pitch = (((size*sizeof(double) + 255)/256)*256)/sizeof(double);
      }
    else if (elementType == SPGPU_TYPE_FLOAT)
      {
	params.pitch = (((size*sizeof(float) + 255)/256)*256)/sizeof(float);
      }
    else if (elementType == SPGPU_TYPE_COMPLEX_FLOAT)
      {
	params.pitch = (((size*sizeof(cuFloatComplex) + 255)/256)*256)/sizeof(cuFloatComplex);
      }
    else if (elementType == SPGPU_TYPE_COMPLEX_DOUBLE)
      {
	params.pitch = (((size*sizeof(cuDoubleComplex) + 255)/256)*256)/sizeof(cuDoubleComplex);
      }
    else
      params.pitch = 0;

  params.elementType = elementType;
	
  params.count = count;
  params.size = size;

  return params;

}
//new
int allocMultiVecDevice(void ** remoteMultiVec, struct MultiVectorDeviceParams *params)
{
  if (params->pitch == 0)
    return SPGPU_UNSUPPORTED; // Unsupported params
  
  struct MultiVectDevice *tmp = (struct MultiVectDevice *)malloc(sizeof(struct MultiVectDevice));
  *remoteMultiVec = (void *)tmp;
  tmp->size_ = params->size;
  tmp->count_ = params->count;

  if (params->elementType == SPGPU_TYPE_INT)
    {
      if (params->count == 1)
	tmp->pitch_ = params->size;
      else
	tmp->pitch_ = (((params->size*sizeof(int) + 255)/256)*256)/sizeof(int);
      //fprintf(stderr,"Allocating  an INT vector %ld\n",tmp->pitch_*tmp->count_*sizeof(double));
      
      return allocRemoteBuffer((void **)&(tmp->v_), tmp->pitch_*params->count*sizeof(int));
    }
  else if (params->elementType == SPGPU_TYPE_FLOAT)
    {
      if (params->count == 1)
	tmp->pitch_ = params->size;
      else
	tmp->pitch_ = (((params->size*sizeof(float) + 255)/256)*256)/sizeof(float);

      return allocRemoteBuffer((void **)&(tmp->v_), tmp->pitch_*params->count*sizeof(float));
    }
  else if (params->elementType == SPGPU_TYPE_DOUBLE)
    {
      
      if (params->count == 1)
	tmp->pitch_ = params->size;
      else
	tmp->pitch_ = (int)(((params->size*sizeof(double) + 255)/256)*256)/sizeof(double);
      //fprintf(stderr,"Allocating  a DOUBLE vector %ld\n",tmp->pitch_*tmp->count_*sizeof(double));
 
      return allocRemoteBuffer((void **)&(tmp->v_), tmp->pitch_*tmp->count_*sizeof(double));
    }
  else if (params->elementType == SPGPU_TYPE_COMPLEX_FLOAT)
    {
      if (params->count == 1)
	tmp->pitch_ = params->size;
      else
	tmp->pitch_ = (int)(((params->size*sizeof(cuFloatComplex) + 255)/256)*256)/sizeof(cuFloatComplex);
      return allocRemoteBuffer((void **)&(tmp->v_), tmp->pitch_*tmp->count_*sizeof(cuFloatComplex));
    }
  else if (params->elementType == SPGPU_TYPE_COMPLEX_DOUBLE)
    {
      if (params->count == 1)
	tmp->pitch_ = params->size;
      else
	tmp->pitch_ = (int)(((params->size*sizeof(cuDoubleComplex) + 255)/256)*256)/sizeof(cuDoubleComplex);
      return allocRemoteBuffer((void **)&(tmp->v_), tmp->pitch_*tmp->count_*sizeof(cuDoubleComplex));
    }
  else
    return SPGPU_UNSUPPORTED; // Unsupported params
  return SPGPU_SUCCESS; // Success
}

int allocateIdx(void **d_idx, int n)
{
  return allocRemoteBuffer((void **)(d_idx), n*sizeof(int));
}

int writeIdx(void *d_idx, int* h_idx, int n)
{
  int i,j;
  int *di;
  i = writeRemoteBuffer((void*)h_idx, (void*)d_idx, n*sizeof(int));
  /* fprintf(stderr,"End of writeIdx: "); */
  /* di = (int *)d_idx; */
  /* for (j=0; j<n; j++) */
  /*   fprintf(stderr,"%d ",di[j]); */
  /* fprintf(stderr,"\n"); */
  //cudaSync();
  return i;
}

int readIdx(void* d_idx, int* h_idx, int n)
{ int i;
  i = readRemoteBuffer((void *) h_idx, (void *) d_idx, n*sizeof(int));
  //cudaSync();
  return(i);
}

int allocateMultiIdx(void **d_idx, int m, int n)
{
  return allocRemoteBuffer((void **)(d_idx), m*n*sizeof(int));
}

int writeMultiIdx(void *d_idx, int* h_idx, int m, int n)
{
  int i,j;
  int *di;
  i = writeRemoteBuffer((void*)h_idx, (void*)d_idx, m*n*sizeof(int));
  /* fprintf(stderr,"End of writeIdx: "); */
  /* di = (int *)d_idx; */
  /* for (j=0; j<n; j++) */
  /*   fprintf(stderr,"%d ",di[j]); */
  /* fprintf(stderr,"\n"); */
  //cudaSync();
  return i;
}

int readMultiIdx(void* d_idx, int* h_idx, int m, int n)
{ int i;
  i = readRemoteBuffer((void *) h_idx, (void *) d_idx, m*n*sizeof(int));
  //cudaSync();
  return(i);
}

void freeIdx(void *d_idx)
{
  //printf("Before freeIdx\n");
  freeRemoteBuffer(d_idx);
}


int registerMappedInt(void  *buff, void **d_p, int n, int dummy)
{
  return registerMappedMemory(buff,d_p,n*sizeof(int));
}

int registerMappedFloat(void  *buff, void **d_p, int n, float dummy)
{
  return registerMappedMemory(buff,d_p,n*sizeof(float));
}


int registerMappedDouble(void  *buff, void **d_p, int n, double dummy)
{
  return registerMappedMemory(buff,d_p,n*sizeof(double));
}

int registerMappedFloatComplex(void  *buff, void **d_p, int n, cuFloatComplex dummy)
{
  return registerMappedMemory(buff,d_p,n*sizeof(cuFloatComplex));
}


int registerMappedDoubleComplex(void  *buff, void **d_p, int n, cuDoubleComplex dummy)
{
  return registerMappedMemory(buff,d_p,n*sizeof(cuDoubleComplex));
}


int unregisterMapped(void *buff)
{
  return unregisterMappedMemory(buff);
}

void freeMultiVecDevice(void* deviceVec)
{
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // fprintf(stderr,"freeMultiVecDevice\n");
  if (devVec != NULL) {
    //fprintf(stderr,"Before freeMultiVecDevice% ld\n",devVec->pitch_*devVec->count_*sizeof(double));
    freeRemoteBuffer(devVec->v_);
    free(deviceVec);
  }
}

int FallocMultiVecDevice(void** deviceMultiVec, unsigned int count,
			 unsigned int size, unsigned int elementType)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectorDeviceParams p;

  p = getMultiVectorDeviceParams(count, size, elementType);
  i = allocMultiVecDevice(deviceMultiVec, &p);
  //cudaSync();
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d, %d %d \n","FallocMultiVecDevice",i, count, size);
  }
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int writeMultiVecDeviceInt(void* deviceVec, int* hostVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // Ex updateFromHost vector function
  i = writeRemoteBuffer((void*) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(int));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceFloat",i);
  }*/
  //  cudaSync();
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int writeMultiVecDeviceFloat(void* deviceVec, float* hostVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // Ex updateFromHost vector function
  i = writeRemoteBuffer((void*) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(float));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceFloat",i);
  }*/
  //  cudaSync();
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int writeMultiVecDeviceDouble(void* deviceVec, double* hostVec)
{ int i;
#ifdef HAVE_SPGPU
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
#else
    return SPGPU_UNSUPPORTED;
#endif
}

int writeMultiVecDeviceFloatComplex(void* deviceVec, cuFloatComplex* hostVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // Ex updateFromHost vector function
  i = writeRemoteBuffer((void*) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(cuFloatComplex));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceDouble",i);
  }*/
  //  cudaSync();
  return(i);
#else
    return SPGPU_UNSUPPORTED;
#endif
}

int writeMultiVecDeviceDoubleComplex(void* deviceVec, cuDoubleComplex* hostVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // Ex updateFromHost vector function
  i = writeRemoteBuffer((void*) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(cuDoubleComplex));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceDouble",i);
  }*/
  //  cudaSync();
  return(i);
#else
    return SPGPU_UNSUPPORTED;
#endif
}

int writeMultiVecDeviceFloatR2(void* deviceVec, float* hostVec, int ld)
{ int i;
#ifdef HAVE_SPGPU
  i = writeMultiVecDeviceFloat(deviceVec, (void *) hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceFloatR2",i);
  }
  //  cudaSync();
  return(i);
#else
    return SPGPU_UNSUPPORTED;
#endif
}

int writeMultiVecDeviceIntR2(void* deviceVec, int* hostVec, int ld)
{ int i;
#ifdef HAVE_SPGPU
  i = writeMultiVecDeviceInt(deviceVec, (void *) hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceFloatR2",i);
  }
  //  cudaSync();
  return(i);
#else
    return SPGPU_UNSUPPORTED;
#endif
}

int writeMultiVecDeviceDoubleR2(void* deviceVec, double* hostVec, int ld)
{ int i;
#ifdef HAVE_SPGPU
  i = writeMultiVecDeviceDouble(deviceVec, (void *) hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceDoubleR2",i);
  }
  //  cudaSync();
  return(i);
#else
    return SPGPU_UNSUPPORTED;
#endif
}

int writeMultiVecDeviceFloatComplexR2(void* deviceVec, cuFloatComplex* hostVec, int ld)
{ int i;
#ifdef HAVE_SPGPU
  i = writeMultiVecDeviceFloatComplex(deviceVec, (void *) hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceDoubleR2",i);
  }
  //  cudaSync();
  return(i);
#else
    return SPGPU_UNSUPPORTED;
#endif
}

int writeMultiVecDeviceDoubleComplexR2(void* deviceVec, cuDoubleComplex* hostVec, int ld)
{ int i;
#ifdef HAVE_SPGPU
  i = writeMultiVecDeviceDoubleComplex(deviceVec, (void *) hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceDoubleR2",i);
  }
  //  cudaSync();
  return(i);
#else
    return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceInt(void* deviceVec, int* hostVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  i = readRemoteBuffer((void *) hostVec, (void *)devVec->v_, 
		       devVec->pitch_*devVec->count_*sizeof(int));
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceInt",i);
  }
  //  cudaSync();
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceFloat(void* deviceVec, float* hostVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  i = readRemoteBuffer((void *) hostVec, (void *)devVec->v_, 
		       devVec->pitch_*devVec->count_*sizeof(float));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceFloat",i);
  }*/
  //  cudaSync();
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceDouble(void* deviceVec, double* hostVec)
{ int i,j;
#ifdef HAVE_SPGPU
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
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceFloatComplex(void* deviceVec, float complex* hostVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  i = readRemoteBuffer((void *) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(cuFloatComplex));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceDouble",i);
  }*/
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceDoubleComplex(void* deviceVec, double complex* hostVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  i = readRemoteBuffer((void *) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(cuDoubleComplex));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceDouble",i);
  }*/
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceGatherDouble(void* deviceVec, double* hostVec, int * idx, int *n)
{int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  //i = readMultiVecDeviceGather(deviceVec, (void *) hostVec, (void *) idx, (void *)n);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceGatherDouble",i);
  }*/
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}


int readMultiVecDeviceIntR2(void* deviceVec, int* hostVec, int ld)
{ int i;
#ifdef HAVE_SPGPU
  i = readMultiVecDeviceInt(deviceVec, (void *) hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceFloatR2",i);
  }
  //  cudaSync();
  return(i);
#else
    return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceFloatR2(void* deviceVec, float* hostVec, int ld)
{ int i;
#ifdef HAVE_SPGPU
  //i = readMultiVecDevice(deviceVec, (void *) hostVec);
  i = readMultiVecDeviceFloat(deviceVec, hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceFloatR2",i);
  }
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceDoubleR2(void* deviceVec, double* hostVec, int ld)
{ int i;
#ifdef HAVE_SPGPU
  //i = readMultiVecDevice(deviceVec, (void *) hostVec);
  i = readMultiVecDeviceDouble(deviceVec, hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceDoubleR2",i);
  }
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceFloatComplexR2(void* deviceVec, float complex* hostVec, int ld)
{ int i;
#ifdef HAVE_SPGPU
  //i = readMultiVecDevice(deviceVec, (void *) hostVec);
  i = readMultiVecDeviceFloatComplex(deviceVec, hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceDoubleR2",i);
  }
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceDoubleComplexR2(void* deviceVec, double complex* hostVec, int ld)
{ int i;
#ifdef HAVE_SPGPU
  //i = readMultiVecDevice(deviceVec, (void *) hostVec);
  i = readMultiVecDeviceDoubleComplex(deviceVec, hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceDoubleR2",i);
  }
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int nrm2MultiVecDeviceFloat(float* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
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
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int nrm2MultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuDmnrm2(handle, y_res, n,(double *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int nrm2MultiVecDeviceFloatComplex(float* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuCmnrm2(handle, y_res, n,(cuFloatComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int nrm2MultiVecDeviceDoubleComplex(double* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  spgpuHandle_t handle=psb_gpuGetHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuZmnrm2(handle, y_res, n,(cuDoubleComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}


int amaxMultiVecDeviceFloat(float* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  cublasHandle_t handle=psb_gpuGetCublasHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuSmamax(handle, y_res, n,(float *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int asumMultiVecDeviceFloat(float* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  cublasHandle_t handle=psb_gpuGetCublasHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuSmasum(handle, y_res, n,(float *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}



int amaxMultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  cublasHandle_t handle=psb_gpuGetCublasHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuDmamax(handle, y_res, n,(double *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int asumMultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  cublasHandle_t handle=psb_gpuGetCublasHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuDmasum(handle, y_res, n,(double *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}



int amaxMultiVecDeviceFloatComplex(float* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  cublasHandle_t handle=psb_gpuGetCublasHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuCmamax(handle, y_res, n,(cuFloatComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int asumMultiVecDeviceFloatComplex(float* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  cublasHandle_t handle=psb_gpuGetCublasHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuCmasum(handle, y_res, n,(cuFloatComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}



int amaxMultiVecDeviceDoubleComplex(double* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  cublasHandle_t handle=psb_gpuGetCublasHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuZmamax(handle, y_res, n,(cuDoubleComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int asumMultiVecDeviceDoubleComplex(double* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  cublasHandle_t handle=psb_gpuGetCublasHandle();
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuZmasum(handle, y_res, n,(cuDoubleComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}



int dotMultiVecDeviceFloat(float* y_res, int n, void* devMultiVecA, void* devMultiVecB)
{ int i = 0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) devMultiVecB;
#ifdef HAVE_SPGPU
  spgpuHandle_t handle=psb_gpuGetHandle();

  spgpuSmdot(handle, y_res, n, (float*)devVecA->v_, (float*)devVecB->v_,devVecA->count_,devVecB->pitch_);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int dotMultiVecDeviceDouble(double* y_res, int n, void* devMultiVecA, void* devMultiVecB)
{int i=0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) devMultiVecB;
#ifdef HAVE_SPGPU
  spgpuHandle_t handle=psb_gpuGetHandle();

  spgpuDmdot(handle, y_res, n, (double*)devVecA->v_, (double*)devVecB->v_,devVecA->count_,devVecB->pitch_);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int dotMultiVecDeviceFloatComplex(float complex* y_res, int n, void* devMultiVecA, void* devMultiVecB)
{int i=0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) devMultiVecB;
#ifdef HAVE_SPGPU
  spgpuHandle_t handle=psb_gpuGetHandle();
  /*for(i=0;i<n;i++)
    res[i] = make_cuFloatComplex(crealf(y_res[i]),cimagf(y_res[i]));*/
    
  spgpuCmdot(handle, (cuFloatComplex *)y_res, n, (cuFloatComplex*)devVecA->v_,
	     (cuFloatComplex*)devVecB->v_,devVecA->count_,devVecB->pitch_);
  return(0);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int dotMultiVecDeviceDoubleComplex(double complex* y_res, int n, void* devMultiVecA, void* devMultiVecB)
{int i=0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) devMultiVecB;
#ifdef HAVE_SPGPU
  spgpuHandle_t handle=psb_gpuGetHandle();
  spgpuZmdot(handle, (cuDoubleComplex*)y_res, n, (cuDoubleComplex*)devVecA->v_,
	     (cuDoubleComplex*)devVecB->v_,devVecA->count_,devVecB->pitch_);
  return(0);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

#if 0 
int geinsMultiVecDeviceDouble(int n, void* devMultiVecIrl, void* devMultiVecVal, 
			      int dupl, int indexBase, void* devMultiVecX)
{ int j=0, i=0,nmin=0,nmax=0;
  int pitch = 0;
#ifdef HAVE_SPGPU
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
#else
  return SPGPU_UNSUPPORTED;
#endif
}
#endif


int axpbyMultiVecDeviceFloat(int n, float alpha, void* devMultiVecX, float beta, void* devMultiVecY)
{ int i=0, j=0, pitch=0;
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
#ifdef HAVE_SPGPU
  spgpuHandle_t handle=psb_gpuGetHandle();
  pitch = devVecY->pitch_;
  if ((n > devVecY->size_) || (n>devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;
  for(j=0;j<devVecY->count_;j++)
    spgpuSaxpby(handle,(float*)devVecY->v_+pitch*j, n, beta,
		(float*)devVecY->v_+pitch*j, alpha,(float*) devVecX->v_+pitch*j);
  return(i); 
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axpbyMultiVecDeviceDouble(int n,double alpha, void* devMultiVecX, 
			      double beta, void* devMultiVecY)
{ int j=0, i=0;
  int pitch = 0;
#ifdef HAVE_SPGPU
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
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axpbyMultiVecDeviceFloatComplex(int n, float complex alpha, void* devMultiVecX, 
				    float complex beta, void* devMultiVecY)
{ int j=0, i=0;
  int pitch = 0;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  spgpuHandle_t handle=psb_gpuGetHandle();
  cuFloatComplex a, b;
  a = make_cuFloatComplex(crealf(alpha),cimagf(alpha));
  b = make_cuFloatComplex(crealf(beta),cimagf(beta));
  pitch = devVecY->pitch_;
  if ((n > devVecY->size_) || (n>devVecX->size_ )) 
    return SPGPU_UNSUPPORTED;

  for(j=0;j<devVecY->count_;j++)
    spgpuCaxpby(handle,(cuFloatComplex*)devVecY->v_+pitch*j, n, b, 
		(cuFloatComplex*)devVecY->v_+pitch*j, a,(cuFloatComplex*) devVecX->v_+pitch*j);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axpbyMultiVecDeviceDoubleComplex(int n, double complex alpha, void* devMultiVecX, 
				     double complex beta, void* devMultiVecY)
{ int i = 0, j=0;
  int pitch = 0;
#ifdef HAVE_SPGPU
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
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int getMultiVecDeviceSize(void* deviceVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *dev = (struct MultiVectDevice *) deviceVec;
  i = dev->size_;
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int getMultiVecDeviceCount(void* deviceVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *dev = (struct MultiVectDevice *) deviceVec;
  i = dev->count_;
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int getMultiVecDevicePitch(void* deviceVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *dev = (struct MultiVectDevice *) deviceVec;
  i = dev->pitch_;
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axyMultiVecDeviceFloat(int n, float alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0;
#ifdef HAVE_SPGPU
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
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axyMultiVecDeviceDouble(int n, double alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0; 
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  spgpuHandle_t handle=psb_gpuGetHandle();
  if ((n > devVecA->size_) || (n>devVecB->size_ )) 
    return SPGPU_UNSUPPORTED;

  spgpuDmaxy(handle, (double*)devVecB->v_, n, alpha, (double*)devVecA->v_, 
	     (double*)devVecB->v_, devVecA->count_, devVecA->pitch_);
  //i = axyMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, deviceVecB);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axyMultiVecDeviceFloatComplex(int n, float complex alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  spgpuHandle_t handle=psb_gpuGetHandle();

  if ((n > devVecA->size_) || (n>devVecB->size_ )) 
    return SPGPU_UNSUPPORTED;

  cuFloatComplex a = make_cuFloatComplex(crealf(alpha),cimagf(alpha));
  spgpuCmaxy(handle, (cuFloatComplex *)devVecB->v_, n, a,
	     (cuFloatComplex *)devVecA->v_, (cuFloatComplex *)devVecB->v_,
	     devVecA->count_, devVecA->pitch_);
  //i = axyMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, deviceVecB);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axyMultiVecDeviceDoubleComplex(int n, double complex alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0;
#ifdef HAVE_SPGPU
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
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axybzMultiVecDeviceFloat(int n, float alpha, void *deviceVecA, 
			     void *deviceVecB, float beta, void *deviceVecZ)
{ int i = 0;
#ifdef HAVE_SPGPU
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
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axybzMultiVecDeviceDouble(int n, double alpha, void *deviceVecA,
			      void *deviceVecB, double beta, void *deviceVecZ)
{ int i=0;
#ifdef HAVE_SPGPU
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
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axybzMultiVecDeviceFloatComplex(int n, float complex alpha, void *deviceVecA,
				    void *deviceVecB, float complex beta, 
				    void *deviceVecZ)
{ int i=0;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) deviceVecZ;
  spgpuHandle_t handle=psb_gpuGetHandle();

  if ((n > devVecA->size_) || (n>devVecB->size_ ) || (n>devVecZ->size_ )) 
    return SPGPU_UNSUPPORTED;

  cuFloatComplex a = make_cuFloatComplex(crealf(alpha),cimagf(alpha));
  cuFloatComplex b = make_cuFloatComplex(crealf(beta),cimagf(beta));
  spgpuCmaxypbz(handle, (cuFloatComplex *)devVecZ->v_, n, b, (cuFloatComplex *)devVecZ->v_, 
	       a, (cuFloatComplex *) devVecA->v_, (cuFloatComplex *) devVecB->v_,
	       devVecB->count_, devVecB->pitch_);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axybzMultiVecDeviceDoubleComplex(int n, double complex alpha, void *deviceVecA,
				     void *deviceVecB, double complex beta, 
				     void *deviceVecZ)
{ int i=0;
#ifdef HAVE_SPGPU
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
#else
  return SPGPU_UNSUPPORTED;
#endif
}

//New gather functions single and double precision

int igathMultiVecDeviceFloat(void* deviceVec, int vectorId, int first,
			     int n, void* indexes, void* host_values, int indexBase)
{
  int i, *idx;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  i=0; 
  idx = (int *) indexes;
  idx = &(idx[first-indexBase]);
  spgpuSgath(handle,(double *)host_values, n, idx,
  	     indexBase, (double *) devVec->v_+vectorId*devVec->pitch_);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int igathMultiVecDeviceDouble(void* deviceVec, int vectorId, int first,
			      int n, void* indexes, void* host_values, int indexBase)
{
  int i, *idx;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  i=0; 
  idx = (int *) indexes;
  idx = &(idx[first-indexBase]);
  spgpuDgath(handle,(double *)host_values, n, idx,
  	     indexBase, (double *) devVec->v_+vectorId*devVec->pitch_);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}


int igathMultiVecDeviceFloatComplex(void* deviceVec, int vectorId, int first,
			     int n, void* indexes, void* host_values, int indexBase)
{
  int i, *idx;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  i=0; 
  idx = (int *) indexes;
  idx = &(idx[first-indexBase]);
  spgpuCgath(handle,(double *)host_values, n, idx,
  	     indexBase, (double *) devVec->v_+vectorId*devVec->pitch_);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}


int igathMultiVecDeviceDoubleComplex(void* deviceVec, int vectorId, int first,
			      int n, void* indexes, void* host_values, int indexBase)
{
  int i, *idx;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  i=0; 
  idx = (int *) indexes;
  idx = &(idx[first-indexBase]);
  spgpuZgath(handle,(double *)host_values, n, idx,
  	     indexBase, (double *) devVec->v_+vectorId*devVec->pitch_);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}



int iscatMultiVecDeviceFloat(void* deviceVec, int vectorId, int first, int n, void *indexes,
			     void* host_values, int indexBase, float beta)
{ int i=0;
#ifdef HAVE_SPGPU
  int *idx=(int *) indexes;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  idx = &(idx[first-indexBase]);
  spgpuSscat(handle, (float *) devVec->v_, n, (float *) host_values, idx, indexBase, beta);
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
  
}

int iscatMultiVecDeviceDouble(void* deviceVec, int vectorId, int first, int n, void *indexes,
			     void* host_values, int indexBase, double beta)
{ int i=0;
#ifdef HAVE_SPGPU
  int *idx=(int *) indexes;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  idx = &(idx[first-indexBase]);
  spgpuDscat(handle, (double *) devVec->v_, n, (double *) host_values, idx, indexBase, beta);
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
  
}


int iscatMultiVecDeviceFloatComplex(void* deviceVec, int vectorId, int first, int n, void *indexes,
				    void* host_values, int indexBase, float complex beta)
{ int i=0;
#ifdef HAVE_SPGPU
  int *idx=(int *) indexes;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();
  cuFloatComplex cuBeta;

  cuBeta = make_cuFloatComplex(crealf(beta),cimagf(beta));
  idx = &(idx[first-indexBase]);
  spgpuCscat(handle, (cuFloatComplex *) devVec->v_, n, (cuFloatComplex *) host_values, 
	     idx, indexBase, cuBeta);
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
  
}


int iscatMultiVecDeviceDoubleComplex(void *deviceVec, int vectorId, int first, int n, void *indexes,
				    void *host_values, int indexBase, double complex beta)
{ int i=0;
#ifdef HAVE_SPGPU
  int *idx=(int *) indexes;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();
  cuDoubleComplex cuBeta;

  idx = &(idx[first-indexBase]);
  cuBeta = make_cuDoubleComplex(crealf(beta),cimagf(beta));

  spgpuZscat(handle, (cuDoubleComplex *) devVec->v_, n, (cuDoubleComplex *) host_values, 
	     idx, indexBase, cuBeta);
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
  
}


#endif
