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
#include "cuComplex.h"
//#include "utils.h"
//#include "common.h"
#if defined(HAVE_SPGPU)
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
    if (elementType == SPGPU_TYPE_DOUBLE)
      {
	fprintf(stderr,"Getting parms for  a DOUBLE vector\n");
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

  if (params->elementType == SPGPU_TYPE_FLOAT)
    {
      if (params->count == 1)
	tmp->pitch_ = params->size;
      else
	tmp->pitch_ = (((params->size*sizeof(float) + 255)/256)*256)/sizeof(float);

      allocRemoteBuffer((void **)&(tmp->v_), tmp->pitch_*params->count*sizeof(float));
    }
  else if (params->elementType == SPGPU_TYPE_DOUBLE)
    {
      //fprintf(stderr,"Allocating  a DOUBLE vector\n");

      if (params->count == 1)
	tmp->pitch_ = params->size;
      else
	tmp->pitch_ = (int)(((params->size*sizeof(double) + 255)/256)*256)/sizeof(double);
      allocRemoteBuffer((void **)&(tmp->v_), tmp->pitch_*tmp->count_*sizeof(double));
    }
  else if (params->elementType == SPGPU_TYPE_COMPLEX_FLOAT)
    {
      if (params->count == 1)
	tmp->pitch_ = params->size;
      else
	tmp->pitch_ = (int)(((params->size*sizeof(cuFloatComplex) + 255)/256)*256)/sizeof(cuFloatComplex);
      allocRemoteBuffer((void **)&(tmp->v_), tmp->pitch_*tmp->count_*sizeof(cuFloatComplex));
    }
  else if (params->elementType == SPGPU_TYPE_COMPLEX_DOUBLE)
    {
      if (params->count == 1)
	tmp->pitch_ = params->size;
      else
	tmp->pitch_ = (int)(((params->size*sizeof(cuDoubleComplex) + 255)/256)*256)/sizeof(cuDoubleComplex);
      allocRemoteBuffer((void **)&(tmp->v_), tmp->pitch_*tmp->count_*sizeof(cuDoubleComplex));
    }
  else
    return SPGPU_UNSUPPORTED; // Unsupported params
  return SPGPU_SUCCESS; // Success
}

void freeMultiVecDevice(void* deviceVec)
{
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // fprintf(stderr,"freeMultiVecDevice\n");
  if (devVec != NULL) {
    freeRemoteBuffer(devVec->v_);
    free(deviceVec);
  }
}

int FallocMultiVecDevice(void** deviceMultiVec, unsigned int count,
			 unsigned int size, unsigned int elementType)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectorDeviceParams p;
  if(!handle_v)
    spgpuCreate(&handle_v, 0);
  p = getMultiVectorDeviceParams(count, size, elementType);
  i = allocMultiVecDevice(deviceMultiVec, &p);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","FallocMultiVecDevice",i);
  }
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
  return(i);
#else
    return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceFloat(void* deviceVec, float* hostVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  i = readRemoteBuffer((void *) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(float));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceFloat",i);
  }*/
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readMultiVecDeviceDouble(void* deviceVec, double* hostVec)
{ int i;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  i = readRemoteBuffer((void *) hostVec, (void *)devVec->v_, devVec->pitch_*devVec->count_*sizeof(double));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceDouble",i);
  }*/
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
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  __assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuSmnrm2(handle_v,y_res,n,(float *)devVecA->v_,devVecA->count_,devVecA->pitch_);
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
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuDmnrm2(handle_v, y_res, n,(double *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int nrm2MultiVecDeviceFloatComplex(float* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuCmnrm2(handle_v, y_res, n,(cuFloatComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
  //i = nrm2MultiVecDevice((void *) y_res, n, devMultiVecA);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int nrm2MultiVecDeviceDoubleComplex(double* y_res, int n, void* devMultiVecA)
{ int i=0;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) devMultiVecA;
  //__assert(n <= devVecA->size_ , "ERROR: wrong N for norm2 ");
  //chiamata alla nuova libreria
  spgpuZmnrm2(handle_v, y_res, n,(cuDoubleComplex *)devVecA->v_, devVecA->count_, devVecA->pitch_);
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
  spgpuSmdot(handle_v, y_res, n, (float*)devVecA->v_, (float*)devVecB->v_,devVecA->count_,devVecB->pitch_);
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
  spgpuDmdot(handle_v, y_res, n, (double*)devVecA->v_, (double*)devVecB->v_,devVecA->count_,devVecB->pitch_);
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
  /*for(i=0;i<n;i++)
    res[i] = make_cuFloatComplex(crealf(y_res[i]),cimagf(y_res[i]));*/
    
  spgpuCmdot(handle_v, (cuFloatComplex *)y_res, n, (cuFloatComplex*)devVecA->v_, (cuFloatComplex*)devVecB->v_,devVecA->count_,devVecB->pitch_);
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
  spgpuZmdot(handle_v, (cuDoubleComplex*)y_res, n, (cuDoubleComplex*)devVecA->v_, (cuDoubleComplex*)devVecB->v_,devVecA->count_,devVecB->pitch_);
  return(0);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axpbyMultiVecDeviceFloat(float alpha, void* devMultiVecX, float beta, void* devMultiVecY)
{ int i=0, j=0, pitch=0;
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  pitch = devVecY->pitch_;
#ifdef HAVE_SPGPU
  for(j=0;j<devVecY->count_;j++)
    spgpuSaxpby(handle_v,(float*)devVecY->v_+pitch*j, devVecY->size_, beta, (float*)devVecY->v_+pitch*j, alpha,(float*) devVecX->v_+pitch*j);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axpbyMultiVecDeviceDouble(double alpha, void* devMultiVecX, 
			      double beta, void* devMultiVecY)
{ int i = 0, j=0;
  int pitch = 0;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  pitch = devVecY->pitch_;
  for(j=0;j<devVecY->count_;j++)
    spgpuDaxpby(handle_v,(double*)devVecY->v_+pitch*j, devVecY->size_, beta, (double*)devVecY->v_+pitch*j, alpha,(double*) devVecX->v_+pitch*j);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axpbyMultiVecDeviceFloatComplex(float complex alpha, void* devMultiVecX, 
				    float complex beta, void* devMultiVecY)
{ int i = 0, j=0;
  int pitch = 0;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  cuFloatComplex a, b;
  a = make_cuFloatComplex(crealf(alpha),cimagf(alpha));
  b = make_cuFloatComplex(crealf(beta),cimagf(beta));
  pitch = devVecY->pitch_;
  for(j=0;j<devVecY->count_;j++)
    spgpuCaxpby(handle_v,(cuFloatComplex*)devVecY->v_+pitch*j, devVecY->size_, b, (cuFloatComplex*)devVecY->v_+pitch*j, a,(cuFloatComplex*) devVecX->v_+pitch*j);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axpbyMultiVecDeviceDoubleComplex(double complex alpha, void* devMultiVecX, 
				     double complex beta, void* devMultiVecY)
{ int i = 0, j=0;
  int pitch = 0;
#ifdef HAVE_SPGPU
  struct MultiVectDevice *devVecX = (struct MultiVectDevice *) devMultiVecX;
  struct MultiVectDevice *devVecY = (struct MultiVectDevice *) devMultiVecY;
  cuDoubleComplex a, b;
  a = make_cuDoubleComplex(crealf(alpha),cimagf(alpha));
  b = make_cuDoubleComplex(crealf(beta),cimagf(beta));
  pitch = devVecY->pitch_;
  for(j=0;j<devVecY->count_;j++)
    spgpuZaxpby(handle_v,(cuDoubleComplex*)devVecY->v_+pitch*j, devVecY->size_, b, (cuDoubleComplex*)devVecY->v_+pitch*j, a,(cuDoubleComplex*) devVecX->v_+pitch*j);
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

int axyMultiVecDeviceFloat(float alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
#ifdef HAVE_SPGPU
  spgpuSmaxy(handle_v, (float*)devVecB->v_, devVecB->size_, alpha, (float*)devVecA->v_,(float *)devVecB->v_, devVecA->count_, devVecA->pitch_);
    //i = axyMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, deviceVecB);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axyMultiVecDeviceDouble(double alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
#ifdef HAVE_SPGPU
  spgpuDmaxy(handle_v, (double*)devVecB->v_, devVecB->size_, alpha, (double*)devVecA->v_, 
	     (double*)devVecB->v_, devVecA->count_, devVecA->pitch_);
  //i = axyMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, deviceVecB);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axyMultiVecDeviceFloatComplex(float complex alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
#ifdef HAVE_SPGPU
  cuFloatComplex a = make_cuFloatComplex(crealf(alpha),cimagf(alpha));
  spgpuCmaxy(handle_v, (cuFloatComplex *)devVecB->v_, devVecB->size_, a, (cuFloatComplex *)devVecA->v_, 
	     (cuFloatComplex *)devVecB->v_, devVecA->count_, devVecA->pitch_);
  //i = axyMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, deviceVecB);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axyMultiVecDeviceDoubleComplex(double complex alpha, void *deviceVecA, void *deviceVecB)
{ int i = 0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
#ifdef HAVE_SPGPU
  cuDoubleComplex a = make_cuDoubleComplex(creal(alpha),cimag(alpha));
  spgpuZmaxy(handle_v, (cuDoubleComplex *)devVecB->v_, devVecB->size_, a, (cuDoubleComplex *)devVecA->v_, 
	     (cuDoubleComplex *)devVecB->v_, devVecA->count_, devVecA->pitch_);
  //i = axyMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, deviceVecB);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axybzMultiVecDeviceFloat(float alpha, void *deviceVecA, 
			     void *deviceVecB, float beta, void *deviceVecZ)
{ int i = 0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) deviceVecZ;
#ifdef HAVE_SPGPU
  spgpuSmaxypbz(handle_v, (float*)devVecZ->v_, 1, beta, (float*)devVecZ->v_, alpha,
	       (float*) devVecA->v_, (float*) devVecB->v_, devVecB->count_,
	       devVecB->pitch_);
  //i = axybzMultiVecDevice((void *) alpha, deviceVecA, deviceVecB, (float *) beta, deviceVecZ);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axybzMultiVecDeviceDouble(double alpha, void *deviceVecA,
			      void *deviceVecB, double beta, void *deviceVecZ)
{ int i=0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) deviceVecZ;
#ifdef HAVE_SPGPU
  spgpuDmaxypbz(handle_v, (double*)devVecZ->v_, devVecZ->size_, beta, (double*)devVecZ->v_, 
	       alpha, (double*) devVecA->v_, (double*) devVecB->v_,
	       devVecB->count_, devVecB->pitch_);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axybzMultiVecDeviceFloatComplex(float complex alpha, void *deviceVecA,
				    void *deviceVecB, float complex beta, 
				    void *deviceVecZ)
{ int i=0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) deviceVecZ;
#ifdef HAVE_SPGPU
  cuFloatComplex a = make_cuFloatComplex(crealf(alpha),cimagf(alpha));
  cuFloatComplex b = make_cuFloatComplex(crealf(beta),cimagf(beta));
  spgpuCmaxypbz(handle_v, (cuFloatComplex *)devVecZ->v_, devVecZ->size_, b, (cuFloatComplex *)devVecZ->v_, 
	       a, (cuFloatComplex *) devVecA->v_, (cuFloatComplex *) devVecB->v_,
	       devVecB->count_, devVecB->pitch_);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int axybzMultiVecDeviceDoubleComplex(double complex alpha, void *deviceVecA,
				     void *deviceVecB, double complex beta, 
				     void *deviceVecZ)
{ int i=0;
  struct MultiVectDevice *devVecA = (struct MultiVectDevice *) deviceVecA;
  struct MultiVectDevice *devVecB = (struct MultiVectDevice *) deviceVecB;
  struct MultiVectDevice *devVecZ = (struct MultiVectDevice *) deviceVecZ;
#ifdef HAVE_SPGPU
  cuDoubleComplex a = make_cuDoubleComplex(creal(alpha),cimag(alpha));
  cuDoubleComplex b = make_cuDoubleComplex(creal(beta),cimag(beta));
  spgpuZmaxypbz(handle_v, (cuDoubleComplex *)devVecZ->v_, devVecZ->size_, b, (cuDoubleComplex *)devVecZ->v_, 
	       a, (cuDoubleComplex *) devVecA->v_, (cuDoubleComplex *) devVecB->v_,
	       devVecB->count_, devVecB->pitch_);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

//New gather functions single and double precision
int igathMultiVecDeviceFloat(void* deviceVec, int vectorId, int n, int* indexes, float* host_values, int firstIndex)
{
  int i=0;
  int * idx;
  float *values;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
#ifdef HAVE_SPGPU
  i = allocRemoteBuffer((void**)&idx, n*sizeof(int));
  i = allocRemoteBuffer((void**)&values, n*sizeof(float));
  i = writeRemoteBuffer((void*) indexes, (void*) idx, n*sizeof(int));
  spgpuSgath(handle_v, values, n, idx, firstIndex, (float *) devVec->v_);
  /*i = igathMultiVecDevice(deviceVec, *vectorId, *count, (void *) idx, (void*) host_values, *firstIndex);*/
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","igathMultiVecDeviceFloat",i);
  }*/
  readRemoteBuffer((void *) host_values, values, n*sizeof(float));
  freeRemoteBuffer(idx);
  freeRemoteBuffer(values);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
  
}

int igathMultiVecDeviceDouble(void* deviceVec, int vectorId, int n, int* indexes, double* host_values, int firstIndex)
{
  int i;
  int *idx;
  double *values;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
#ifdef HAVE_SPGPU
  i = allocRemoteBuffer((void**)&idx, n*sizeof(int));
  i = allocRemoteBuffer((void**)&values, n*sizeof(double));
  i = writeRemoteBuffer((void*) indexes, (void*) idx, n*sizeof(int));
  /*i = igathMultiVecDevice(deviceVec, *vectorId, *count, (void *) idx, (void*) host_values, *firstIndex);*/
  spgpuDgath(handle_v,(double *)values, n, idx, firstIndex, (double *) devVec->v_+vectorId*devVec->pitch_);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","igathMultiVecDeviceDouble",i);
  }*/
  readRemoteBuffer((void *) host_values, values, n*sizeof(double));
  freeRemoteBuffer(idx);
  freeRemoteBuffer(values);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int igathMultiVecDeviceFloatComplex(void* deviceVec, int vectorId, int n, int* indexes, float complex* host_values, int firstIndex)
{
  int i;
  int *idx;
  cuFloatComplex *values;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
#ifdef HAVE_SPGPU
  i = allocRemoteBuffer((void**)&idx, n*sizeof(int));
  i = allocRemoteBuffer((void**)&values, n*sizeof(cuFloatComplex));
  i = writeRemoteBuffer((void*) indexes, (void*) idx, n*sizeof(int));
  /*i = igathMultiVecDevice(deviceVec, *vectorId, *count, (void *) idx, (void*) host_values, *firstIndex);*/
  spgpuCgath(handle_v,(cuFloatComplex *)values, n, idx, firstIndex, (cuFloatComplex *) devVec->v_+vectorId*devVec->pitch_);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","igathMultiVecDeviceDouble",i);
  }*/
  readRemoteBuffer((void *) host_values, values, n*sizeof(cuFloatComplex));
  freeRemoteBuffer(idx);
  freeRemoteBuffer(values);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int igathMultiVecDeviceDoubleComplex(void* deviceVec, int vectorId, int n, int* indexes, double complex* host_values, int firstIndex)
{
  int i;
  int *idx;
  cuDoubleComplex *values;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
#ifdef HAVE_SPGPU
  i = allocRemoteBuffer((void**)&idx, n*sizeof(int));
  i = allocRemoteBuffer((void**)&values, n*sizeof(cuDoubleComplex));
  i = writeRemoteBuffer((void*) indexes, (void*) idx, n*sizeof(int));
  /*i = igathMultiVecDevice(deviceVec, *vectorId, *count, (void *) idx, (void*) host_values, *firstIndex);*/
  spgpuZgath(handle_v,(cuDoubleComplex *)values, n, idx, firstIndex, (cuDoubleComplex *) devVec->v_+vectorId*devVec->pitch_);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","igathMultiVecDeviceDouble",i);
  }*/
  readRemoteBuffer((void *) host_values, values, n*sizeof(cuDoubleComplex));
  freeRemoteBuffer(idx);
  freeRemoteBuffer(values);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int iscatMultiVecDeviceFloat(void* deviceVec, int vectorId, int n, int* indexes, float* host_values, int firstIndex, float beta)
{int i=0;
  int * idx;
  float *values;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
#ifdef HAVE_SPGPU
  i = allocRemoteBuffer((void**)&idx, n*sizeof(int));
  i = allocRemoteBuffer((void**)&values, n*sizeof(float));
  i = writeRemoteBuffer((void*) indexes, (void*) idx, n*sizeof(int));
  i = writeRemoteBuffer((void*) host_values, (void*) values, n*sizeof(float));
  /*i = iscatMultiVecDevice(deviceVec, *vectorId, *count, (void *) idx, 
    (void*) host_values, *firstIndex, betaPointer);*/
  spgpuSscat(handle_v, (float *) devVec->v_, n, values, idx, firstIndex, beta);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","iscatMultiVecDeviceFloat",i);
  }*/
  freeRemoteBuffer(idx);
  freeRemoteBuffer(values);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
  
}

int iscatMultiVecDeviceDouble(void* deviceVec, int vectorId, int n, int* indexes, double* host_values, int firstIndex, double beta)
{
  int i;
  int *idx;
  double *values;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
#ifdef HAVE_SPGPU
  i = allocRemoteBuffer((void**)&idx, n*sizeof(int));
  i = allocRemoteBuffer((void**)&values, n*sizeof(double));
  i = writeRemoteBuffer((void*) indexes, (void*) idx, n*sizeof(int));
  i = writeRemoteBuffer((void*) host_values, (void*) values, n*sizeof(double));
  /*i = iscatMultiVecDevice(deviceVec, *vectorId, *count, (void *) idx, 
			  (void*) host_values, *firstIndex, betaPointer);*/
  spgpuDscat(handle_v, (double*) devVec->v_+vectorId*devVec->pitch_, n, values, idx, firstIndex, beta);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","iscatMultiVecDeviceDouble",i);
  }*/
  freeRemoteBuffer(idx);
  freeRemoteBuffer(values);
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
 
}

int iscatMultiVecDeviceFloatComplex(void* deviceVec, int vectorId, int n, int* indexes, float complex* host_values, int firstIndex, float complex beta)
{
  int i;
  int *idx;
  cuFloatComplex *values;
  cuFloatComplex b = make_cuFloatComplex(crealf(beta),cimagf(beta));
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
#ifdef HAVE_SPGPU
  i = allocRemoteBuffer((void**)&idx, n*sizeof(int));
  i = allocRemoteBuffer((void**)&values, n*sizeof(cuFloatComplex));
  i = writeRemoteBuffer((void*) indexes, (void*) idx, n*sizeof(int));
  i = writeRemoteBuffer((void*) host_values, (void*) values, n*sizeof(cuFloatComplex));
  /*i = iscatMultiVecDevice(deviceVec, *vectorId, *count, (void *) idx, 
			  (void*) host_values, *firstIndex, betaPointer);*/
  spgpuCscat(handle_v, (cuFloatComplex *) devVec->v_+vectorId*devVec->pitch_, n, (cuFloatComplex *)values, idx, firstIndex, b);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","iscatMultiVecDeviceDouble",i);
  }*/
  freeRemoteBuffer(idx);
  freeRemoteBuffer(values);
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
 
}

int iscatMultiVecDeviceDoubleComplex(void* deviceVec, int vectorId, int n, int* indexes, double complex* host_values, int firstIndex, double complex beta)
{
  int i;
  int *idx;
  cuDoubleComplex *values;
  cuDoubleComplex b = make_cuDoubleComplex(creal(beta),cimag(beta));
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
#ifdef HAVE_SPGPU
  i = allocRemoteBuffer((void**)&idx, n*sizeof(int));
  i = allocRemoteBuffer((void**)&values, n*sizeof(cuDoubleComplex));
  i = writeRemoteBuffer((void*) indexes, (void*) idx, n*sizeof(int));
  i = writeRemoteBuffer((void*) host_values, (void*) values, n*sizeof(cuDoubleComplex));
  /*i = iscatMultiVecDevice(deviceVec, *vectorId, *count, (void *) idx, 
			  (void*) host_values, *firstIndex, betaPointer);*/
  spgpuZscat(handle_v, (cuDoubleComplex *) devVec->v_+vectorId*devVec->pitch_, n, (cuDoubleComplex *)values, idx, firstIndex, b);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","iscatMultiVecDeviceDouble",i);
  }*/
  freeRemoteBuffer(idx);
  freeRemoteBuffer(values);
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif 
}


#endif
