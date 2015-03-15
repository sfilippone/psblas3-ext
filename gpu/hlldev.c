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
 

#include "hlldev.h"
#if defined(HAVE_SPGPU)
//new
HllDeviceParams getHllDeviceParams(unsigned int hksize, unsigned int rows, unsigned int allocsize, unsigned int elementType, unsigned int firstIndex)
{
  HllDeviceParams params;

  params.elementType = elementType;
  params.hackSize = hksize;
  //numero di elementi di val
  params.allocsize = allocsize;
  params.rows = rows;
  params.firstIndex = firstIndex;

  return params;

}
//new
int allocHllDevice(void ** remoteMatrix, HllDeviceParams* params)
{
  struct HllDevice *tmp = (struct HllDevice *)malloc(sizeof(struct HllDevice));
  int ret=SPGPU_SUCCESS;
  *remoteMatrix = (void *)tmp;

  tmp->hackSize = params->hackSize;

  tmp->allocsize = params->allocsize;

  tmp->rows = params->rows;

  tmp->hackOffsLength = (int)(tmp->rows+tmp->hackSize-1)/tmp->hackSize;

  //printf("hackOffsLength %d\n",tmp->hackOffsLength);
 
  if (ret == SPGPU_SUCCESS)
    ret=allocRemoteBuffer((void **)&(tmp->rP), tmp->allocsize*sizeof(int));
  
  if (ret == SPGPU_SUCCESS)
    ret=allocRemoteBuffer((void **)&(tmp->rS), tmp->rows*sizeof(int));

  if (ret == SPGPU_SUCCESS)
    ret=allocRemoteBuffer((void **)&(tmp->hackOffs), (tmp->hackOffsLength*sizeof(int)));
  
  tmp->baseIndex = params->firstIndex;

  if (params->elementType == SPGPU_TYPE_INT)
    {
      if (ret == SPGPU_SUCCESS)
	ret=allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(int));
    }
  else if (params->elementType == SPGPU_TYPE_FLOAT)
    {
      if (ret == SPGPU_SUCCESS)
	ret=allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(float));
    }    
  else if (params->elementType == SPGPU_TYPE_DOUBLE)
    {
      if (ret == SPGPU_SUCCESS)
	ret=allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(double));
    }
  else if (params->elementType == SPGPU_TYPE_COMPLEX_FLOAT)
    {
      if (ret == SPGPU_SUCCESS)
	ret=allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(cuFloatComplex));
    }
  else if (params->elementType == SPGPU_TYPE_COMPLEX_DOUBLE)
    {
      if (ret == SPGPU_SUCCESS)
	ret=allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(cuDoubleComplex));
    }
  else
    return SPGPU_UNSUPPORTED; // Unsupported params
  return ret;
}

void freeHllDevice(void* remoteMatrix)
{
  struct HllDevice *devMat = (struct HllDevice *) remoteMatrix;  
  //fprintf(stderr,"freeHllDevice\n");
  if (devMat != NULL) {
    freeRemoteBuffer(devMat->rS);
    freeRemoteBuffer(devMat->rP);
    freeRemoteBuffer(devMat->cM);
    free(remoteMatrix);
  }
}

//new
int FallocHllDevice(void** deviceMat,unsigned int hksize, unsigned int rows, unsigned int allocsize, unsigned int elementType, unsigned int firstIndex)
{ int i;
#ifdef HAVE_SPGPU
  HllDeviceParams p;

  p = getHllDeviceParams(hksize, rows, allocsize, elementType, firstIndex);
  i = allocHllDevice(deviceMat, &p);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","FallocEllDevice",i);
  }
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}


int spmvHllDeviceFloat(void *deviceMat, float alpha, void* deviceX, 
			float beta, void* deviceY)
{
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY;
  spgpuHandle_t handle=psb_gpuGetHandle();

#ifdef HAVE_SPGPU
#ifdef VERBOSE
  /*__assert(x->count_ == x->count_, "ERROR: x and y don't share the same number of vectors");*/
  /*__assert(x->size_ >= devMat->columns, "ERROR: x vector's size is not >= to matrix size (columns)");*/
  /*__assert(y->size_ >= devMat->rows, "ERROR: y vector's size is not >= to matrix size (rows)");*/
#endif
  /*dspmdmm_gpu ((double *)z->v_, y->count_, y->pitch_, (double *)y->v_, alpha, (double *)devMat->cM, 
	       devMat->rP, devMat->rS, devMat->rows, devMat->pitch, (double *)x->v_, beta,
	       devMat->baseIndex);*/

  spgpuShellspmv (handle, (float *)y->v_, (float *)y->v_, alpha, (float *)devMat->cM, 
		  devMat->rP,devMat->hackSize,devMat->hackOffs, devMat->rS, NULL,
		  0, devMat->rows, (float *)x->v_, beta, devMat->baseIndex);

  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

//new
int spmvHllDeviceDouble(void *deviceMat, double alpha, void* deviceX, 
		       double beta, void* deviceY)
{
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY;
  spgpuHandle_t handle=psb_gpuGetHandle();

#ifdef HAVE_SPGPU
#ifdef VERBOSE
  /*__assert(x->count_ == x->count_, "ERROR: x and y don't share the same number of vectors");*/
  /*__assert(x->size_ >= devMat->columns, "ERROR: x vector's size is not >= to matrix size (columns)");*/
  /*__assert(y->size_ >= devMat->rows, "ERROR: y vector's size is not >= to matrix size (rows)");*/
#endif
  /*dspmdmm_gpu ((double *)z->v_, y->count_, y->pitch_, (double *)y->v_, alpha, (double *)devMat->cM, 
	       devMat->rP, devMat->rS, devMat->rows, devMat->pitch, (double *)x->v_, beta,
	       devMat->baseIndex);*/

  spgpuDhellspmv (handle, (double *)y->v_, (double *)y->v_, alpha, (double*)devMat->cM, 
		  devMat->rP,devMat->hackSize,devMat->hackOffs, devMat->rS, NULL,
		  0, devMat->rows, (double *)x->v_, beta, devMat->baseIndex);
  //cudaSync();
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int spmvHllDeviceFloatComplex(void *deviceMat, float complex alpha, void* deviceX, 
		       float complex beta, void* deviceY)
{
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY;
  spgpuHandle_t handle=psb_gpuGetHandle();

#ifdef HAVE_SPGPU
  cuFloatComplex a = make_cuFloatComplex(crealf(alpha),cimagf(alpha));
  cuFloatComplex b = make_cuFloatComplex(crealf(beta),cimagf(beta));
#ifdef VERBOSE
  /*__assert(x->count_ == x->count_, "ERROR: x and y don't share the same number of vectors");*/
  /*__assert(x->size_ >= devMat->columns, "ERROR: x vector's size is not >= to matrix size (columns)");*/
  /*__assert(y->size_ >= devMat->rows, "ERROR: y vector's size is not >= to matrix size (rows)");*/
#endif
  /*dspmdmm_gpu ((double *)z->v_, y->count_, y->pitch_, (double *)y->v_, alpha, (double *)devMat->cM, 
	       devMat->rP, devMat->rS, devMat->rows, devMat->pitch, (double *)x->v_, beta,
	       devMat->baseIndex);*/

  spgpuChellspmv (handle, (cuFloatComplex *)y->v_, (cuFloatComplex *)y->v_, a, (cuFloatComplex *)devMat->cM, 
		  devMat->rP,devMat->hackSize,devMat->hackOffs, devMat->rS, NULL,
		  0, devMat->rows, (cuFloatComplex *)x->v_, b, devMat->baseIndex);

  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int spmvHllDeviceDoubleComplex(void *deviceMat, double complex alpha, void* deviceX, 
		       double complex beta, void* deviceY)
{
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY;
  spgpuHandle_t handle=psb_gpuGetHandle();

#ifdef HAVE_SPGPU
  cuDoubleComplex a = make_cuDoubleComplex(creal(alpha),cimag(alpha));
  cuDoubleComplex b = make_cuDoubleComplex(creal(beta),cimag(beta));
#ifdef VERBOSE
  /*__assert(x->count_ == x->count_, "ERROR: x and y don't share the same number of vectors");*/
  /*__assert(x->size_ >= devMat->columns, "ERROR: x vector's size is not >= to matrix size (columns)");*/
  /*__assert(y->size_ >= devMat->rows, "ERROR: y vector's size is not >= to matrix size (rows)");*/
#endif

  spgpuZhellspmv (handle, (cuDoubleComplex *)y->v_, (cuDoubleComplex *)y->v_, a, (cuDoubleComplex *)devMat->cM, 
		  devMat->rP,devMat->hackSize,devMat->hackOffs, devMat->rS, NULL,
		  0,devMat->rows, (cuDoubleComplex *)x->v_, b, devMat->baseIndex);

  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int writeHllDeviceFloat(void* deviceMat, float* val, int* ja, int *hkoffs, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  // Ex updateFromHost function
  i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(float));
  i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  i = writeRemoteBuffer((void*) hkoffs, (void *)devMat->hackOffs, devMat->hackOffsLength*sizeof(int));
  //i = writeEllDevice(deviceMat, (void *) val, ja, irn);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeEllDeviceFloat",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int writeHllDeviceDouble(void* deviceMat, double* val, int* ja, int *hkoffs, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  // Ex updateFromHost function
  i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(double));
  i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  i = writeRemoteBuffer((void*) hkoffs, (void *)devMat->hackOffs, devMat->hackOffsLength*sizeof(int));
  /*i = writeEllDevice(deviceMat, (void *) val, ja, irn);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeEllDeviceDouble",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int writeHllDeviceFloatComplex(void* deviceMat, float complex* val, int* ja, int *hkoffs, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  // Ex updateFromHost function
  i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuFloatComplex));
  i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  i = writeRemoteBuffer((void*) hkoffs, (void *)devMat->hackOffs, devMat->hackOffsLength*sizeof(int));
  /*i = writeEllDevice(deviceMat, (void *) val, ja, irn);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeEllDeviceDouble",i); 
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int writeHllDeviceDoubleComplex(void* deviceMat, double complex* val, int* ja, int *hkoffs, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  // Ex updateFromHost function
  i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuDoubleComplex));
  i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  i = writeRemoteBuffer((void*) hkoffs, (void *)devMat->hackOffs, devMat->hackOffsLength*sizeof(int));
  /*i = writeEllDevice(deviceMat, (void *) val, ja, irn);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeEllDeviceDouble",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readHllDeviceFloat(void* deviceMat, float* val, int* ja, int *hkoffs, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(float));
  i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  i = readRemoteBuffer((void*) hkoffs, (void *)devMat->hackOffs, devMat->hackOffsLength*sizeof(int));
  /*i = readEllDevice(deviceMat, (void *) val, ja, irn);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readEllDeviceFloat",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readHllDeviceDouble(void* deviceMat, double* val, int* ja, int *hkoffs, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(double));
  i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  i = readRemoteBuffer((void*) hkoffs, (void *)devMat->hackOffs, devMat->hackOffsLength*sizeof(int));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readEllDeviceDouble",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readHllDeviceFloatComplex(void* deviceMat, float complex* val, int* ja, int *hkoffs, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuFloatComplex));
  i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  i = readRemoteBuffer((void*) hkoffs, (void *)devMat->hackOffs, devMat->hackOffsLength*sizeof(int));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readEllDeviceDouble",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readHllDeviceDoubleComplex(void* deviceMat, double complex* val, int* ja, int *hkoffs, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct HllDevice *devMat = (struct HllDevice *) deviceMat;
  i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuDoubleComplex));
  i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  i = readRemoteBuffer((void*) hkoffs, (void *)devMat->hackOffs, devMat->hackOffsLength*sizeof(int));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readEllDeviceDouble",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

#endif
