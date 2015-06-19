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
 
#include <sys/time.h>
#include "dnsdev.h"

#if defined(HAVE_SPGPU)

#define PASS_RS  0

DnsDeviceParams getDnsDeviceParams(unsigned int rows, unsigned int maxRowSize,
				   unsigned int nnzeros, 
				   unsigned int columns, unsigned int elementType,
				   unsigned int firstIndex)
{
  DnsDeviceParams params;

  if (elementType == SPGPU_TYPE_DOUBLE)
    {
      params.pitch = ((rows + ELL_PITCH_ALIGN_D - 1)/ELL_PITCH_ALIGN_D)*ELL_PITCH_ALIGN_D;
    }
  else
    {
      params.pitch = ((rows + ELL_PITCH_ALIGN_S - 1)/ELL_PITCH_ALIGN_S)*ELL_PITCH_ALIGN_S;
    }
  //For complex?
  params.elementType = elementType;
  params.rows = rows;
  params.columns = columns;
  params.firstIndex = firstIndex;

  //params.pitch = computeDnsAllocPitch(rows);

  return params;

}
//new
int allocDnsDevice(void ** remoteMatrix, DnsDeviceParams* params)
{
  struct DnsDevice *tmp = (struct DnsDevice *)malloc(sizeof(struct DnsDevice));
  *remoteMatrix = (void *)tmp;
  tmp->rows = params->rows;
  tmp->columns = params->columns;
  tmp->cMPitch = params->pitch;
  tmp->pitch= tmp->cMPitch;
  tmp->allocsize = (int)tmp->columns * tmp->pitch;
  tmp->baseIndex = params->firstIndex;
  //fprintf(stderr,"allocDnsDevice: %d %d %d \n",tmp->pitch, params->maxRowSize, params->avgRowSize);
  if (params->elementType == SPGPU_TYPE_FLOAT)
    allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(float));
  else if (params->elementType == SPGPU_TYPE_DOUBLE)
    allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(double));
  else if (params->elementType == SPGPU_TYPE_COMPLEX_FLOAT)
    allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(cuFloatComplex));
  else if (params->elementType == SPGPU_TYPE_COMPLEX_DOUBLE)
    allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(cuDoubleComplex));
  else
    return SPGPU_UNSUPPORTED; // Unsupported params
  //fprintf(stderr,"From allocDnsDevice: %d %d %d %p %p %p\n",tmp->maxRowSize,
  //	  tmp->avgRowSize,tmp->allocsize,tmp->rS,tmp->rP,tmp->cM);

  return SPGPU_SUCCESS;
}

void freeDnsDevice(void* remoteMatrix)
{
  struct DnsDevice *devMat = (struct DnsDevice *) remoteMatrix;  
  //fprintf(stderr,"freeDnsDevice\n");
  if (devMat != NULL) {
    freeRemoteBuffer(devMat->cM);
    free(remoteMatrix);
  }
}

//new
int FallocDnsDevice(void** deviceMat, unsigned int rows, unsigned int maxRowSize, 
		    unsigned int nnzeros,
		    unsigned int columns, unsigned int elementType, 
		    unsigned int firstIndex)
{ int i;
#ifdef HAVE_SPGPU
  DnsDeviceParams p;

  p = getDnsDeviceParams(rows, maxRowSize, nnzeros, columns, elementType, firstIndex);
  i = allocDnsDevice(deviceMat, &p);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","FallocDnsDevice",i);
  }
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

#if 0 
void sspmdmm_gpu(float *z,int s, int vPitch, float *y, float alpha, float* cM, int* rP, int* rS, 
		 int avgRowSize, int maxRowSize, int rows, int pitch, float *x, float beta, int firstIndex)
{
  int i=0;
  cublasHandle_t handle=psb_gpuGetCublasHandle();

  for (i=0; i<s; i++)
    {
      if (PASS_RS) {
	spgpuSdnsspmv (handle, (float*) z, (float*)y, alpha, (float*) cM, rP, pitch, pitch, rS, 
		       NULL, avgRowSize, maxRowSize, rows, (float*)x, beta, firstIndex);
      } else {
	spgpuSdnsspmv (handle, (float*) z, (float*)y, alpha, (float*) cM, rP, pitch, pitch, NULL, 
		       NULL, avgRowSize, maxRowSize, rows, (float*)x, beta, firstIndex);
      }
      z += vPitch;
      y += vPitch;
      x += vPitch;		
    }
}
//new
int spmvDnsDeviceFloat(void *deviceMat, float alpha, void* deviceX, 
		       float beta, void* deviceY)
{ int i=SPGPU_SUCCESS;
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY; 

#ifdef HAVE_SPGPU
#ifdef VERBOSE
  __assert(x->count_ == x->count_, "ERROR: x and y don't share the same number of vectors");
  __assert(x->size_ >= devMat->columns, "ERROR: x vector's size is not >= to matrix size (columns)");
  __assert(y->size_ >= devMat->rows, "ERROR: y vector's size is not >= to matrix size (rows)");
#endif
  /*spgpuSdnsspmv (handle, (float*) y->v_, (float*)y->v_, alpha, 
		 (float*) devMat->cM, devMat->rP, devMat->cMPitch, 
		 devMat->rPPitch, devMat->rS, devMat->rows, 
		 (float*)x->v_, beta, devMat->baseIndex);*/
  sspmdmm_gpu ( (float *)y->v_,y->count_, y->pitch_, (float *)y->v_, alpha, (float *)devMat->cM, devMat->rP, devMat->rS, 
		devMat->avgRowSize, devMat->maxRowSize, devMat->rows, devMat->pitch,
		(float *)x->v_, beta, devMat->baseIndex);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}
#endif

void
dspmdmm_gpu (double *z,int s, int vPitch, double *y, double alpha, double* cM, int* rP,
	     int* rS, int avgRowSize, int maxRowSize, int rows, int pitch, 
	     double *x, double beta, int firstIndex)
{
  int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  for (i=0; i<s; i++)
    {
      if (PASS_RS) {
	spgpuDdnsspmv (handle, (double*) z, (double*)y, alpha, (double*) cM, rP,
		       pitch, pitch, rS,
		       NULL,  avgRowSize, maxRowSize, rows, (double*)x, beta, firstIndex);
      } else {
	spgpuDdnsspmv (handle, (double*) z, (double*)y, alpha, (double*) cM, rP,
		       pitch, pitch, NULL,
		       NULL,  avgRowSize, maxRowSize, rows, (double*)x, beta, firstIndex);
      } 
      z += vPitch;
      y += vPitch;
      x += vPitch;		
    }
}

//new
int spmvDnsDeviceDouble(void *deviceMat, double alpha, void* deviceX, 
		       double beta, void* deviceY)
{
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY;

#ifdef HAVE_SPGPU
  /*spgpuDdnsspmv (handle, (double*) y->v_, (double*)y->v_, alpha, (double*) devMat->cM, devMat->rP, devMat->cMPitch, devMat->rPPitch, devMat->rS, devMat->rows, (double*)x->v_, beta, devMat->baseIndex);*/
  /* fprintf(stderr,"From spmvDnsDouble: mat %d %d %d %d y %d %d \n", */
  /* 	  devMat->avgRowSize, devMat->maxRowSize, devMat->rows, */
  /* 	  devMat->pitch, y->count_, y->pitch_); */
  #if 0 
  dspmdmm_gpu ((double *)y->v_, y->count_, y->pitch_, (double *)y->v_,
	       alpha, (double *)devMat->cM, 
	       devMat->rP, devMat->rS, devMat->avgRowSize,
	       devMat->maxRowSize, devMat->rows, devMat->pitch,
	       (double *)x->v_, beta, devMat->baseIndex);
  #endif
  
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

#if 0 
void
cspmdmm_gpu (cuFloatComplex *z, int s, int vPitch, cuFloatComplex *y,  
	     cuFloatComplex alpha, cuFloatComplex* cM,
	     int* rP, int* rS, int avgRowSize, int maxRowSize, int rows, int pitch,
	     cuFloatComplex *x, cuFloatComplex beta, int firstIndex)
{
  int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  for (i=0; i<s; i++)
    {
      if (PASS_RS) {
	spgpuCdnsspmv (handle, (cuFloatComplex *) z, (cuFloatComplex *)y, alpha, (cuFloatComplex *) cM, rP, 
		       pitch, pitch, rS, NULL, avgRowSize, maxRowSize, rows, (cuFloatComplex *) x, beta, firstIndex);
      } else {
	spgpuCdnsspmv (handle, (cuFloatComplex *) z, (cuFloatComplex *)y, alpha, (cuFloatComplex *) cM, rP, 
		       pitch, pitch, NULL, NULL, avgRowSize, maxRowSize, rows, (cuFloatComplex *) x, beta, firstIndex);
      }
      z += vPitch;
      y += vPitch;
      x += vPitch;		
    }
}

int spmvDnsDeviceFloatComplex(void *deviceMat, float complex alpha, void* deviceX,
			      float complex beta, void* deviceY)
{
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY;

#ifdef HAVE_SPGPU
  cuFloatComplex a = make_cuFloatComplex(crealf(alpha),cimagf(alpha));
  cuFloatComplex b = make_cuFloatComplex(crealf(beta),cimagf(beta));
  cspmdmm_gpu ((cuFloatComplex *)y->v_, y->count_, y->pitch_, (cuFloatComplex *)y->v_, a, (cuFloatComplex *)devMat->cM, 
	       devMat->rP, devMat->rS, devMat->avgRowSize, devMat->maxRowSize, devMat->rows, devMat->pitch,
	       (cuFloatComplex *)x->v_, b, devMat->baseIndex);
  
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

void
zspmdmm_gpu (cuDoubleComplex *z, int s, int vPitch, cuDoubleComplex *y, cuDoubleComplex alpha, cuDoubleComplex* cM,
	     int* rP, int* rS, int avgRowSize, int maxRowSize, int rows, int pitch,
	     cuDoubleComplex *x, cuDoubleComplex beta, int firstIndex)
{
  int i=0;
  spgpuHandle_t handle=psb_gpuGetHandle();
  for (i=0; i<s; i++)
    {
      if (PASS_RS) {
	spgpuZdnsspmv (handle, (cuDoubleComplex *) z, (cuDoubleComplex *)y, alpha, (cuDoubleComplex *) cM, rP, 
		       pitch, pitch, rS, NULL,  avgRowSize, maxRowSize, rows, (cuDoubleComplex *) x, beta, firstIndex);
      } else {
	spgpuZdnsspmv (handle, (cuDoubleComplex *) z, (cuDoubleComplex *)y, alpha, (cuDoubleComplex *) cM, rP, 
		       pitch, pitch, NULL, NULL,  avgRowSize, maxRowSize, rows, (cuDoubleComplex *) x, beta, firstIndex);
      }
      z += vPitch;
      y += vPitch;
      x += vPitch;		
    }
}

int spmvDnsDeviceDoubleComplex(void *deviceMat, double complex alpha, void* deviceX,
			      double complex beta, void* deviceY)
{
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY;

#ifdef HAVE_SPGPU
  cuDoubleComplex a = make_cuDoubleComplex(creal(alpha),cimag(alpha));
  cuDoubleComplex b = make_cuDoubleComplex(creal(beta),cimag(beta));
  zspmdmm_gpu ((cuDoubleComplex *)y->v_, y->count_, y->pitch_, (cuDoubleComplex *)y->v_, a, (cuDoubleComplex *)devMat->cM, 
	       devMat->rP, devMat->rS, devMat->avgRowSize, devMat->maxRowSize, devMat->rows,
	       devMat->pitch, (cuDoubleComplex *)x->v_, b, devMat->baseIndex);
  
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int writeDnsDeviceFloat(void* deviceMat, float* val, int* ja, int ldj, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  // Ex updateFromHost function
  i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(float));
  i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  //i = writeDnsDevice(deviceMat, (void *) val, ja, irn);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeDnsDeviceFloat",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}
#endif

int writeDnsDeviceDouble(void* deviceMat, double* val, int lda, int nc)
{ int i;
#ifdef HAVE_SPGPU
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  int pitch=devMat->pitch; 
  // Ex updateFromHost function
  // i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(double));
  i = cublasSetMatrix(lda,nc,sizeof(double), (void*) val,lda, (void *)devMat->cM, pitch);
  /*i = writeDnsDevice(deviceMat, (void *) val, ja, irn);*/
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeDnsDeviceDouble",i);
  }
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

#if 0 
int writeDnsDeviceFloatComplex(void* deviceMat, float complex* val, int* ja, int ldj, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  // Ex updateFromHost function
  i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuFloatComplex));
  i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int));

  /*i = writeDnsDevice(deviceMat, (void *) val, ja, irn);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeDnsDeviceDouble",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int writeDnsDeviceDoubleComplex(void* deviceMat, double complex* val, int* ja, int ldj, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  // Ex updateFromHost function
  i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuDoubleComplex));
  i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int));

  /*i = writeDnsDevice(deviceMat, (void *) val, ja, irn);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeDnsDeviceDouble",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readDnsDeviceFloat(void* deviceMat, float* val, int* ja, int ldj, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(float));
  i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  /*i = readDnsDevice(deviceMat, (void *) val, ja, irn);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readDnsDeviceFloat",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}
#endif


int readDnsDeviceDouble(void* deviceMat, double* val, int lda, int nc)
{ int i;
#ifdef HAVE_SPGPU
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  int pitch=devMat->pitch; 
  // Ex updateFromHost function
  // i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(double));
  i = cublasGetMatrix(lda,nc,sizeof(double), (void *)devMat->cM, pitch, (void*) val, lda);
  /*i = writeDnsDevice(deviceMat, (void *) val, ja, irn);*/
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readDnsDeviceDouble",i);
  }
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

#if 0
int readDnsDeviceFloatComplex(void* deviceMat, float complex* val, int* ja, int ldj, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuFloatComplex));
  i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readDnsDeviceDouble",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readDnsDeviceDoubleComplex(void* deviceMat, double complex* val, int* ja, int ldj, int* irn)
{ int i;
#ifdef HAVE_SPGPU
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
  i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuDoubleComplex));
  i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int));
  i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readDnsDeviceDouble",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}
#endif
int getDnsDevicePitch(void* deviceMat)
{ int i;
  struct DnsDevice *devMat = (struct DnsDevice *) deviceMat;
#ifdef HAVE_SPGPU
  i = devMat->pitch; //old
  //i = getPitchDnsDevice(deviceMat);
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}



#endif

