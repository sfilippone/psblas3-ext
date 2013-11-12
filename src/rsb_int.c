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
#include "rsb.h"
#include "rsb_int.h"
#if defined(HAVE_RSB)

int rsbInit()
{
  rsb_err_t errval = RSB_ERR_NO_ERROR;

  if((errval = rsb_lib_init(RSB_NULL_INIT_OPTIONS))!=RSB_ERR_NO_ERROR)
    {
      printf("Error initializing the library!\n");
      return 1;
    }
  
  return 0;
}

int rsbExit()
{
  rsb_err_t errval = RSB_ERR_NO_ERROR;

  if((errval = rsb_lib_exit(RSB_NULL_INIT_OPTIONS))!=RSB_ERR_NO_ERROR)
    {
      printf("Error finalizing the library!\n");
      return 1;
    }
  
  return 0;
}

int Rsb_double_from_coo(void **rsbMat, double *va, int *ia,int *ja,int nnz,int nr,
			int nc, int br, int bc)
{
  int i=0;
  rsb_err_t errval = RSB_ERR_NO_ERROR;
  
  for (i=0;i<nnz;i++)
    {
      ia[i] -= 1;
      ja[i] -= 1;
    }

  *rsbMat = rsb_mtx_alloc_from_coo_const(va,ia,ja,nnz,RSB_NUMERICAL_TYPE_DOUBLE,nr,nc,br,bc,RSB_FLAG_NOFLAGS,&errval);

  if((!*rsbMat) || (errval != RSB_ERR_NO_ERROR))
    {
      printf("Error while allocating the matrix!\n");
      return 1;
    }
  return 0;
}

//X is the input and y is the output
int Rsb_double_spmv(void *rsbMat, double *x, double alfa, double *y, double beta,char trans)
{
  rsb_err_t errval = RSB_ERR_NO_ERROR;

  if(trans=='N')
    errval = rsb_spmv(RSB_TRANSPOSITION_N,&alfa,(struct rsb_mtx_t *)rsbMat,x,1,&beta,y,1);
  else
    errval = rsb_spmv(RSB_TRANSPOSITION_T,&alfa,(struct rsb_mtx_t *)rsbMat,x,1,&beta,y,1);
  
  if(errval != RSB_ERR_NO_ERROR)
    {
      printf("Error performing a multiplication!\n");
      return 1;
    }
  
  return 0;
}

void freeRsbMat(void *rsbMat)
{
  rsb_mtx_free(rsbMat);
}

/* EllDeviceParams getEllDeviceParams(unsigned int rows, unsigned int maxRowSize, unsigned int columns, unsigned int elementType, unsigned int firstIndex) */
/* { */
/*   EllDeviceParams params; */

/*   if (elementType == SPGPU_TYPE_DOUBLE) */
/*     { */
/*       params.pitch = ((rows + ELL_PITCH_ALIGN_D - 1)/ELL_PITCH_ALIGN_D)*ELL_PITCH_ALIGN_D; */
/*     } */
/*   else */
/*     { */
/*       params.pitch = ((rows + ELL_PITCH_ALIGN_S - 1)/ELL_PITCH_ALIGN_S)*ELL_PITCH_ALIGN_S; */
/*     } */
/*   //For complex? */
/*   params.elementType = elementType; */
	
/*   params.rows = rows; */
/*   params.maxRowSize = maxRowSize; */
/*   params.columns = columns; */
/*   params.firstIndex = firstIndex; */

/*   //params.pitch = computeEllAllocPitch(rows); */

/*   return params; */

/* } */
/* //new */
/* int allocEllDevice(void ** remoteMatrix, EllDeviceParams* params) */
/* { */
/*   struct EllDevice *tmp = (struct EllDevice *)malloc(sizeof(struct EllDevice)); */
/*   *remoteMatrix = (void *)tmp; */
/*   tmp->rows = params->rows; */
/*   tmp->cMPitch = computeEllAllocPitch(tmp->rows); */
/*   tmp->rPPitch = tmp->cMPitch; */
/*   tmp->pitch= tmp->cMPitch; */
/*   tmp->maxRowSize = params->maxRowSize; */
/*   tmp->allocsize = (int)tmp->maxRowSize * tmp->pitch; */
/*   //tmp->allocsize = (int)params->maxRowSize * tmp->cMPitch; */
/*   allocRemoteBuffer((void **)&(tmp->rS), tmp->rows*sizeof(int)); */
/*   allocRemoteBuffer((void **)&(tmp->rP), tmp->allocsize*sizeof(int)); */
/*   tmp->columns = params->columns; */
/*   tmp->baseIndex = params->firstIndex; */

/*   if (params->elementType == SPGPU_TYPE_FLOAT) */
/*     allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(float)); */
/*   else if (params->elementType == SPGPU_TYPE_DOUBLE) */
/*     allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(double)); */
/*   else if (params->elementType == SPGPU_TYPE_COMPLEX_FLOAT) */
/*     allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(cuFloatComplex)); */
/*   else if (params->elementType == SPGPU_TYPE_COMPLEX_DOUBLE) */
/*     allocRemoteBuffer((void **)&(tmp->cM), tmp->allocsize*sizeof(cuDoubleComplex)); */
/*   else */
/*     return SPGPU_UNSUPPORTED; // Unsupported params */
/*   return SPGPU_SUCCESS; */
/* } */

/* void freeEllDevice(void* remoteMatrix) */
/* { */
/*   struct EllDevice *devMat = (struct EllDevice *) remoteMatrix;   */
/*   //fprintf(stderr,"freeEllDevice\n"); */
/*   if (devMat != NULL) { */
/*     freeRemoteBuffer(devMat->rS); */
/*     freeRemoteBuffer(devMat->rP); */
/*     freeRemoteBuffer(devMat->cM); */
/*     free(remoteMatrix); */
/*   } */
/* } */

/* //new */
/* int FallocEllDevice(void** deviceMat,unsigned int rows, unsigned int maxRowSize,  */
/* 		    unsigned int columns, unsigned int elementType,  */
/* 		    unsigned int firstIndex) */
/* { int i; */
/* #ifdef HAVE_SPGPU */
/*   EllDeviceParams p; */

/*   if(!handle) */
/*     spgpuCreate(&handle, 0); */

/*   p = getEllDeviceParams(rows, maxRowSize, columns, elementType, firstIndex); */
/*   i = allocEllDevice(deviceMat, &p); */
/*   if (i != 0) { */
/*     fprintf(stderr,"From routine : %s : %d \n","FallocEllDevice",i); */
/*   } */
/*   return(i); */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* void sspmdmm_gpu(float *z,int s, int vPitch, float *y, float alpha, float* cM, int* rP, int* rS,  */
/* 		 int n, int pitch, float *x, float beta, int firstIndex) */
/* { */
/*   int i=0; */
/*   for (i=0; i<s; i++) */
/*     { */
/*       spgpuSellspmv (handle, (float*) z, (float*)y, alpha, (float*) cM, rP, pitch, pitch, rS,  */
/* 		     NULL, n, (float*)x, beta, firstIndex); */
/*       z += vPitch; */
/*       y += vPitch; */
/*       x += vPitch;		 */
/*     } */
/* } */

/* void */
/* dspmdmm_gpu (double *z,int s, int vPitch, double *y, double alpha, double* cM, int* rP, int* rS, int n, int pitch, double *x, double beta, int firstIndex) */
/* { */
/*   int i=0; */
/*   for (i=0; i<s; i++) */
/*     { */
/*       spgpuDellspmv (handle, (double*) z, (double*)y, alpha, (double*) cM, rP, pitch, pitch, rS, NULL, n, (double*)x, beta, firstIndex); */
/*       z += vPitch; */
/*       y += vPitch; */
/*       x += vPitch;		 */
/*     } */
/* } */

/* void */
/* cspmdmm_gpu (cuFloatComplex *z, int s, int vPitch, cuFloatComplex *y, cuFloatComplex alpha, cuFloatComplex* cM, int* rP, int* rS, int n, int pitch, cuFloatComplex *x, cuFloatComplex beta, int firstIndex) */
/* { */
/*   int i=0; */
/*   for (i=0; i<s; i++) */
/*     { */
/*       spgpuCellspmv (handle, (cuFloatComplex *) z, (cuFloatComplex *)y, alpha, (cuFloatComplex *) cM, rP, pitch, pitch, rS, NULL, n, (cuFloatComplex *) x, beta, firstIndex); */
/*       z += vPitch; */
/*       y += vPitch; */
/*       x += vPitch;		 */
/*     } */
/* } */

/* void */
/* zspmdmm_gpu (cuDoubleComplex *z, int s, int vPitch, cuDoubleComplex *y, cuDoubleComplex alpha, cuDoubleComplex* cM, int* rP, int* rS, int n, int pitch, cuDoubleComplex *x, cuDoubleComplex beta, int firstIndex) */
/* { */
/*   int i=0; */
/*   for (i=0; i<s; i++) */
/*     { */
/*       spgpuZellspmv (handle, (cuDoubleComplex *) z, (cuDoubleComplex *)y, alpha, (cuDoubleComplex *) cM, rP, pitch, pitch, rS, NULL, n, (cuDoubleComplex *) x, beta, firstIndex); */
/*       z += vPitch; */
/*       y += vPitch; */
/*       x += vPitch;		 */
/*     } */
/* } */

/* //new */
/* int spmvEllDeviceFloat(void *deviceMat, float alpha, void* deviceX,  */
/* 		       float beta, void* deviceY) */
/* { int i=SPGPU_SUCCESS; */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX; */
/*   struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY;  */
/*   spgpuHandle_t handle; */

/* #ifdef HAVE_SPGPU */
/* #ifdef VERBOSE */
/*   __assert(x->count_ == x->count_, "ERROR: x and y don't share the same number of vectors"); */
/*   __assert(x->size_ >= devMat->columns, "ERROR: x vector's size is not >= to matrix size (columns)"); */
/*   __assert(y->size_ >= devMat->rows, "ERROR: y vector's size is not >= to matrix size (rows)"); */
/* #endif */
/*   /\*spgpuSellspmv (handle, (float*) y->v_, (float*)y->v_, alpha,  */
/* 		 (float*) devMat->cM, devMat->rP, devMat->cMPitch,  */
/* 		 devMat->rPPitch, devMat->rS, devMat->rows,  */
/* 		 (float*)x->v_, beta, devMat->baseIndex);*\/ */
/*   sspmdmm_gpu ( (float *)y->v_,y->count_, y->pitch_, (float *)y->v_, alpha, (float *)devMat->cM, devMat->rP, devMat->rS,  */
/* 	       devMat->rows, devMat->pitch, (float *)x->v_, beta, devMat->baseIndex); */
/*   return(i); */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* //new */
/* int spmvEllDeviceDouble(void *deviceMat, double alpha, void* deviceX,  */
/* 		       double beta, void* deviceY) */
/* { */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX; */
/*   struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY; */

/* #ifdef HAVE_SPGPU */
/*   /\*spgpuDellspmv (handle, (double*) y->v_, (double*)y->v_, alpha, (double*) devMat->cM, devMat->rP, devMat->cMPitch, devMat->rPPitch, devMat->rS, devMat->rows, (double*)x->v_, beta, devMat->baseIndex);*\/  */
/*   dspmdmm_gpu ((double *)y->v_, y->count_, y->pitch_, (double *)y->v_, alpha, (double *)devMat->cM,  */
/* 	       devMat->rP, devMat->rS, devMat->rows, devMat->pitch, (double *)x->v_, beta, */
/* 	       devMat->baseIndex); */
  
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int spmvEllDeviceFloatComplex(void *deviceMat, float complex alpha, void* deviceX, */
/* 			      float complex beta, void* deviceY) */
/* { */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX; */
/*   struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY; */

/* #ifdef HAVE_SPGPU */
/*   cuFloatComplex a = make_cuFloatComplex(crealf(alpha),cimagf(alpha)); */
/*   cuFloatComplex b = make_cuFloatComplex(crealf(beta),cimagf(beta)); */
/*   cspmdmm_gpu ((cuFloatComplex *)y->v_, y->count_, y->pitch_, (cuFloatComplex *)y->v_, a, (cuFloatComplex *)devMat->cM,  */
/* 	       devMat->rP, devMat->rS, devMat->rows, devMat->pitch, (cuFloatComplex *)x->v_, b, */
/* 	       devMat->baseIndex); */
  
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int spmvEllDeviceDoubleComplex(void *deviceMat, double complex alpha, void* deviceX, */
/* 			      double complex beta, void* deviceY) */
/* { */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX; */
/*   struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY; */

/* #ifdef HAVE_SPGPU */
/*   cuDoubleComplex a = make_cuDoubleComplex(creal(alpha),cimag(alpha)); */
/*   cuDoubleComplex b = make_cuDoubleComplex(creal(beta),cimag(beta)); */
/*   zspmdmm_gpu ((cuDoubleComplex *)y->v_, y->count_, y->pitch_, (cuDoubleComplex *)y->v_, a, (cuDoubleComplex *)devMat->cM,  */
/* 	       devMat->rP, devMat->rS, devMat->rows, devMat->pitch, (cuDoubleComplex *)x->v_, b, */
/* 	       devMat->baseIndex); */
  
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int writeEllDeviceFloat(void* deviceMat, float* val, int* ja, int ldj, int* irn) */
/* { int i; */
/* #ifdef HAVE_SPGPU */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   // Ex updateFromHost function */
/*   i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(float)); */
/*   i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int)); */
/*   i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int)); */
/*   //i = writeEllDevice(deviceMat, (void *) val, ja, irn); */
/*   /\*if (i != 0) { */
/*     fprintf(stderr,"From routine : %s : %d \n","writeEllDeviceFloat",i); */
/*   }*\/ */
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int writeEllDeviceDouble(void* deviceMat, double* val, int* ja, int ldj, int* irn) */
/* { int i; */
/* #ifdef HAVE_SPGPU */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   // Ex updateFromHost function */
/*   i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(double)); */
/*   i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int)); */
/*   i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int)); */

/*   /\*i = writeEllDevice(deviceMat, (void *) val, ja, irn); */
/*   if (i != 0) { */
/*     fprintf(stderr,"From routine : %s : %d \n","writeEllDeviceDouble",i); */
/*   }*\/ */
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int writeEllDeviceFloatComplex(void* deviceMat, float complex* val, int* ja, int ldj, int* irn) */
/* { int i; */
/* #ifdef HAVE_SPGPU */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   // Ex updateFromHost function */
/*   i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuFloatComplex)); */
/*   i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int)); */
/*   i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int)); */

/*   /\*i = writeEllDevice(deviceMat, (void *) val, ja, irn); */
/*   if (i != 0) { */
/*     fprintf(stderr,"From routine : %s : %d \n","writeEllDeviceDouble",i); */
/*   }*\/ */
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int writeEllDeviceDoubleComplex(void* deviceMat, double complex* val, int* ja, int ldj, int* irn) */
/* { int i; */
/* #ifdef HAVE_SPGPU */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   // Ex updateFromHost function */
/*   i = writeRemoteBuffer((void*) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuDoubleComplex)); */
/*   i = writeRemoteBuffer((void*) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int)); */
/*   i = writeRemoteBuffer((void*) irn, (void *)devMat->rS, devMat->rows*sizeof(int)); */

/*   /\*i = writeEllDevice(deviceMat, (void *) val, ja, irn); */
/*   if (i != 0) { */
/*     fprintf(stderr,"From routine : %s : %d \n","writeEllDeviceDouble",i); */
/*   }*\/ */
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int readEllDeviceFloat(void* deviceMat, float* val, int* ja, int ldj, int* irn) */
/* { int i; */
/* #ifdef HAVE_SPGPU */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(float)); */
/*   i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int)); */
/*   i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int)); */
/*   /\*i = readEllDevice(deviceMat, (void *) val, ja, irn); */
/*   if (i != 0) { */
/*     fprintf(stderr,"From routine : %s : %d \n","readEllDeviceFloat",i); */
/*   }*\/ */
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int readEllDeviceDouble(void* deviceMat, double* val, int* ja, int ldj, int* irn) */
/* { int i; */
/* #ifdef HAVE_SPGPU */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(double)); */
/*   i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int)); */
/*   i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int)); */
/*   /\*if (i != 0) { */
/*     fprintf(stderr,"From routine : %s : %d \n","readEllDeviceDouble",i); */
/*   }*\/ */
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int readEllDeviceFloatComplex(void* deviceMat, float complex* val, int* ja, int ldj, int* irn) */
/* { int i; */
/* #ifdef HAVE_SPGPU */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuFloatComplex)); */
/*   i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int)); */
/*   i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int)); */
/*   /\*if (i != 0) { */
/*     fprintf(stderr,"From routine : %s : %d \n","readEllDeviceDouble",i); */
/*   }*\/ */
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int readEllDeviceDoubleComplex(void* deviceMat, double complex* val, int* ja, int ldj, int* irn) */
/* { int i; */
/* #ifdef HAVE_SPGPU */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/*   i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuDoubleComplex)); */
/*   i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int)); */
/*   i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int)); */
/*   /\*if (i != 0) { */
/*     fprintf(stderr,"From routine : %s : %d \n","readEllDeviceDouble",i); */
/*   }*\/ */
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int getEllDevicePitch(void* deviceMat) */
/* { int i; */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/* #ifdef HAVE_SPGPU */
/*   i = devMat->pitch; //old */
/*   //i = getPitchEllDevice(deviceMat); */
/*   return(i); */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

/* int getEllDeviceMaxRowSize(void* deviceMat) */
/* { int i; */
/*   struct EllDevice *devMat = (struct EllDevice *) deviceMat; */
/* #ifdef HAVE_SPGPU */
/*   i = devMat->maxRowSize; */
/*   return(i); */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */
#endif
