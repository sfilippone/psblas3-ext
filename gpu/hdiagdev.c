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
 

#include "hdiagdev.h"
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#if defined(HAVE_SPGPU)
//new

HdiagDeviceParams getHdiagDeviceParams(unsigned int rows, unsigned int columns, unsigned int diags, unsigned int hackSize, unsigned int elementType)
{
  HdiagDeviceParams params;

  params.elementType = elementType;
  //numero di elementi di val
  params.rows = rows;
  params.columns = columns; 
  params.diags = diags;
  params.hackSize = hackSize;

  return params;

}
//new
int allocHdiagDevice(void ** remoteMatrix, HdiagDeviceParams* params, double *data)
{
  struct HdiagDevice *tmp = (struct HdiagDevice *)malloc(sizeof(struct HdiagDevice));
  int ret=SPGPU_SUCCESS;
  int *tmpOff = NULL;
  *remoteMatrix = (void *)tmp;

  tmp->rows = params->rows;

  tmp->hackSize = params->hackSize;

  tmp->cols = params->columns;

  tmp->diags = params->diags;

  tmp->hackCount = getHdiaHacksCount(tmp->hackSize, tmp->rows);

  tmp->hackOffsets = (int *)calloc(tmp->hackCount+1,sizeof(int));

  computeHdiaHackOffsets(
			 &tmp->allocationHeight,
			 tmp->hackOffsets,
			 tmp->hackSize,
			 data,
			 tmp->rows,
			 tmp->diags,
			 tmp->rows,
			 params->elementType);
 
  if (ret == SPGPU_SUCCESS)
    ret=allocRemoteBuffer((void **)&(tmp->hdiaOffsets), tmp->allocationHeight*sizeof(int));
  
  /* tmp->baseIndex = params->firstIndex; */

  if (params->elementType == SPGPU_TYPE_INT)
    {
      if (ret == SPGPU_SUCCESS)
	ret=allocRemoteBuffer((void **)&(tmp->cM), tmp->hackSize*tmp->allocationHeight*sizeof(int));
    }
  else if (params->elementType == SPGPU_TYPE_FLOAT)
    {
      if (ret == SPGPU_SUCCESS)
	ret=allocRemoteBuffer((void **)&(tmp->cM), tmp->hackSize*tmp->allocationHeight*sizeof(float));
    }    
  else if (params->elementType == SPGPU_TYPE_DOUBLE)
    {
      if (ret == SPGPU_SUCCESS)
	ret=allocRemoteBuffer((void **)&(tmp->cM), tmp->hackSize*tmp->allocationHeight*sizeof(double));
    }
  else if (params->elementType == SPGPU_TYPE_COMPLEX_FLOAT)
    {
      if (ret == SPGPU_SUCCESS)
	ret=allocRemoteBuffer((void **)&(tmp->cM), tmp->hackSize*tmp->allocationHeight*sizeof(cuFloatComplex));
    }
  else if (params->elementType == SPGPU_TYPE_COMPLEX_DOUBLE)
    {
      if (ret == SPGPU_SUCCESS)
	ret=allocRemoteBuffer((void **)&(tmp->cM), tmp->hackSize*tmp->allocationHeight*sizeof(cuDoubleComplex));
    }
  else
    return SPGPU_UNSUPPORTED; // Unsupported params
  return ret;
}

void freeHdiagDevice(void* remoteMatrix)
{
  struct HdiagDevice *devMat = (struct HdiagDevice *) remoteMatrix;  
  //fprintf(stderr,"freeHllDevice\n");
  if (devMat != NULL) {
    freeRemoteBuffer(devMat->hackOffsets);
    freeRemoteBuffer(devMat->cM);
    free(remoteMatrix);
  }
}

//new
int FallocHdiagDevice(void** deviceMat, unsigned int rows, unsigned int columns,unsigned int diags,unsigned int hackSize,double *data,unsigned int elementType)
{ int i;
#ifdef HAVE_SPGPU
  HdiagDeviceParams p;

  p = getHdiagDeviceParams(rows, columns, diags, hackSize, elementType);
  i = allocHdiagDevice(deviceMat, &p, data);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","FallocEllDevice",i);
  }
  return(i);
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int writeHdiagDeviceDouble(void* deviceMat, double* a, int* off, int n)
{ int i=0,fo,fa;
  int *hoff=NULL,*hackoff=NULL;
  double *values=NULL;
  char buf_a[255], buf_o[255],tmp[255];
#ifdef HAVE_SPGPU
  struct HdiagDevice *devMat = (struct HdiagDevice *) deviceMat;
  // Ex updateFromHost function
  /* memset(buf_a,'\0',255); */
  /* memset(buf_o,'\0',255); */
  /* memset(tmp,'\0',255); */

  /* strcat(buf_a,"mat_"); */
  /* strcat(buf_o,"off_"); */
  /* sprintf(tmp,"%d_%d.dat",devMat->rows,devMat->cols); */
  /* strcat(buf_a,tmp); */
  /* memset(tmp,'\0',255); */
  /* sprintf(tmp,"%d.dat",devMat->cols); */
  /* strcat(buf_o,tmp); */

  /* fa = open(buf_a, O_CREAT | O_WRONLY | O_TRUNC, 0664); */
  /* fo = open(buf_o, O_CREAT | O_WRONLY | O_TRUNC, 0664); */

  /* i = write(fa, a, sizeof(double)*devMat->cols*devMat->rows); */
  /* i = write(fo, off, sizeof(int)*devMat->cols); */

  /* close(fa); */
  /* close(fo); */

  hoff = (int *)calloc(devMat->allocationHeight,sizeof(int));
  values = (double *)calloc(devMat->hackSize*devMat->allocationHeight,sizeof(double));

  diaToHdia((void *)values,hoff,devMat->hackOffsets,devMat->hackSize,(void *)a,off,devMat->rows,devMat->diags,devMat->rows,SPGPU_TYPE_DOUBLE);

  hackoff = devMat->hackOffsets;

  if (i == SPGPU_SUCCESS)
    i=allocRemoteBuffer((void **)&(devMat->hackOffsets), (devMat->hackCount+1)*sizeof(int));
  
  if(i== SPGPU_SUCCESS)
    i = writeRemoteBuffer(hackoff,devMat->hackOffsets,(devMat->hackCount+1)*sizeof(int));

  free(hackoff);

  i = writeRemoteBuffer((void*) hoff, (void *)devMat->hdiaOffsets, devMat->allocationHeight*sizeof(int));
  i = writeRemoteBuffer((void*) values, (void *)devMat->cM, devMat->allocationHeight*devMat->hackSize*sizeof(double));

  free(hoff);
  free(values);

  if(i==0)
    return SPGPU_SUCCESS;
  else
    return SPGPU_UNSUPPORTED;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

int readHdiagDeviceDouble(void* deviceMat, double* a, int* off)
{ int i;
#ifdef HAVE_SPGPU
  struct HdiagDevice *devMat = (struct HdiagDevice *) deviceMat;
  /* i = readRemoteBuffer((void *) a, (void *)devMat->cM,devMat->rows*devMat->diags*sizeof(double)); */
  /* i = readRemoteBuffer((void *) off, (void *)devMat->off, devMat->diags*sizeof(int)); */


  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readEllDeviceDouble",i);
  }*/
  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

//new
int spmvHdiagDeviceDouble(void *deviceMat, double alpha, void* deviceX, 
			  double beta, void* deviceY)
{
  struct HdiagDevice *devMat = (struct HdiagDevice *) deviceMat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY;
  spgpuHandle_t handle=psb_gpuGetHandle();

#ifdef HAVE_SPGPU
#ifdef VERBOSE
  /*__assert(x->count_ == x->count_, "ERROR: x and y don't share the same number of vectors");*/
  /*__assert(x->size_ >= devMat->columns, "ERROR: x vector's size is not >= to matrix size (columns)");*/
  /*__assert(y->size_ >= devMat->rows, "ERROR: y vector's size is not >= to matrix size (rows)");*/
#endif

  /* spgpuDdiaspmv(handle, (double *)y->v_, (double *)y->v_,alpha,(double *)devMat->cM,devMat->off,devMat->rows,devMat->rows,devMat->cols,devMat->diags,x->v_,beta); */
  
  spgpuDhdiaspmv (handle, (double*)y->v_, (double *)y->v_, alpha,(double *)devMat->cM,devMat->hdiaOffsets, devMat->hackSize, devMat->hackOffsets, devMat->rows,devMat->cols,x->v_, beta);
  
  //cudaSync();

  return SPGPU_SUCCESS;
#else
  return SPGPU_UNSUPPORTED;
#endif
}

/* int spmvHllDeviceFloatComplex(void *deviceMat, float complex alpha, void* deviceX,  */
/* 		       float complex beta, void* deviceY) */
/* { */
/*   struct HllDevice *devMat = (struct HllDevice *) deviceMat; */
/*   struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX; */
/*   struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY; */

/* #ifdef HAVE_SPGPU */
/*   cuFloatComplex a = make_cuFloatComplex(crealf(alpha),cimagf(alpha)); */
/*   cuFloatComplex b = make_cuFloatComplex(crealf(beta),cimagf(beta)); */
/* #ifdef VERBOSE */
/*   /\*__assert(x->count_ == x->count_, "ERROR: x and y don't share the same number of vectors");*\/ */
/* int readHllDeviceDoubleComplex(void* deviceMat, double complex* val, int* ja, int *hkoffs, int* irn) */
/* { int i; */
/* #ifdef HAVE_SPGPU */
/*   struct HllDevice *devMat = (struct HllDevice *) deviceMat; */
/*   i = readRemoteBuffer((void *) val, (void *)devMat->cM, devMat->allocsize*sizeof(cuDoubleComplex)); */
/*   i = readRemoteBuffer((void *) ja, (void *)devMat->rP, devMat->allocsize*sizeof(int)); */
/*   i = readRemoteBuffer((void *) irn, (void *)devMat->rS, devMat->rows*sizeof(int)); */
/*   i = readRemoteBuffer((void*) hkoffs, (void *)devMat->hackOffs, devMat->hackOffsLength*sizeof(int)); */
/*   /\*if (i != 0) { */
/*     fprintf(stderr,"From routine : %s : %d \n","readEllDeviceDouble",i); */
/*   }*\/ */
/*   return SPGPU_SUCCESS; */
/* #else */
/*   return SPGPU_UNSUPPORTED; */
/* #endif */
/* } */

#endif
