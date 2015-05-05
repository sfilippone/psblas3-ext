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
#include "ivectordev.h"


int registerMappedInt(void  *buff, void **d_p, int n, int dummy)
{
  return registerMappedMemory(buff,d_p,n*sizeof(int));
}

int writeMultiVecDeviceInt(void* deviceVec, int* hostVec)
{ int i;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  // Ex updateFromHost vector function
  i = writeRemoteBuffer((void*) hostVec, (void *)devVec->v_, 
			devVec->pitch_*devVec->count_*sizeof(int));
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceInt",i);
  }*/
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","FallocMultiVecDevice",i);
  }

  //  cudaSync();
  return(i);
}

int writeMultiVecDeviceIntR2(void* deviceVec, int* hostVec, int ld)
{ int i;
  i = writeMultiVecDeviceInt(deviceVec, (void *) hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","writeMultiVecDeviceIntR2",i);
  }
  //  cudaSync();
  return(i);
}

int readMultiVecDeviceInt(void* deviceVec, int* hostVec)
{ int i,j;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  i = readRemoteBuffer((void *) hostVec, (void *)devVec->v_, 
		       devVec->pitch_*devVec->count_*sizeof(int));
#if 0
  for (j=0;j<devVec->size_; j++) {
    fprintf(stderr,"readInt:  %d  %d \n",j,hostVec[j]);
  }
#endif
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceInt",i);
  }
  //  cudaSync();
  return(i);
}

int readMultiVecDeviceGatherInt(void* deviceVec, int* hostVec, int * idx, int *n)
{int i;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  //i = readMultiVecDeviceGather(deviceVec, (void *) hostVec, (void *) idx, (void *)n);
  /*if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceGatherInt",i);
  }*/
  return(i);
}

int readMultiVecDeviceIntR2(void* deviceVec, int* hostVec, int ld)
{ int i;
  //i = readMultiVecDevice(deviceVec, (void *) hostVec);
  i = readMultiVecDeviceInt(deviceVec, hostVec);
  if (i != 0) {
    fprintf(stderr,"From routine : %s : %d \n","readMultiVecDeviceIntR2",i);
  }
  return(i);
}

#if 0
int geinsMultiVecDeviceInt(int n, void* devMultiVecIrl, void* devMultiVecVal, 
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
  int *hval=(int *) malloc(n*sizeof(int));
  int nx=devVecX->size_;
  int *hx=(int *) malloc(nx*sizeof(int));
  i = readRemoteBuffer((void *) hidx, (void *)devVecIrl->v_, 
		       n*sizeof(int));
  i = readRemoteBuffer((void *) hval, (void *)devVecVal->v_, 
		       n*sizeof(int));
  
  i = readRemoteBuffer((void *) hx, (void *)devVecX->v_, 
		       nx*sizeof(int));
  if (n<nx) {
    for (j=0; j<n; j++) {
      fprintf(stderr,"before: %d  %d %12d %12d \n",j,hidx[j],hval[j],hx[j]);
    }
    for (j=n; j<nx; j++) {
      fprintf(stderr,"%d  %d \n",j,hx[j]);
    }
  } else {
    for (j=0; j<nx; j++) {
      fprintf(stderr,"before: %d  %d %12d %12d \n",j,hidx[j],hval[j],hx[j]);
    }
    for (j=nx; j<n; j++) {
      fprintf(stderr,"%d   %d %lf \n",j,hidx[j],hval[j]);
    }
  }


#endif
  
  spgpuIgeins(handle,n, (int*)devVecIrl->v_, 
  	      (int*)devVecVal->v_, dupl, indexBase,(int*) devVecX->v_);

#if DEBUG_GEINS
  i = readRemoteBuffer((void *) hidx, (void *)devVecIrl->v_, 
		       n*sizeof(int));
  i = readRemoteBuffer((void *) hval, (void *)devVecVal->v_, 
		       n*sizeof(int));
  
  i = readRemoteBuffer((void *) hx, (void *)devVecX->v_, 
		       nx*sizeof(int));
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

int igathMultiVecDeviceIntVecIdx(void* deviceVec, int vectorId, int first,
			      int n, void* deviceIdx, void* host_values, int indexBase)
{
  int i, *idx;
  struct MultiVectDevice *devIdx = (struct MultiVectDevice *) deviceIdx;
  i= igathMultiVecDeviceInt(deviceVec, vectorId, first,
			       n, (void*) devIdx->v_,  host_values, indexBase);
  return(i);
}

int igathMultiVecDeviceInt(void* deviceVec, int vectorId, int first,
			      int n, void* indexes, void* host_values, int indexBase)
{
  int i, *idx;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  i=0; 
  idx = (int *) indexes;
  idx = &(idx[first-indexBase]);
  spgpuIgath(handle,(int *)host_values, n, idx,
  	     indexBase, (int *) devVec->v_+vectorId*devVec->pitch_);
  return(i);
}

int iscatMultiVecDeviceIntVecIdx(void* deviceVec, int vectorId, int first, int n, void *deviceIdx,
			      void* host_values, int indexBase, int beta)
{  
  int i, *idx;
  struct MultiVectDevice *devIdx = (struct MultiVectDevice *) deviceIdx;
  i= iscatMultiVecDeviceInt(deviceVec, vectorId, first, n, 
			       (void*) devIdx->v_,  host_values, indexBase, beta);
  return(i);
}

int iscatMultiVecDeviceInt(void* deviceVec, int vectorId, int first, int n, void *indexes,
			     void* host_values, int indexBase, int beta)
{ int i=0;
  int *idx=(int *) indexes;
  struct MultiVectDevice *devVec = (struct MultiVectDevice *) deviceVec;
  spgpuHandle_t handle=psb_gpuGetHandle();

  idx = &(idx[first-indexBase]);
  spgpuIscat(handle, (int *) devVec->v_, n, (int *) host_values, idx, indexBase, beta);
  return SPGPU_SUCCESS;
  
}

#endif

