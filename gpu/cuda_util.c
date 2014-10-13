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
 
  

#include "cuda_util.h"

#if defined(HAVE_CUDA)

int allocRemoteBuffer(void** buffer, int count)
{
  cudaError_t err = cudaMalloc(buffer, count);
  if (err == cudaSuccess)
    {
      return SPGPU_SUCCESS;
    }
  else
    { 
      fprintf(stderr,"CUDA allocRemoteBuffer Error: %s\n", cudaGetErrorString(err));
      if(err == cudaErrorMemoryAllocation)
	return SPGPU_OUTOFMEMORY;
      else
	return SPGPU_UNSPECIFIED;
    }
}

int hostRegisterMapped(void *pointer, long size) 
{
  cudaError_t err = cudaHostRegister(pointer, size, cudaHostRegisterMapped);

 if (err == cudaSuccess)
    {
      return SPGPU_SUCCESS;
    }
  else
    { 
      fprintf(stderr,"CUDA hostRegisterMapped Error: %s\n", cudaGetErrorString(err));
      if(err == cudaErrorMemoryAllocation)
	return SPGPU_OUTOFMEMORY;
      else
	return SPGPU_UNSPECIFIED;
    }
}

int getDevicePointer(void **d_p, void * h_p)
{
  cudaError_t err = cudaHostGetDevicePointer(d_p,h_p,0);

 if (err == cudaSuccess)
    {
      return SPGPU_SUCCESS;
    }
  else
    { 
      fprintf(stderr,"CUDA getDevicePointer Error: %s\n", cudaGetErrorString(err));
      if(err == cudaErrorMemoryAllocation)
	return SPGPU_OUTOFMEMORY;
      else
	return SPGPU_UNSPECIFIED;
    }
}

int registerMappedMemory(void *buffer, void **dp, int size)
{
  //cudaError_t err = cudaHostAlloc(buffer,size,cudaHostAllocMapped);
  cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterMapped);
  if (err == cudaSuccess) err = cudaHostGetDevicePointer(dp,buffer,0);

  if (err == cudaSuccess)
    {
      err = cudaHostGetDevicePointer(dp,buffer,0);
      if (err == cudaSuccess)
	{
	  return SPGPU_SUCCESS;
	}
      else
	{
	  fprintf(stderr,"CUDA registerMappedMemory Error: %s\n", cudaGetErrorString(err));
	  return SPGPU_UNSPECIFIED;
	}
    }
  else
    { 
      fprintf(stderr,"CUDA registerMappedMemory Error: %s\n", cudaGetErrorString(err));
      if(err == cudaErrorMemoryAllocation)
	return SPGPU_OUTOFMEMORY;
      else
	return SPGPU_UNSPECIFIED;
    }
}

int allocMappedMemory(void **buffer, void **dp, int size)
{
  cudaError_t err = cudaHostAlloc(buffer,size,cudaHostAllocMapped);
  if (err == 0) err = cudaHostGetDevicePointer(dp,*buffer,0);

  if (err == cudaSuccess)
    {
      return SPGPU_SUCCESS;
    }
  else
    { 
      fprintf(stderr,"CUDA allocMappedMemory Error: %s\n", cudaGetErrorString(err));
      if(err == cudaErrorMemoryAllocation)
	return SPGPU_OUTOFMEMORY;
      else
	return SPGPU_UNSPECIFIED;
    }
}

int unregisterMappedMemory(void *buffer)
{
  //cudaError_t err = cudaHostAlloc(buffer,size,cudaHostAllocMapped);
  cudaError_t err = cudaHostUnregister(buffer);

  if (err == cudaSuccess)
    {
      return SPGPU_SUCCESS;
    }
  else
    { 
      fprintf(stderr,"CUDA unregisterMappedMemory Error: %s\n", cudaGetErrorString(err));
      if(err == cudaErrorMemoryAllocation)
	return SPGPU_OUTOFMEMORY;
      else
	return SPGPU_UNSPECIFIED;
    }
}

int writeRemoteBuffer(void* hostSrc, void* buffer, int count)
{
  cudaError_t err = cudaMemcpy(buffer, hostSrc, count, cudaMemcpyHostToDevice);

  if (err == cudaSuccess)
    return SPGPU_SUCCESS;	
  else {
    fprintf(stderr,"CUDA Error writeRemoteBuffer: %s\n", cudaGetErrorString(err));
    return SPGPU_UNSPECIFIED;
  }
}

int readRemoteBuffer(void* hostDest, void* buffer, int count)
{
  cudaError_t err = cudaMemcpy(hostDest, buffer, count, cudaMemcpyDeviceToHost);

  if (err == cudaSuccess)
    return SPGPU_SUCCESS;	
  else {
    fprintf(stderr,"CUDA Error readRemoteBuffer: %s %p  %p %d %d\n", 
	    cudaGetErrorString(err),hostDest,buffer,count,err);
    return SPGPU_UNSPECIFIED;
  }
}

int freeRemoteBuffer(void* buffer)
{
  cudaError_t err = cudaFree(buffer);
  if (err == cudaSuccess)
    return SPGPU_SUCCESS;	
  else {
    fprintf(stderr,"CUDA Error freeRemoteBuffer: %s\n", cudaGetErrorString(err));
    return SPGPU_UNSPECIFIED;
  }
}

int gpuInit(int dev)
{

  int count,err;
  
  if ((err=cudaSetDeviceFlags(cudaDeviceMapHost))!=cudaSuccess) 
    fprintf(stderr,"Error On SetDeviceFlags: %d '%s'\n",err,cudaGetErrorString(err));
  err = cudaGetDeviceCount(&count);
  if (err == cudaSuccess)
    return SPGPU_SUCCESS;	
  else {
    fprintf(stderr,"CUDA Error gpuInit2: %s\n", cudaGetErrorString(err));
    return SPGPU_UNSPECIFIED;
  }
  
  if ((0<=dev)&&(dev<count))
    err = cudaSetDevice(dev);
  else
    err = cudaSetDevice(0);

  if (err == cudaSuccess)
    return SPGPU_SUCCESS;	
  else {
    fprintf(stderr,"CUDA Error gpuInit: %s\n", cudaGetErrorString(err));
    return SPGPU_UNSPECIFIED;
  }

  return err;
  
}

int getDeviceCount()
{ int count;
  cudaError_t err;
  err = cudaGetDeviceCount(&count);
  if (err == cudaSuccess)
    return SPGPU_SUCCESS;	
  else {
    fprintf(stderr,"CUDA Error getDeviceCount: %s\n", cudaGetErrorString(err));
    return SPGPU_UNSPECIFIED;
  }
  return(count);
}

void cudaSync()
{
  cudaError_t err;
  err = cudaDeviceSynchronize();
  if (err == cudaSuccess)
    return SPGPU_SUCCESS;	
  else {
    fprintf(stderr,"CUDA Error cudaSync: %s\n", cudaGetErrorString(err));
    return SPGPU_UNSPECIFIED;
  }
}

void cudaReset()
{
  cudaError_t err;
  err = cudaDeviceReset();
  if (err != cudaSuccess) {
    fprintf(stderr,"CUDA Error Reset: %s\n", cudaGetErrorString(err));
    return SPGPU_UNSPECIFIED;
  }
}

static spgpuHandle_t psb_gpu_handle = NULL;
static spgpuHandle_t psb_gpu_handle_v = NULL;


spgpuHandle_t psb_gpuGetHandle()
{
  return psb_gpu_handle;
}

void psb_gpuCreateHandle()
{
  if (!psb_gpu_handle)
    spgpuCreate(&psb_gpu_handle, 0);
}

void psb_gpuDestroyHandle()
{
  spgpuDestroy(psb_gpu_handle);
}

cudaStream_t psb_gpuGetStream()
{
  return spgpuGetStream(psb_gpu_handle);
}

void  psb_gpuSetStream(cudaStream_t stream)
{
  spgpuSetStream(psb_gpu_handle, stream);
  return ;
}


#endif
