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
 
  

#pragma once
#if defined(HAVE_SPGPU)
#include "cuComplex.h"
#include "vectordev.h"
#include "cuda_runtime.h"
#include "core.h"

int registerMappedFloatComplex(void *, void **, int, cuFloatComplex);

int writeMultiVecDeviceFloatComplex(void* deviceVec, cuFloatComplex* hostVec);
int writeMultiVecDeviceFloatComplexR2(void* deviceVec, cuFloatComplex* hostVec, int ld);
int readMultiVecDeviceFloatComplex(void* deviceVec, float complex* hostVec);
int readMultiVecDeviceFloatComplexR2(void* deviceMultiVec, float complex* hostMultiVec, int ld);
int nrm2MultiVecDeviceFloatComplex(float* y_res, int n, void* devMultiVecA);
int amaxMultiVecDeviceFloatComplex(float* y_res, int n, void* devVecA);
int asumMultiVecDeviceFloatComplex(float* y_res, int n, void* devVecA);
int dotMultiVecDeviceFloatComplex(float complex* y_res, int n, void* devMultiVecA, void* devMultiVecB);
int axpbyMultiVecDeviceFloatComplex(int n, float complex alpha, void* devMultiVecX, 
				    float complex beta, void* devMultiVecY);
int axyMultiVecDeviceFloatComplex(int n, float complex alpha, void *deviceVecA, void *deviceVecB);
int axybzMultiVecDeviceFloatComplex(int n, float complex alpha, void *deviceVecA,
				    void *deviceVecB, float complex beta, 
				    void *deviceVecZ);
int igathMultiVecDeviceFloatComplexVecIdx(void* deviceVec, int vectorId, int first,
				    int n, void* indexes, void* host_values, int indexBase);
int igathMultiVecDeviceFloatComplex(void* deviceVec, int vectorId, int first,
				    int n, void* indexes, void* host_values, int indexBase);
int iscatMultiVecDeviceFloatComplexVecIdx(void* deviceVec, int vectorId, int first, int n, void *indexes,
				    void* host_values, int indexBase, float complex beta);
int iscatMultiVecDeviceFloatComplex(void* deviceVec, int vectorId, int first, int n, void *indexes,
				    void* host_values, int indexBase, float complex beta);

#endif
