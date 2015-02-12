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
//#include "utils.h"
#include "cuda_runtime.h"
//#include "common.h"
#include "cintrf.h"
#include <complex.h>
#include "cuComplex.h"
#include "vectordev.h"
#include "cuda_runtime.h"
#include "core.h"

int registerMappedDoubleComplex(void *, void **, int, cuDoubleComplex);
int writeMultiVecDeviceDoubleComplex(void* deviceVec, cuDoubleComplex* hostVec);
int writeMultiVecDeviceDoubleComplexR2(void* deviceVec, cuDoubleComplex* hostVec, int ld);
int readMultiVecDeviceDoubleComplex(void* deviceVec, double complex* hostVec);
int readMultiVecDeviceDoubleComplexR2(void* deviceMultiVec, double complex* hostMultiVec, int ld);
int nrm2MultiVecDeviceDoubleComplex(double* y_res, int n, void* devMultiVecA);
int amaxMultiVecDeviceDoubleComplex(double* y_res, int n, void* devVecA);
int asumMultiVecDeviceDoubleComplex(double* y_res, int n, void* devVecA);
int dotMultiVecDeviceDoubleComplex(double complex* y_res, int n, void* devMultiVecA, void* devMultiVecB);
int axpbyMultiVecDeviceDoubleComplex(int n, double complex alpha, void* devMultiVecX, 
				     double complex beta, void* devMultiVecY);
int axyMultiVecDeviceDoubleComplex(int n, double complex alpha, void *deviceVecA, void *deviceVecB);
int axybzMultiVecDeviceDoubleComplex(int n, double complex alpha, void *deviceVecA,
				     void *deviceVecB, double complex beta, 
				     void *deviceVecZ);
int igathMultiVecDeviceDoubleComplexVecIdx(void* deviceVec, int vectorId, int first,
				    int n, void* indexes, void* host_values, int indexBase);
int igathMultiVecDeviceDoubleComplex(void* deviceVec, int vectorId, int first,
				    int n, void* indexes, void* host_values, int indexBase);
int iscatMultiVecDeviceDoubleComplexVecIdx(void* deviceVec, int vectorId, int first, int n, void *indexes,
				    void* host_values, int indexBase, double complex beta);
int iscatMultiVecDeviceDoubleComplex(void* deviceVec, int vectorId, int first, int n, void *indexes,
				    void* host_values, int indexBase, double complex beta);

#endif
