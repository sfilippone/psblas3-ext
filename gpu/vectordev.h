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

struct MultiVectDevice
{
  // number of vectors
  int count_;

  //number of elements for a single vector
  int size_;

  //pithc in number of elements
  int pitch_;

  // Vectors in device memory (single allocation)
  void *v_;
};

typedef struct MultiVectorDeviceParams
{			
	// number on vectors
	unsigned int count; //1 for a simple vector

	// The resulting allocation will be pitch*s*(size of the elementType)
	unsigned int elementType;
	
	// Pitch (in number of elements)
	unsigned int pitch;

	// Size of a single vector (in number of elements).
	unsigned int size; 
} MultiVectorDeviceParams;



int allocateIdx(void **, int);
int writeIdx(void *, int *, int);
int readIdx(void *, int *, int);
void freeIdx(void *);

int registerMappedDouble(void *, void **, int, double);
int unregisterMapped(void *);

int FallocMultiVecDevice(void** deviceMultiVec, unsigned count, unsigned int size, unsigned int elementType);
void freeMultiVecDevice(void* deviceVec);
int allocMultiVecDevice(void ** remoteMultiVec, struct MultiVectorDeviceParams *params);

int writeMultiVecDeviceInt(void* deviceMultiVec, int* hostMultiVec);
int writeMultiVecDeviceFloat(void* deviceMultiVec, float* hostMultiVec);
int writeMultiVecDeviceDouble(void* deviceMultiVec, double* hostMultiVec);
int writeMultiVecDeviceFloatComplex(void* deviceVec, cuFloatComplex* hostVec);
int writeMultiVecDeviceDoubleComplex(void* deviceVec, cuDoubleComplex* hostVec);

int writeMultiVecDeviceIntR2(void* deviceMultiVec, int* hostMultiVec, int ld);
int writeMultiVecDeviceFloatR2(void* deviceMultiVec, float* hostMultiVec, int ld);
int writeMultiVecDeviceDoubleR2(void* deviceMultiVec, double* hostMultiVec, int ld);
int writeMultiVecDeviceFloatComplexR2(void* deviceVec, cuFloatComplex* hostVec, int ld);
int writeMultiVecDeviceDoubleComplexR2(void* deviceVec, cuDoubleComplex* hostVec, int ld);

int getMultiVecDeviceSize(void* deviceVec);
int getMultiVecDeviceCount(void* deviceVec);
int getMultiVecDevicePitch(void* deviceVec);

int readMultiVecDeviceInt(void* deviceMultiVec, int* hostMultiVec);
int readMultiVecDeviceFloat(void* deviceMultiVec, float* hostMultiVec);
int readMultiVecDeviceDouble(void* deviceMultiVec, double* hostMultiVec);
int readMultiVecDeviceFloatComplex(void* deviceVec, float complex* hostVec);
int readMultiVecDeviceDoubleComplex(void* deviceVec, double complex* hostVec);

int readMultiVecDeviceIntR2(void* deviceMultiVec, int* hostMultiVec, int ld);
int readMultiVecDeviceFloatR2(void* deviceMultiVec, float* hostMultiVec, int ld);
int readMultiVecDeviceDoubleR2(void* deviceMultiVec, double* hostMultiVec, int ld);
int readMultiVecDeviceFloatComplexR2(void* deviceMultiVec, float complex* hostMultiVec, int ld);
int readMultiVecDeviceDoubleComplexR2(void* deviceMultiVec, double complex* hostMultiVec, int ld);

int nrm2MultiVecDeviceFloat(float* y_res, int n, void* devVecA);
int nrm2MultiVecDeviceDouble(double* y_res, int n, void* devVecA);
int nrm2MultiVecDeviceFloatComplex(float* y_res, int n, void* devMultiVecA);
int nrm2MultiVecDeviceDoubleComplex(double* y_res, int n, void* devMultiVecA);

int amaxMultiVecDeviceFloat(float* y_res, int n, void* devVecA);
int amaxMultiVecDeviceDouble(double* y_res, int n, void* devVecA);
int amaxMultiVecDeviceFloatComplex(float* y_res, int n, void* devVecA);
int amaxMultiVecDeviceDoubleComplex(double* y_res, int n, void* devVecA);


int asumMultiVecDeviceFloat(float* y_res, int n, void* devVecA);
int asumMultiVecDeviceDouble(double* y_res, int n, void* devVecA);
int asumMultiVecDeviceFloatComplex(float* y_res, int n, void* devVecA);
int asumMultiVecDeviceDoubleComplex(double* y_res, int n, void* devVecA);



int dotMultiVecDeviceFloat(float* y_res, int n, void* devVecA, void* devVecB);
int dotMultiVecDeviceDouble(double* y_res, int n, void* devVecA, void* devVecB);
int dotMultiVecDeviceFloatComplex(float complex* y_res, int n, void* devMultiVecA, void* devMultiVecB);
int dotMultiVecDeviceDoubleComplex(double complex* y_res, int n, void* devMultiVecA, void* devMultiVecB);

int geinsMultiVecDeviceDouble(int n, void* devVecIrl, void* devVecVal, 
			      int dupl, int indexBase, void* devVecX); 


int axpbyMultiVecDeviceFloat(int n, float alpha, void* devVecX, float beta, void* devVecY);
int axpbyMultiVecDeviceDouble(int n, double alpha, void* devVecX, double beta, void* devVecY);
int axpbyMultiVecDeviceFloatComplex(int n, float complex alpha, void* devMultiVecX, 
				    float complex beta, void* devMultiVecY);
int axpbyMultiVecDeviceDoubleComplex(int n, double complex alpha, void* devMultiVecX, 
				     double complex beta, void* devMultiVecY);

int axyMultiVecDeviceFloat(int n, float alpha, void *deviceVecA, void *deviceVecB);
int axyMultiVecDeviceDouble(int n, double alpha, void *deviceVecA, void *deviceVecB);
int axyMultiVecDeviceFloatComplex(int n, float complex alpha, void *deviceVecA, void *deviceVecB);
int axyMultiVecDeviceDoubleComplex(int n, double complex alpha, void *deviceVecA, void *deviceVecB);

int axybzMultiVecDeviceFloat(int n, float alpha, void *deviceVecA, 
			     void *deviceVecB, float beta, void *deviceVecZ);
int axybzMultiVecDeviceDouble(int n, double alpha, void *deviceVecA,
			      void *deviceVecB, double beta, void *deviceVecZ);
int axybzMultiVecDeviceFloatComplex(int n, float complex alpha, void *deviceVecA,
				    void *deviceVecB, float complex beta, 
				    void *deviceVecZ);
int axybzMultiVecDeviceDoubleComplex(int n, double complex alpha, void *deviceVecA,
				     void *deviceVecB, double complex beta, 
				     void *deviceVecZ);



int igathMultiVecDeviceFloat(void* deviceVec, int vectorId, int first,
			     int n, void* indexes, void* host_values, int indexBase);
int igathMultiVecDeviceDouble(void* deviceVec, int vectorId, int first,
			      int n, void* indexes, void* host_values, int indexBase);
int igathMultiVecDeviceFloatComplex(void* deviceVec, int vectorId, int first,
				    int n, void* indexes, void* host_values, int indexBase);
int igathMultiVecDeviceDoubleComplex(void* deviceVec, int vectorId, int first,
				     int n, void* indexes, void* host_values, int indexBase);

int iscatMultiVecDeviceFloat(void* deviceVec, int vectorId, int first, int n, void *indexes,
			     void* host_values, int indexBase, float beta);
int iscatMultiVecDeviceDouble(void* deviceVec, int vectorId, int first, int n, void *indexes,
			      void* host_values, int indexBase, double beta);
int iscatMultiVecDeviceFloatComplex(void* deviceVec, int vectorId, int first, int n, void *indexes,
				    void* host_values, int indexBase, float complex beta);

int iscatMultiVecDeviceDoubleComplex(void* deviceVec, int vectorId, int first, int n, void *indexes,
				     void* host_values, int indexBase, double complex beta);

#endif
