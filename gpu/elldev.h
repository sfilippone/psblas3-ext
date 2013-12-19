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
 
  

#ifndef _ELLDEV_H_
#define _ELLDEV_H_

#if defined(HAVE_SPGPU)
#include "cintrf.h"
#include "ell.h"
#include "ell_conv.h"


struct EllDevice
{
  // Compressed matrix
  void *cM; //it can be float or double

  // row pointers (same size of cM)
  int *rP;
  
  // row size
  int *rS;

  //matrix size (uncompressed)
  int rows;
  int columns;

  int pitch; //old

  int cMPitch;
  
  int rPPitch;

  int maxRowSize;

  //allocation size (in elements)
  int allocsize;

  /*(i.e. 0 for C, 1 for Fortran)*/
  int baseIndex;
};

typedef struct EllDeviceParams
{			
	// The resulting allocation for cM and rP will be pitch*maxRowSize*(size of the elementType)
	unsigned int elementType;
	
	// Pitch (in number of elements)
	unsigned int pitch;

	// Number of rows.
	// Used to allocate rS array
	unsigned int rows; 
		
	// Number of columns.
	// Used for error-checking
	unsigned int columns; 
	
	// Largest row size
	unsigned int maxRowSize;
	
	// First index (e.g 0 or 1)
	unsigned int firstIndex;
} EllDeviceParams;

int FallocEllDevice(void** deviceMat, unsigned int rows, unsigned int maxRowSize, 
		    unsigned int columns, unsigned int elementType, 
		    unsigned int firstIndex);
int allocEllDevice(void ** remoteMatrix, EllDeviceParams* params);
void freeEllDevice(void* remoteMatrix);
int writeEllDeviceFloat(void* deviceMat, float* val, int* ja, int ldj, int* irn);
int writeEllDeviceDouble(void* deviceMat, double* val, int* ja, int ldj, int* irn);
int readEllDeviceFloat(void* deviceMat, float* val, int* ja, int ldj, int* irn);
int readEllDeviceDouble(void* deviceMat, double* val, int* ja, int ldj, int* irn);

int getEllDevicePitch(void* deviceMat);

// sparse Ell matrix-vector product
//int spmvEllDeviceFloat(void *deviceMat, float* alpha, void* deviceX, float* beta, void* deviceY);
//int spmvEllDeviceDouble(void *deviceMat, double* alpha, void* deviceX, double* beta, void* deviceY);
/*struct FloatEllDevice
{
  // Compressed matrix
  float *cM;

  // row pointers (same size of cM)
  int *rP;

  // row size
  int *rS;

  //matrix size (uncompressed)
  int rows;
  int columns;

  int pitch; //old

  int cMPitch;
  
  int rPPitch;

  int maxRowSize;

  //allocation size (in elements)
  int allocsize;

  //(i.e. 0 for C, 1 for Fortran)
  int baseIndex;
}*/
#else
#define CINTRF_UNSUPPORTED   -1
#endif

#endif