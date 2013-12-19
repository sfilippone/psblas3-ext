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
 

#ifndef _HDIAGDEV_H_
#define _HDIAGDEV_H_

#ifdef HAVE_SPGPU
#include "cintrf.h"
#include "hdia.h"

struct HdiagDevice
{
  // Compressed matrix
  void *cM; //it can be float or double

  // offset (same size of cM)
  int *hdiaOffsets;

  int *hackOffsets;

  int hackCount;

  int rows;

  int cols;

  int diags;

  int hackSize;

  int allocationHeight;

};

typedef struct HdiagDeviceParams
{			
  
  unsigned int elementType;
  
  // Number of rows.
  // Used to allocate rS array
  unsigned int rows;
  //unsigned int hackOffsLength;
  
  // Number of columns.
  // Used for error-checking
  unsigned int columns; 

  unsigned int diags;

  unsigned int hackSize;

} HdiagDeviceParams;

int dia2hdia(void *hdiaValues,int *hdiaOffsets,int *hackOffsets,int hackSize,
	     void* diaValues, int* diaOffsets, int diaValuesPitch, int diagonals,
	     int rowsCount, int elementType);

HdiagDeviceParams getHdiagDeviceParams(unsigned int rows, unsigned int columns,
				       unsigned int diags, unsigned int hackSize,
				       unsigned int elementType);
int FallocHdiagDevice(void** deviceMat, unsigned int rows, unsigned int cols, 
		      unsigned int diags, unsigned int hackSize,double *data,unsigned int elementType);
int allocHdiagDevice(void ** remoteMatrix, HdiagDeviceParams* params,double * data);
void freeHdiagDevice(void* remoteMatrix);
/* int writeHllDeviceFloat(void* deviceMat, float* val, int* ja, int *hkoffs, int* irn); */
int writeHdiagDeviceDouble(void* deviceMat, double* a, int* off, int n);
#else
#define CINTRF_UNSUPPORTED   -1
#endif

#endif