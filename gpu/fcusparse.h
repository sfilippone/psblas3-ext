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
 
  

#ifndef FCUSPARSE_
#define FCUSPARSE_

#ifdef HAVE_SPGPU
#include <cuda_runtime.h>
#include <cusparse_v2.h>
#include "cintrf.h"

typedef struct d_CSRGDeviceMat
{			
  cusparseMatDescr_t descr;
  cusparseSolveAnalysisInfo_t triang;
  int                 m, n, nz;
  double             *val;
  int                *irp;
  int                *ja;
} d_CSRGDeviceMat;

typedef struct s_CSRGDeviceMat
{			
  cusparseMatDescr_t descr;
  cusparseSolveAnalysisInfo_t triang;
  int                 m, n, nz;
  float              *val;
  int                *irp;
  int                *ja;
} s_CSRGDeviceMat;

/* Interoperability: type coming from Fortran side to distinguish D/S/C/Z. */
typedef struct s_Cmat
{
  s_CSRGDeviceMat *mat;
} s_Cmat;

typedef struct d_Cmat
{
  d_CSRGDeviceMat *mat;
} d_Cmat;

typedef struct s_HYBGDeviceMat
{			
  cusparseMatDescr_t descr;
  cusparseSolveAnalysisInfo_t triang;
  cusparseHybMat_t hybA;
  int                m, n, nz;
  float             *val;
  int               *irp;
  int               *ja;
} s_HYBGDeviceMat;

typedef struct d_HYBGDeviceMat
{			
  cusparseMatDescr_t descr;
  cusparseSolveAnalysisInfo_t triang;
  cusparseHybMat_t hybA;
  int                 m, n, nz;
  double             *val;
  int                *irp;
  int                *ja;
} d_HYBGDeviceMat;

/* Interoperability: type coming from Fortran side to distinguish D/S/C/Z. */
typedef struct s_Hmat
{
  s_HYBGDeviceMat *mat;
} s_Hmat;

typedef struct d_Hmat
{
  d_HYBGDeviceMat *mat;
} d_Hmat;


int FcusparseCreate(void **context);
int FcusparseDestroy(void **context);


int s_spmvCSRGDevice(s_Cmat *Mat, float alpha, void *deviceX, 
		     float beta, void *deviceY);
int s_spsvCSRGDevice(s_Cmat *Mat, float alpha, void *deviceX, 
		     float beta, void *deviceY);
int s_CSRGDeviceAlloc(s_Cmat *Mat,int nr, int nc, int nz);
int s_CSRGDeviceFree(s_Cmat *Mat);

int s_CSRGDeviceSetMatType(s_Cmat *Mat, int type);
int s_CSRGDeviceSetMatFillMode(s_Cmat *Mat, int type);
int s_CSRGDeviceSetMatDiagType(s_Cmat *Mat, int type);
int s_CSRGDeviceSetMatIndexBase(s_Cmat *Mat, int type);

int s_CSRGDeviceCsrsmAnalysis(s_Cmat *Mat);

int s_CSRGHost2Device(s_Cmat *Mat, int m, int n, int nz,
		      int *irp, int *ja, float *val);
int s_CSRGDevice2Host(s_Cmat *Mat, int m, int n, int nz,
		      int *irp, int *ja, float *val);

int s_HYBGDeviceFree(s_Hmat *Matrix);
int s_spmvHYBGDevice(s_Hmat *Matrix, float alpha, void *deviceX,
		     float beta, void *deviceY);
int s_HYBGDeviceAlloc(s_Hmat *Matrix,int nr, int nc, int nz);
int s_HYBGDeviceSetMatDiagType(s_Hmat *Matrix, int type);
int s_HYBGDeviceSetMatIndexBase(s_Hmat *Matrix, int type);
int s_HYBGDeviceSetMatType(s_Hmat *Matrix, int type);
int s_HYBGDeviceSetMatFillMode(s_Hmat *Matrix, int type);
int s_HYBGDeviceHybsmAnalysis(s_Hmat *Matrix);
int s_spsvHYBGDevice(s_Hmat *Matrix, float alpha, void *deviceX,
		     float beta, void *deviceY);
int s_HYBGHost2Device(s_Hmat *Matrix, int m, int n, int nz,
			  int *irp, int *ja, float *val);


int d_spmvCSRGDevice(d_Cmat *Mat, double alpha, void *deviceX, 
		     double beta, void *deviceY);
int d_spsvCSRGDevice(d_Cmat *Mat, double alpha, void *deviceX, 
		     double beta, void *deviceY);
int d_CSRGDeviceAlloc(d_Cmat *Mat,int nr, int nc, int nz);
int d_CSRGDeviceFree(d_Cmat *Mat);

int d_CSRGDeviceSetMatType(d_Cmat *Mat, int type);
int d_CSRGDeviceSetMatFillMode(d_Cmat *Mat, int type);
int d_CSRGDeviceSetMatDiagType(d_Cmat *Mat, int type);
int d_CSRGDeviceSetMatIndexBase(d_Cmat *Mat, int type);

int d_CSRGDeviceCsrsmAnalysis(d_Cmat *Mat);

int d_CSRGHost2Device(d_Cmat *Mat, int m, int n, int nz,
		      int *irp, int *ja, double *val);
int d_CSRGDevice2Host(d_Cmat *Mat, int m, int n, int nz,
		      int *irp, int *ja, double *val);

int d_HYBGDeviceFree(d_Hmat *Matrix);
int d_spmvHYBGDevice(d_Hmat *Matrix, double alpha, void *deviceX,
		     double beta, void *deviceY);
int d_HYBGDeviceAlloc(d_Hmat *Matrix,int nr, int nc, int nz);
int d_HYBGDeviceSetMatDiagType(d_Hmat *Matrix, int type);
int d_HYBGDeviceSetMatIndexBase(d_Hmat *Matrix, int type);
int d_HYBGDeviceSetMatType(d_Hmat *Matrix, int type);
int d_HYBGDeviceSetMatFillMode(d_Hmat *Matrix, int type);
int d_HYBGDeviceHybsmAnalysis(d_Hmat *Matrix);
int d_spsvHYBGDevice(d_Hmat *Matrix, double alpha, void *deviceX,
		     double beta, void *deviceY);
int d_HYBGHost2Device(d_Hmat *Matrix, int m, int n, int nz,
			  int *irp, int *ja, double *val);

#endif
#endif
