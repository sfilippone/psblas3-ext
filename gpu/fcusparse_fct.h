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
 
typedef struct T_CSRGDeviceMat
{			
  cusparseMatDescr_t descr;
  cusparseSolveAnalysisInfo_t triang;
  int                 m, n, nz;
  TYPE               *val;
  int                *irp;
  int                *ja;
} T_CSRGDeviceMat;

/* Interoperability: type coming from Fortran side to distinguish D/S/C/Z. */
typedef struct T_Cmat
{
  T_CSRGDeviceMat *mat;
} T_Cmat;

typedef struct T_HYBGDeviceMat
{			
  cusparseMatDescr_t descr;
  cusparseSolveAnalysisInfo_t triang;
  cusparseHybMat_t hybA;
  int                m, n, nz;
  TYPE              *val;
  int               *irp;
  int               *ja;
} T_HYBGDeviceMat;


/* Interoperability: type coming from Fortran side to distinguish D/S/C/Z. */
typedef struct T_Hmat
{
  T_HYBGDeviceMat *mat;
} T_Hmat;

int T_spmvCSRGDevice(T_Cmat *Mat, TYPE alpha, void *deviceX, 
		     TYPE beta, void *deviceY);
int T_spsvCSRGDevice(T_Cmat *Mat, TYPE alpha, void *deviceX, 
		     TYPE beta, void *deviceY);
int T_CSRGDeviceAlloc(T_Cmat *Mat,int nr, int nc, int nz);
int T_CSRGDeviceFree(T_Cmat *Mat);

int T_CSRGDeviceSetMatType(T_Cmat *Mat, int type);
int T_CSRGDeviceSetMatFillMode(T_Cmat *Mat, int type);
int T_CSRGDeviceSetMatDiagType(T_Cmat *Mat, int type);
int T_CSRGDeviceSetMatIndexBase(T_Cmat *Mat, int type);

int T_CSRGDeviceCsrsmAnalysis(T_Cmat *Mat);

int T_CSRGHost2Device(T_Cmat *Mat, int m, int n, int nz,
		      int *irp, int *ja, TYPE *val);
int T_CSRGDevice2Host(T_Cmat *Mat, int m, int n, int nz,
		      int *irp, int *ja, TYPE *val);

int T_HYBGDeviceFree(T_Hmat *Matrix);
int T_spmvHYBGDevice(T_Hmat *Matrix, TYPE alpha, void *deviceX,
		     TYPE beta, void *deviceY);
int T_HYBGDeviceAlloc(T_Hmat *Matrix,int nr, int nc, int nz);
int T_HYBGDeviceSetMatDiagType(T_Hmat *Matrix, int type);
int T_HYBGDeviceSetMatIndexBase(T_Hmat *Matrix, int type);
int T_HYBGDeviceSetMatType(T_Hmat *Matrix, int type);
int T_HYBGDeviceSetMatFillMode(T_Hmat *Matrix, int type);
int T_HYBGDeviceHybsmAnalysis(T_Hmat *Matrix);
int T_spsvHYBGDevice(T_Hmat *Matrix, TYPE alpha, void *deviceX,
		     TYPE beta, void *deviceY);
int T_HYBGHost2Device(T_Hmat *Matrix, int m, int n, int nz,
			  int *irp, int *ja, TYPE *val);

  
int T_spmvCSRGDevice(T_Cmat *Matrix, TYPE alpha, void *deviceX,
		     TYPE beta, void *deviceY)
{
  T_CSRGDeviceMat *cMat=Matrix->mat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY; 
  void *vX, *vY;
  int r,n;
  cusparseHandle_t *my_handle=getHandle();
  /*getAddrMultiVecDevice(deviceX, &vX);
    getAddrMultiVecDevice(deviceY, &vY); */
  vX=x->v_;
  vY=y->v_;

  return cusparseTcsrmv(*my_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
			cMat->m,cMat->n,cMat->nz,(const TYPE *) &alpha,cMat->descr,
			cMat->val, cMat->irp, cMat->ja,
			(const TYPE *) vX, (const TYPE *) &beta, (TYPE *) vY);
}

int T_spsvCSRGDevice(T_Cmat *Matrix, TYPE alpha, void *deviceX,
		     TYPE beta, void *deviceY)
{
  T_CSRGDeviceMat *cMat=Matrix->mat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY; 
  void *vX, *vY;
  int r,n;
  cusparseHandle_t *my_handle=getHandle();
  /*getAddrMultiVecDevice(deviceX, &vX);
    getAddrMultiVecDevice(deviceY, &vY); */
  vX=x->v_;
  vY=y->v_;

  return cusparseTcsrsv_solve(*my_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
			      cMat->m,(const TYPE *) &alpha,cMat->descr,
			      cMat->val, cMat->irp, cMat->ja, cMat->triang,
			      (const TYPE *) vX,  (TYPE *) vY);
}

int T_CSRGDeviceAlloc(T_Cmat *Matrix,int nr, int nc, int nz)
{
  T_CSRGDeviceMat *cMat;
  int nr1=nr, nz1=nz, rc;
  if ((nr<0)||(nc<0)||(nz<0)) 
    return((int) CUSPARSE_STATUS_INVALID_VALUE);
  if ((cMat=(T_CSRGDeviceMat *) malloc(sizeof(T_CSRGDeviceMat)))==NULL)
    return((int) CUSPARSE_STATUS_ALLOC_FAILED);
  cMat->m  = nr;
  cMat->n  = nc;
  cMat->nz = nz;
  if (nr1 == 0) nr1 = 1;
  if (nz1 == 0) nz1 = 1;
  if ((rc= allocRemoteBuffer(((void **) &(cMat->irp)), ((nr1+1)*sizeof(int)))) != 0)
    return(rc);
  if ((rc= allocRemoteBuffer(((void **) &(cMat->ja)), ((nz1)*sizeof(int)))) != 0)
    return(rc);
  if ((rc= allocRemoteBuffer(((void **) &(cMat->val)), ((nz1)*sizeof(TYPE)))) != 0)
    return(rc);
  if ((rc= cusparseCreateMatDescr(&(cMat->descr))) !=0) 
    return(rc);
  if ((rc= cusparseCreateSolveAnalysisInfo(&(cMat->triang))) !=0)
    return(rc);
  Matrix->mat = cMat;
  return(CUSPARSE_STATUS_SUCCESS);
}

int T_CSRGDeviceFree(T_Cmat *Matrix)
{
  T_CSRGDeviceMat *cMat= Matrix->mat;
  
  if (cMat!=NULL) {
    //freeRemoteBuffer(cMat->irp);
    //freeRemoteBuffer(cMat->ja);
    //freeRemoteBuffer(cMat->val);
    cusparseDestroyMatDescr(cMat->descr);
    cusparseDestroySolveAnalysisInfo(cMat->triang);  
    free(cMat);
    Matrix->mat = NULL;
  }
  return(CUSPARSE_STATUS_SUCCESS);
}

int T_CSRGDeviceSetMatType(T_Cmat *Matrix, int type)
{
  T_CSRGDeviceMat *cMat= Matrix->mat;
  return ((int) cusparseSetMatType(cMat->descr,type));
}

int T_CSRGDeviceSetMatFillMode(T_Cmat *Matrix, int type)
{
  T_CSRGDeviceMat *cMat= Matrix->mat;
  return ((int) cusparseSetMatFillMode(cMat->descr,type));
}

int T_CSRGDeviceSetMatDiagType(T_Cmat *Matrix, int type)
{
  T_CSRGDeviceMat *cMat= Matrix->mat;
  return ((int) cusparseSetMatDiagType(cMat->descr,type));
}

int T_CSRGDeviceSetMatIndexBase(T_Cmat *Matrix, int type)
{
  T_CSRGDeviceMat *cMat= Matrix->mat;
  return ((int) cusparseSetMatIndexBase(cMat->descr,type));
}

int T_CSRGDeviceCsrsmAnalysis(T_Cmat *Matrix)
{
  T_CSRGDeviceMat *cMat= Matrix->mat;  
  cusparseSolveAnalysisInfo_t info;
  int rc;
  cusparseHandle_t *my_handle=getHandle();

  rc= (int)  cusparseTcsrsv_analysis(*my_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
				     cMat->m,cMat->nz,cMat->descr,
				     cMat->val, cMat->irp, cMat->ja,
				     cMat->triang);
  if (rc !=0) {
    fprintf(stderr,"From csrsv_analysis: %d\n",rc);
  }
}


int T_CSRGHost2Device(T_Cmat *Matrix, int m, int n, int nz,
		      int *irp, int *ja, TYPE *val) 
{
  int rc;
  T_CSRGDeviceMat *cMat= Matrix->mat;
  
  if ((rc=writeRemoteBuffer((void *) irp, (void *) cMat->irp, 
			    (m+1)*sizeof(int)))
      != SPGPU_SUCCESS) 
    return(rc);
  
  if ((rc=writeRemoteBuffer((void *) ja,(void *) cMat->ja, 
			    (nz)*sizeof(int)))
      != SPGPU_SUCCESS) 
    return(rc);
  if ((rc=writeRemoteBuffer((void *) val, (void *) cMat->val, 
			    (nz)*sizeof(TYPE)))
      != SPGPU_SUCCESS) 
    return(rc);
  return(CUSPARSE_STATUS_SUCCESS);
}

int T_CSRGDevice2Host(T_Cmat *Matrix, int m, int n, int nz,
		      int *irp, int *ja, TYPE *val) 
{
  int rc;
  T_CSRGDeviceMat *cMat = Matrix->mat;
  
  if ((rc=readRemoteBuffer((void *) irp, (void *) cMat->irp, (m+1)*sizeof(int))) 
      != SPGPU_SUCCESS) 
    return(rc);

  if ((rc=readRemoteBuffer((void *) ja, (void *) cMat->ja, (nz)*sizeof(int))) 
      != SPGPU_SUCCESS) 
    return(rc);
  if ((rc=readRemoteBuffer((void *) val, (void *) cMat->val, (nz)*sizeof(TYPE))) 
      != SPGPU_SUCCESS) 
    return(rc);

  return(CUSPARSE_STATUS_SUCCESS);
}

int T_HYBGDeviceFree(T_Hmat *Matrix)
{
  T_HYBGDeviceMat *hMat= Matrix->mat;
  
  //freeRemoteBuffer(hMat->irp);
  //freeRemoteBuffer(hMat->ja);
  //freeRemoteBuffer(hMat->val);
  cusparseDestroyMatDescr(hMat->descr);
  cusparseDestroySolveAnalysisInfo(hMat->triang);
  cusparseDestroyHybMat(hMat->hybA);
  free(hMat);
  Matrix->mat = NULL;
  return(CUSPARSE_STATUS_SUCCESS);
}

int T_spmvHYBGDevice(T_Hmat *Matrix, TYPE alpha, void *deviceX,
		     TYPE beta, void *deviceY)
{
  T_HYBGDeviceMat *hMat=Matrix->mat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY; 
  void *vX, *vY;
  int r,n,rc;
  cusparseMatrixType_t type;
  cusparseHandle_t *my_handle=getHandle();

  /*getAddrMultiVecDevice(deviceX, &vX);
    getAddrMultiVecDevice(deviceY, &vY); */
  vX=x->v_;
  vY=y->v_;

  /* rc = (int) cusparseGetMatType(hMat->descr); */
  /* fprintf(stderr,"Spmv MatType: %d\n",rc); */
  /* rc = (int) cusparseGetMatDiagType(hMat->descr); */
  /* fprintf(stderr,"Spmv DiagType: %d\n",rc); */
  /* rc = (int) cusparseGetMatFillMode(hMat->descr); */
  /* fprintf(stderr,"Spmv FillMode: %d\n",rc); */
  /* Dirty trick: apparently hybmv does not accept a triangular
     matrix even though it should not make a difference. So 
     we claim it's general anyway */ 
  type =  cusparseGetMatType(hMat->descr);
  rc = cusparseSetMatType(hMat->descr,CUSPARSE_MATRIX_TYPE_GENERAL);
  if (rc == 0) 
    rc = (int) cusparseThybmv(*my_handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			      (const TYPE *) &alpha, hMat->descr, hMat->hybA, 
			      (const TYPE *) vX, (const TYPE *) &beta,
			      (TYPE *) vY);
  if (rc == 0) 
    rc = cusparseSetMatType(hMat->descr,type);
  return(rc);
}

int T_HYBGDeviceAlloc(T_Hmat *Matrix,int nr, int nc, int nz)
{
  T_HYBGDeviceMat *hMat;
  int nr1=nr, nz1=nz, rc;
  if ((nr<0)||(nc<0)||(nz<0)) 
    return((int) CUSPARSE_STATUS_INVALID_VALUE);
  if ((hMat=(T_HYBGDeviceMat *) malloc(sizeof(T_HYBGDeviceMat)))==NULL)
    return((int) CUSPARSE_STATUS_ALLOC_FAILED);
  hMat->m  = nr;
  hMat->n  = nc;
  hMat->nz = nz;
  /* if (nr1 == 0) nr1 = 1; */
  /* if (nz1 == 0) nz1 = 1; */
  /* if ((rc= allocRemoteBuffer(((void **) &(hMat->irp)), ((nr1+1)*sizeof(int)))) != 0) */
  /*   return(rc); */
  /* if ((rc= allocRemoteBuffer(((void **) &(hMat->ja)), ((nz1)*sizeof(int)))) != 0) */
  /*   return(rc); */
  /* if ((rc= allocRemoteBuffer(((void **) &(hMat->val)), ((nz1)*sizeof(TYPE)))) != 0) */
  /*   return(rc); */
  if ((rc= cusparseCreateMatDescr(&(hMat->descr))) !=0) 
    return(rc);
  if ((rc= cusparseCreateSolveAnalysisInfo(&(hMat->triang))) !=0)
    return(rc);
  if((rc = cusparseCreateHybMat(&(hMat->hybA))) != 0)
    return(rc);
  Matrix->mat = hMat;
  return(CUSPARSE_STATUS_SUCCESS);
}

int T_HYBGDeviceSetMatDiagType(T_Hmat *Matrix, int type)
{
  T_HYBGDeviceMat *hMat= Matrix->mat;
  return ((int) cusparseSetMatDiagType(hMat->descr,type));
}

int T_HYBGDeviceSetMatIndexBase(T_Hmat *Matrix, int type)
{
  T_HYBGDeviceMat *hMat= Matrix->mat;
  return ((int) cusparseSetMatIndexBase(hMat->descr,type));
}

int T_HYBGDeviceSetMatType(T_Hmat *Matrix, int type)
{
  T_HYBGDeviceMat *hMat= Matrix->mat;
  return ((int) cusparseSetMatType(hMat->descr,type));
}

int T_HYBGDeviceSetMatFillMode(T_Hmat *Matrix, int type)
{
  T_HYBGDeviceMat *hMat= Matrix->mat;
  return ((int) cusparseSetMatFillMode(hMat->descr,type));
}

int T_spsvHYBGDevice(T_Hmat *Matrix, TYPE alpha, void *deviceX,
		     TYPE beta, void *deviceY)
{
  //beta??
  T_HYBGDeviceMat *hMat=Matrix->mat;
  struct MultiVectDevice *x = (struct MultiVectDevice *) deviceX;
  struct MultiVectDevice *y = (struct MultiVectDevice *) deviceY; 
  void *vX, *vY;
  int r,n;
  cusparseHandle_t *my_handle=getHandle();
  /*getAddrMultiVecDevice(deviceX, &vX);
    getAddrMultiVecDevice(deviceY, &vY); */
  vX=x->v_;
  vY=y->v_;

  return cusparseThybsv_solve(*my_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
			      (const TYPE *) &alpha, hMat->descr,
			      hMat->hybA, hMat->triang,
			      (const TYPE *) vX,  (TYPE *) vY);
}

int T_HYBGDeviceHybsmAnalysis(T_Hmat *Matrix)
{
  T_HYBGDeviceMat *hMat= Matrix->mat;  
  cusparseSolveAnalysisInfo_t info;
  int rc;
  cusparseHandle_t *my_handle=getHandle();

  /* rc = (int) cusparseGetMatType(hMat->descr); */
  /* fprintf(stderr,"Analysis MatType: %d\n",rc); */
  /* rc = (int) cusparseGetMatDiagType(hMat->descr); */
  /* fprintf(stderr,"Analysis DiagType: %d\n",rc); */
  /* rc = (int) cusparseGetMatFillMode(hMat->descr); */
  /* fprintf(stderr,"Analysis FillMode: %d\n",rc); */
  rc = (int) cusparseThybsv_analysis(*my_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,
				     hMat->descr, hMat->hybA, hMat->triang);

  if (rc !=0) {
    fprintf(stderr,"From csrsv_analysis: %d\n",rc);
  }
  return(rc);
}

int T_HYBGHost2Device(T_Hmat *Matrix, int m, int n, int nz,
		      int *irp, int *ja, TYPE *val) 
{
  int rc; double t1,t2;
  int nr1=m, nz1=nz;
  T_HYBGDeviceMat *hMat= Matrix->mat;
  cusparseHandle_t *my_handle=getHandle();

  if (nr1 == 0) nr1 = 1;
  if (nz1 == 0) nz1 = 1;
  if ((rc= allocRemoteBuffer(((void **) &(hMat->irp)), ((nr1+1)*sizeof(int)))) != 0)
    return(rc);
  if ((rc= allocRemoteBuffer(((void **) &(hMat->ja)), ((nz1)*sizeof(int)))) != 0)
    return(rc);
  if ((rc= allocRemoteBuffer(((void **) &(hMat->val)), ((nz1)*sizeof(TYPE)))) != 0)
    return(rc);

  if ((rc=writeRemoteBuffer((void *) irp, (void *) hMat->irp, 
			    (m+1)*sizeof(int)))
      != SPGPU_SUCCESS) 
    return(rc);
  
  if ((rc=writeRemoteBuffer((void *) ja,(void *) hMat->ja, 
			    (nz)*sizeof(int)))
      != SPGPU_SUCCESS) 
    return(rc);
  if ((rc=writeRemoteBuffer((void *) val, (void *) hMat->val, 
			    (nz)*sizeof(TYPE)))
      != SPGPU_SUCCESS) 
    return(rc);
  /* rc = (int) cusparseGetMatType(hMat->descr); */
  /* fprintf(stderr,"Conversion MatType: %d\n",rc); */
  /* rc = (int) cusparseGetMatDiagType(hMat->descr); */
  /* fprintf(stderr,"Conversion DiagType: %d\n",rc); */
  /* rc = (int) cusparseGetMatFillMode(hMat->descr); */
  /* fprintf(stderr,"Conversion FillMode: %d\n",rc); */
  //t1=etime();
  rc = (int) cusparseTcsr2hyb(*my_handle, m, n,
		   hMat->descr, 
		   (const TYPE *)hMat->val,
		   (const int *)hMat->irp, (const int *)hMat->ja, 
		   hMat->hybA,0,
		   CUSPARSE_HYB_PARTITION_AUTO);

  freeRemoteBuffer(hMat->irp);  hMat->irp = NULL;
  freeRemoteBuffer(hMat->ja);   hMat->ja  = NULL;
  freeRemoteBuffer(hMat->val);  hMat->val = NULL;

  //cudaSync();
  //t2 = etime();
  //fprintf(stderr,"Inner call to cusparseTcsr2hyb: %lf\n",(t2-t1));
  if (rc != 0) {
    fprintf(stderr,"From csr2hyb: %d\n",rc);
  }
  return(rc);
}



