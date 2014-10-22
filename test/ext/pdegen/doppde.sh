#!/bin/sh

exe=$1
shift
prec=$1
shift
n=$1
shift
max=$1
shift
incr=$1
shift
while (( n <= $max )) 
do
echo "Testing with $n"
   
./$exe  <<EOF
7                      Number of entries below this
BICGSTAB                 Iterative method BICGSTAB CGS  BICG BICGSTABL RGMRES
$prec                    Preconditioner NONE  DIAG  BJAC 
CSR                    Storage format for matrix A:  CSR COO JAD 
$n                     Domain size (acutal system is this**3)
2                      Stopping criterion
1000                   MAXIT
-1                     ITRACE
20                     IRST    restart for RGMRES  and BiCGSTABL
EOF
echo ""
echo ""
let n=$n+$incr

done

