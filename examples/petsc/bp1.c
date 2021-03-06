//                        libCEED + PETSc Example: BP1
//
// This example demonstrates a simple usage of libCEED with PETSc to solve the
// CEED BP1 benchmark problem, see http://ceed.exascaleproject.org/bps.
//
// The code is intentionally "raw", using only low-level communication
// primitives.
//
// Build with:
//
//     make bp1 [PETSC_DIR=</path/to/mfem>] [CEED_DIR=</path/to/libceed>]
//
// Sample runs:
//
//     bp1
//     bp1 -ceed /cpu/self
//     bp1 -ceed /gpu/occa
//     bp1 -ceed /cpu/occa
//     bp1 -ceed /omp/occa
//     bp1 -ceed /ocl/occa

const char help[] = "Solve CEED BP1 using PETSc\n";

#include <petscksp.h>
#include <ceed.h>
#include <stdbool.h>

static void Split3(PetscInt size, PetscInt m[3], bool reverse) {
  for (PetscInt d=0,sizeleft=size; d<3; d++) {
    PetscInt try = (PetscInt)PetscCeilReal(PetscPowReal(sizeleft, 1./(3 - d)));
    while (try * (sizeleft / try) != sizeleft) try++;
    m[reverse ? 2-d : d] = try;
    sizeleft /= try;
  }
}

static PetscInt Max3(const PetscInt a[3]) {
  return PetscMax(a[0], PetscMax(a[1], a[2]));
}
static PetscInt Min3(const PetscInt a[3]) {
  return PetscMin(a[0], PetscMin(a[1], a[2]));
}
static void GlobalDof(const PetscInt p[3], const PetscInt irank[3],
                      PetscInt degree, const PetscInt melem[3],
                      PetscInt mdof[3]) {
  for (int d=0; d<3; d++)
    mdof[d] = degree*melem[d] + (irank[d] == p[d]-1);
}
static PetscInt GlobalStart(const PetscInt p[3], const PetscInt irank[3],
                            PetscInt degree, const PetscInt melem[3]) {
  PetscInt start = 0;
  // Dumb brute-force is easier to read
  for (PetscInt i=0; i<p[0]; i++) {
    for (PetscInt j=0; j<p[1]; j++) {
      for (PetscInt k=0; k<p[2]; k++) {
        PetscInt mdof[3], ijkrank[] = {i,j,k};
        if (i == irank[0] && j == irank[1] && k == irank[2]) return start;
        GlobalDof(p, ijkrank, degree, melem, mdof);
        start += mdof[0] * mdof[1] * mdof[2];
      }
    }
  }
  return -1;
}
static int CreateRestriction(Ceed ceed, const CeedInt melem[3],
                             CeedInt P,
                             CeedElemRestriction *Erestrict) {
  const PetscInt Nelem = melem[0]*melem[1]*melem[2];
  PetscInt mdof[3], *idx, *idxp;

  for (int d=0; d<3; d++) mdof[d] = melem[d]*(P-1) + 1;
  idxp = idx = malloc(Nelem*P*P*P*sizeof idx[0]);
  for (CeedInt i=0; i<melem[0]; i++) {
    for (CeedInt j=0; j<melem[1]; j++) {
      for (CeedInt k=0; k<melem[2]; k++,idxp += P*P*P) {
        for (CeedInt ii=0; ii<P; ii++) {
          for (CeedInt jj=0; jj<P; jj++) {
            for (CeedInt kk=0; kk<P; kk++) {
              if (0) { // This is the C-style (i,j,k) ordering that I prefer
                idxp[(ii*P+jj)*P+kk] = (((i*(P-1)+ii)*mdof[1]
                                         + (j*(P-1)+jj))*mdof[2]
                                        + (k*(P-1)+kk));
              } else { // (k,j,i) ordering for consistency with MFEM example
                idxp[ii+P*(jj+P*kk)] = (((i*(P-1)+ii)*mdof[1]
                                         + (j*(P-1)+jj))*mdof[2]
                                        + (k*(P-1)+kk));
              }
            }
          }
        }
      }
    }
  }
  CeedElemRestrictionCreate(ceed, Nelem, P*P*P, mdof[0]*mdof[1]*mdof[2],
                            CEED_MEM_HOST, CEED_OWN_POINTER, idx, Erestrict);
  PetscFunctionReturn(0);
}

static int Setup(void *ctx, void *qdata, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {
  CeedScalar *rho = qdata, *target = rho + Q;
  const CeedScalar (*x)[Q] = (const CeedScalar (*)[Q])in[0];
  const CeedScalar (*J)[3][Q] = (const CeedScalar (*)[3][Q])in[1];
  const CeedScalar *w = in[4];
  for (CeedInt i=0; i<Q; i++) {
    CeedScalar det = (+ J[0][0][i] * (J[1][1][i]*J[2][2][i] - J[1][2][i]*J[2][1][i])
                      - J[0][1][i] * (J[1][0][i]*J[2][2][i] - J[1][2][i]*J[2][0][i])
                      + J[0][2][i] * (J[1][0][i]*J[2][1][i] - J[1][1][i]*J[2][0][i]));
    rho[i] = det * w[i];
    target[i] = PetscSqrtScalar(PetscSqr(x[0][i]) + PetscSqr(x[1][i]) + PetscSqr(
                                  x[2][i]));
  }
  return 0;
}

static int Mass(void *ctx, void *qdata, CeedInt Q,
                const CeedScalar *const *in, CeedScalar *const *out) {
  bool *residual = ctx;
  CeedScalar *rho = qdata, *target = rho + Q;
  for (CeedInt i=0; i<Q; i++) {
    out[0][i] = rho[i] * (in[0][i] - (*residual ? target[i] : 0.));
  }
  return 0;
}

static int Error(void *ctx, void *qdata, CeedInt Q,
                 const CeedScalar *const *in, CeedScalar *const *out) {
  CeedScalar *maxerror = ctx;
  CeedScalar *rho = qdata, *target = rho + Q;
  for (CeedInt i=0; i<Q; i++) {
    *maxerror = PetscMax(*maxerror, PetscAbsScalar(in[0][i] - target[i]));
  }
  return 0;
}

typedef struct User_ *User;
struct User_ {
  VecScatter ltog;
  Vec Xloc, Yloc;
  CeedVector xceed, yceed;
  CeedOperator op;
  CeedVector qdata;
};

// We abuse this function to also compute residuals (an affine operation) and to
// compute pointwise error (with no output vector Y).
static PetscErrorCode MatMult_Mass(Mat A, Vec X, Vec Y) {
  PetscErrorCode ierr;
  User user;
  PetscScalar *x, *y;

  PetscFunctionBeginUser;
  ierr = MatShellGetContext(A, &user); CHKERRQ(ierr);
  ierr = VecScatterBegin(user->ltog, X, user->Xloc, INSERT_VALUES,
                         SCATTER_REVERSE); CHKERRQ(ierr);
  ierr = VecScatterEnd(user->ltog, X, user->Xloc, INSERT_VALUES, SCATTER_REVERSE);
  CHKERRQ(ierr);
  ierr = VecZeroEntries(user->Yloc); CHKERRQ(ierr);

  ierr = VecGetArrayRead(user->Xloc, (const PetscScalar**)&x); CHKERRQ(ierr);
  ierr = VecGetArray(user->Yloc, &y); CHKERRQ(ierr);
  CeedVectorSetArray(user->xceed, CEED_MEM_HOST, CEED_USE_POINTER, x);
  CeedVectorSetArray(user->yceed, CEED_MEM_HOST, CEED_USE_POINTER, y);

  CeedOperatorApply(user->op, user->qdata, user->xceed, user->yceed,
                    CEED_REQUEST_IMMEDIATE);

  ierr = VecRestoreArrayRead(user->Xloc, (const PetscScalar**)&x); CHKERRQ(ierr);
  ierr = VecRestoreArray(user->Yloc, &y); CHKERRQ(ierr);

  if (Y) {
    ierr = VecZeroEntries(Y); CHKERRQ(ierr);
    ierr = VecScatterBegin(user->ltog, user->Yloc, Y, ADD_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
    ierr = VecScatterEnd(user->ltog, user->Yloc, Y, ADD_VALUES, SCATTER_FORWARD);
    CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

int main(int argc, char **argv) {
  PetscInt ierr;
  MPI_Comm comm;
  char ceedresource[4096] = "/cpu/self";
  PetscInt degree, qextra, localdof, localelem, melem[3], mdof[3], p[3],
           irank[3], ldof[3], lsize;
  PetscMPIInt size, rank;
  VecScatter ltog;
  Ceed ceed;
  CeedBasis basisx, basisu;
  CeedElemRestriction Erestrictx, Erestrictu;
  CeedQFunction qf_setup, qf_mass, qf_error;
  CeedOperator op_setup, op_mass, op_error;
  CeedVector xcoord, qdata;
  CeedInt P, Q;
  Vec X, Xloc, rhs;
  Mat mat;
  KSP ksp;
  User user;

  ierr = PetscInitialize(&argc, &argv, NULL, help);
  if (ierr) return ierr;
  comm = PETSC_COMM_WORLD;
  ierr = PetscOptionsBegin(comm, NULL, "CEED BP1 in PETSc", NULL); CHKERRQ(ierr);
  degree = 1;
  ierr = PetscOptionsInt("-degree", "Polynomial degree of tensor product basis",
                         NULL, degree, &degree, NULL); CHKERRQ(ierr);
  qextra = 0;
  ierr = PetscOptionsInt("-qextra", "Number of extra quadrature points",
                         NULL, qextra, &qextra, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsString("-ceed", "CEED resource specifier",
                            NULL, ceedresource, ceedresource,
                            sizeof(ceedresource), NULL); CHKERRQ(ierr);
  localdof = 1000;
  ierr = PetscOptionsInt("-local",
                         "Target number of locally owned degrees of freedom per process",
                         NULL, localdof, &localdof, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd(); CHKERRQ(ierr);

  // Determine size of process grid
  ierr = MPI_Comm_size(comm, &size); CHKERRQ(ierr);
  Split3(size, p, false);

  // Find a nicely composite number of elements no less than localdof
  for (localelem = PetscMax(1, localdof / (degree*degree*degree)); ;
       localelem++) {
    Split3(localelem, melem, true);
    if (Max3(melem) / Min3(melem) <= 2) break;
  }

  // Find my location in the process grid
  ierr = MPI_Comm_rank(comm, &rank); CHKERRQ(ierr);
  for (int d=0,rankleft=rank; d<3; d++) {
    const int pstride[3] = {p[1]*p[2], p[2], 1};
    irank[d] = rankleft / pstride[d];
    rankleft -= irank[d] * pstride[d];
  }

  GlobalDof(p, irank, degree, melem, mdof);

  ierr = PetscPrintf(comm, "Process decomposition: %D %D %D\n",
                     p[0], p[1], p[2]); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Local elements: %D = %D %D %D\n", localelem,
                     melem[0], melem[1], melem[2]); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Owned dofs: %D = %D %D %D\n",
                     mdof[0]*mdof[1]*mdof[2], mdof[0], mdof[1], mdof[2]); CHKERRQ(ierr);

  {
    ierr = VecCreate(comm, &X); CHKERRQ(ierr);
    ierr = VecSetSizes(X, mdof[0] * mdof[1] * mdof[2], PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetUp(X); CHKERRQ(ierr);

    lsize = 1;
    for (int d=0; d<3; d++) {
      ldof[d] = melem[d]*degree + 1;
      lsize *= ldof[d];
    }
    ierr = VecCreate(PETSC_COMM_SELF, &Xloc); CHKERRQ(ierr);
    ierr = VecSetSizes(Xloc, lsize, PETSC_DECIDE); CHKERRQ(ierr);
    ierr = VecSetUp(Xloc); CHKERRQ(ierr);

    // Create local-to-global scatter
    PetscInt *ltogind;
    IS ltogis;
    PetscInt gstart[2][2][2], gmdof[2][2][2][3];

    for (int i=0; i<2; i++) {
      for (int j=0; j<2; j++) {
        for (int k=0; k<2; k++) {
          PetscInt ijkrank[3] = {irank[0]+i, irank[1]+j, irank[2]+k};
          gstart[i][j][k] = GlobalStart(p, ijkrank, degree, melem);
          GlobalDof(p, ijkrank, degree, melem, gmdof[i][j][k]);
        }
      }
    }

    ierr = PetscMalloc1(lsize, &ltogind); CHKERRQ(ierr);
    for (PetscInt i=0,ir,ii; ir=i>=mdof[0], ii=i-ir*mdof[0], i<ldof[0]; i++) {
      for (PetscInt j=0,jr,jj; jr=j>=mdof[1], jj=j-jr*mdof[1], j<ldof[1]; j++) {
        for (PetscInt k=0,kr,kk; kr=k>=mdof[2], kk=k-kr*mdof[2], k<ldof[2]; k++) {
          ltogind[(i*ldof[1]+j)*ldof[2]+k] =
            gstart[ir][jr][kr] + (ii*gmdof[ir][jr][kr][1]+jj)*gmdof[ir][jr][kr][2]+kk;
        }
      }
    }
    ierr = ISCreateGeneral(comm, lsize, ltogind, PETSC_OWN_POINTER, &ltogis);
    CHKERRQ(ierr);
    ierr = VecScatterCreate(Xloc, NULL, X, ltogis, &ltog); CHKERRQ(ierr);
    CHKERRQ(ierr);
    ierr = ISDestroy(&ltogis); CHKERRQ(ierr);
  }

  CeedInit(ceedresource, &ceed);
  P = degree + 1;
  Q = P + qextra;
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 1, P, Q, CEED_GAUSS, &basisu);
  CeedBasisCreateTensorH1Lagrange(ceed, 3, 3, 2, Q, CEED_GAUSS, &basisx);

  CreateRestriction(ceed, melem, P, &Erestrictu);
  CreateRestriction(ceed, melem, 2, &Erestrictx);
  {
    CeedScalar *xloc;
    CeedInt shape[3] = {melem[0]+1, melem[1]+1, melem[2]+1}, len =
                         shape[0]*shape[1]*shape[2];
    xloc = malloc(len*3*sizeof xloc[0]);
    for (CeedInt i=0; i<shape[0]; i++) {
      for (CeedInt j=0; j<shape[1]; j++) {
        for (CeedInt k=0; k<shape[2]; k++) {
          xloc[((i*shape[1]+j)*shape[2]+k) + 0*len] = 1.*(irank[0]*melem[0]+i) /
              (p[0]*melem[0]);
          xloc[((i*shape[1]+j)*shape[2]+k) + 1*len] = 1.*(irank[1]*melem[1]+j) /
              (p[1]*melem[1]);
          xloc[((i*shape[1]+j)*shape[2]+k) + 2*len] = 1.*(irank[2]*melem[2]+k) /
              (p[2]*melem[2]);
        }
      }
    }
    CeedVectorCreate(ceed, len*3, &xcoord);
    CeedVectorSetArray(xcoord, CEED_MEM_HOST, CEED_OWN_POINTER, xloc);
  }

  CeedQFunctionCreateInterior(ceed, 1, 3, 2*sizeof(CeedScalar),
                              CEED_EVAL_INTERP | CEED_EVAL_GRAD | CEED_EVAL_WEIGHT,
                              CEED_EVAL_NONE,
                              Setup, __FILE__ ":Setup", &qf_setup);
  CeedQFunctionCreateInterior(ceed, 1, 1, 2*sizeof(CeedScalar),
                              CEED_EVAL_INTERP, CEED_EVAL_INTERP,
                              Mass, __FILE__ ":Mass", &qf_mass);
  CeedQFunctionCreateInterior(ceed, 1, 1, 2*sizeof(CeedScalar),
                              CEED_EVAL_INTERP, CEED_EVAL_NONE,
                              Error, __FILE__ ":Error", &qf_error);

  CeedOperatorCreate(ceed, Erestrictx, basisx, qf_setup, NULL, NULL, &op_setup);
  CeedOperatorCreate(ceed, Erestrictu, basisu, qf_mass, NULL, NULL, &op_mass);
  CeedOperatorCreate(ceed, Erestrictu, basisu, qf_error, NULL, NULL, &op_error);

  CeedOperatorGetQData(op_setup, &qdata);
  CeedOperatorApply(op_setup, qdata, xcoord, NULL, CEED_REQUEST_IMMEDIATE);
  CeedVectorDestroy(&xcoord);

  ierr = PetscMalloc1(1, &user); CHKERRQ(ierr);
  user->ltog = ltog;
  user->Xloc = Xloc;
  ierr = VecDuplicate(Xloc, &user->Yloc); CHKERRQ(ierr);
  CeedVectorCreate(ceed, lsize, &user->xceed);
  CeedVectorCreate(ceed, lsize, &user->yceed);
  user->op = op_mass;
  user->qdata = qdata;

  ierr = MatCreateShell(comm, mdof[0]*mdof[1]*mdof[2], mdof[0]*mdof[1]*mdof[2],
                        PETSC_DECIDE, PETSC_DECIDE, user, &mat); CHKERRQ(ierr);
  ierr = MatShellSetOperation(mat, MATOP_MULT, (void(*)(void))MatMult_Mass);
  CHKERRQ(ierr);

  ierr = MatCreateVecs(mat, &rhs, NULL); CHKERRQ(ierr);
  ierr = VecZeroEntries(X); CHKERRQ(ierr);
  // We abuse MatMult_Mass to compute a residual using u=0, then use the negative for the RHS
  bool residual = true;
  CeedQFunctionSetContext(qf_mass, &residual, sizeof residual);
  ierr = MatMult_Mass(mat, X, rhs); CHKERRQ(ierr);
  residual = false;
  ierr = VecScale(rhs, -1.); CHKERRQ(ierr);

  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);
  {
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCJACOBI); CHKERRQ(ierr);
    ierr = PCJacobiSetType(pc, PC_JACOBI_ROWSUM); CHKERRQ(ierr);
    ierr = KSPSetType(ksp, KSPCG); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, 1e-10, PETSC_DEFAULT, PETSC_DEFAULT,
                            PETSC_DEFAULT); CHKERRQ(ierr);
  }
  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);
  ierr = KSPSetOperators(ksp, mat, mat); CHKERRQ(ierr);
  ierr = KSPSolve(ksp, rhs, X); CHKERRQ(ierr);
  {
    KSPType ksptype;
    KSPConvergedReason reason;
    PetscReal rnorm;
    PetscInt its;
    ierr = KSPGetType(ksp, &ksptype); CHKERRQ(ierr);
    ierr = KSPGetConvergedReason(ksp, &reason); CHKERRQ(ierr);
    ierr = KSPGetIterationNumber(ksp, &its); CHKERRQ(ierr);
    ierr = KSPGetResidualNorm(ksp, &rnorm); CHKERRQ(ierr);
    ierr = PetscPrintf(comm, "KSP %s %s iterations %D rnorm %e\n", ksptype,
                       KSPConvergedReasons[reason], its, (double)rnorm); CHKERRQ(ierr);
  }

  CeedScalar maxerror = 0;
  user->op = op_error;
  CeedQFunctionSetContext(qf_error, &maxerror, sizeof maxerror);
  ierr = MatMult_Mass(mat, X, NULL); CHKERRQ(ierr);
  ierr = PetscPrintf(comm, "Pointwise error (max) %e\n", (double)maxerror);
  CHKERRQ(ierr);

  ierr = VecDestroy(&rhs); CHKERRQ(ierr);
  ierr = VecDestroy(&X); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Xloc); CHKERRQ(ierr);
  ierr = VecDestroy(&user->Yloc); CHKERRQ(ierr);
  ierr = VecScatterDestroy(&ltog); CHKERRQ(ierr);
  ierr = MatDestroy(&mat); CHKERRQ(ierr);
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

  CeedVectorDestroy(&user->xceed);
  CeedVectorDestroy(&user->yceed);
  CeedOperatorDestroy(&op_setup);
  CeedOperatorDestroy(&op_mass);
  CeedOperatorDestroy(&op_error);
  CeedElemRestrictionDestroy(&Erestrictu);
  CeedElemRestrictionDestroy(&Erestrictx);
  CeedQFunctionDestroy(&qf_setup);
  CeedQFunctionDestroy(&qf_mass);
  CeedQFunctionDestroy(&qf_error);
  CeedBasisDestroy(&basisu);
  CeedBasisDestroy(&basisx);
  CeedDestroy(&ceed);
  ierr = PetscFree(user); CHKERRQ(ierr);
  return PetscFinalize();
}
