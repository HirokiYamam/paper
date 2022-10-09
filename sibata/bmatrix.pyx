cimport numpy as np
import numpy as np

#引数は r, nel, D
#返り値は tk, B, C, tvol

ctypedef np.float64_t f64
ctypedef np.int32_t i32

#全体剛性行列の組み立て
cpdef tkmatrix(np.ndarray[f64, ndim=2] r, np.ndarray[i32, ndim=2] nel, np.ndarray[f64, ndim=2] D, int mem):
    cdef int mel = mem**3
    cdef np.ndarray[f64, ndim=4] tk_ = np.zeros(((mem+1)**3, (mem+1)**3, 3, 3))
    cdef np.ndarray[f64, ndim=4] ak = np.zeros((8, 8, 3, 3))
    cdef np.ndarray[f64, ndim=4] B = np.zeros((mel, 8, 6, 3))
    cdef np.ndarray[f64, ndim=4] C = np.zeros((mel, 8, 3, 3))
    cdef np.ndarray[f64, ndim=1] tvol = np.zeros((mel))
    cdef int m, i, j
    
    for m in range(mel):
        akmatrix(m, r, nel, ak, B, C, tvol, D)
        
        for i in range(8):
            for j in range(8):
                tk_[nel[m][i]][nel[m][j]] += ak[i][j]
                
    return tk_, B, C, tvol

#要素剛性行列の組み立て, Bマトリクス, Cマトリクスの計算
#ak, B, Cは引数で受け渡しする
cdef void akmatrix(int m, np.ndarray[f64, ndim=2] r, np.ndarray[i32, ndim=2] nel, np.ndarray[f64, ndim=4] ak, np.ndarray[f64, ndim=4] B, np.ndarray[f64, ndim=4] C, np.ndarray[f64, ndim=1] tvol, np.ndarray[f64, ndim=2] D):
    cdef np.ndarray[i32, ndim=2] lfour = np.array([[0,1,2,4], [3,2,1,7], [6,2,7,4], [1,4,5,7], [1,2,4,7], [0,1,3,5], [2,0,3,6], [0,4,5,6], [7,3,5,6], [0,3,6,5]], dtype="int32")
    cdef int t, i, j
    cdef double volum = 0
    cdef np.ndarray[f64, ndim=3] Bsub = np.zeros((4, 6, 3))
    cdef np.ndarray[f64, ndim=3] Csub = np.zeros((4, 3, 3))
    cdef np.ndarray[f64, ndim=4] aksub = np.zeros((4, 4, 3, 3))

    ak *= 0
    for t in range(10):
        volum = bmatrix(r[nel[m][lfour[t][0]]], r[nel[m][lfour[t][1]]], r[nel[m][lfour[t][2]]], r[nel[m][lfour[t][3]]], Bsub, Csub)
        aksub = Bsub.transpose((0,2,1)).dot(D).dot(Bsub).transpose((0,2,1,3))*volum 
        for i in range(4):
            for j in range(4):
                ak[lfour[t][i]][lfour[t][j]] += aksub[i][j]
            B[m][lfour[t][i]] += Bsub[i] * volum
            C[m][lfour[t][i]] += Csub[i] * volum

        tvol[m] += volum
    B[m] /= tvol[m]
    C[m] /= tvol[m]
    tvol[m] /= 2.
    ak /= 2.

    return

#四面体要素のBマトリクス, Cマトリクス, 体積を求める
#B, Cは引数で受け渡す 体積は戻り値で受け渡す
cdef float bmatrix(np.ndarray[f64, ndim=1] r0, np.ndarray[f64, ndim=1] r1, np.ndarray[f64, ndim=1] r2, np.ndarray[f64, ndim=1] r3, np.ndarray[f64, ndim=3] Bsub, np.ndarray[f64, ndim=3] Csub):

    cdef double volum = 0
    cdef double ai, bi, ci, di

    ai = det(r1[0], r1[1], r1[2], r2[0], r2[1], r2[2], r3[0], r3[1], r3[2])
    bi = det(1, r1[1], r1[2], 1, r2[1], r2[2], 1, r3[1], r3[2])*(-1)
    ci = det(r1[0], 1, r1[2], r2[0], 1, r2[2], r3[0], 1, r3[2])*(-1)
    di = det(r1[0], r1[1], 1, r2[0], r2[1], 1, r3[0], r3[1], 1)*(-1)
    volum += ai
    Bsub[0] = [[bi,0,0],[0,ci,0],[0,0,di],[0,di,ci],[di,0,bi],[ci,bi,0]]
    Csub[0] = [[0,-di,ci],[di,0,-bi],[-ci,bi,0]]

    ai = det(r2[0], r2[1], r2[2], r3[0], r3[1], r3[2], r0[0], r0[1], r0[2])*(-1)
    bi = det(1, r2[1], r2[2], 1, r3[1], r3[2], 1, r0[1], r0[2])
    ci = det(r2[0], 1, r2[2], r3[0], 1, r3[2], r0[0], 1, r0[2])
    di = det(r2[0], r2[1], 1, r3[0], r3[1], 1, r0[0], r0[1], 1)
    volum += ai
    Bsub[1] = [[bi,0,0],[0,ci,0],[0,0,di],[0,di,ci],[di,0,bi],[ci,bi,0]]
    Csub[1] = [[0,-di,ci],[di,0,-bi],[-ci,bi,0]]

    ai = det(r3[0], r3[1], r3[2], r0[0], r0[1], r0[2], r1[0], r1[1], r1[2])
    bi = det(1, r3[1], r3[2], 1, r0[1], r0[2], 1, r1[1], r1[2])*(-1)
    ci = det(r3[0], 1, r3[2], r0[0], 1, r0[2], r1[0], 1, r1[2])*(-1)
    di = det(r3[0], r3[1], 1, r0[0], r0[1], 1, r1[0], r1[1], 1)*(-1)
    volum += ai
    Bsub[2] = [[bi,0,0],[0,ci,0],[0,0,di],[0,di,ci],[di,0,bi],[ci,bi,0]]
    Csub[2] = [[0,-di,ci],[di,0,-bi],[-ci,bi,0]]

    ai = det(r0[0], r0[1], r0[2], r1[0], r1[1], r1[2], r2[0], r2[1], r2[2])*(-1)
    bi = det(1, r0[1], r0[2], 1, r1[1], r1[2], 1, r2[1], r2[2])
    ci = det(r0[0], 1, r0[2], r1[0], 1, r1[2], r2[0], 1, r2[2])
    di = det(r0[0], r0[1], 1, r1[0], r1[1], 1, r2[0], r2[1], 1)
    volum += ai
    Bsub[3] = [[bi,0,0],[0,ci,0],[0,0,di],[0,di,ci],[di,0,bi],[ci,bi,0]]
    Csub[3] = [[0,-di,ci],[di,0,-bi],[-ci,bi,0]]

    Bsub /= volum
    Csub /= volum*2.
    volum /= 6.
    
    return volum


cdef double det(double a11,double a12,double a13,double a21,double a22,double a23,double a31,double a32,double a33):
    return a11*a22*a33 + a12*a23*a31 + a13*a21*a32 - a13*a22*a31 - a11*a23*a32 - a12*a21*a33
