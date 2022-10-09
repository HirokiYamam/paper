import numpy as np
from scipy.sparse.linalg import spsolve, splu
from scipy.sparse import csc_matrix
import math
import sys
import time
import fib

#memはfepmの1辺の要素数
mem = 10
mem1 = mem + 1
mel = mem**3

#ポアソン比とヤング率
poas = 1./3.
young = 185
young *= 1000

#ひずみステップ
deex = -0.00001
max_strain = 0.1

# non-schmid parameter
a1 = 0.19
a2 = 0.24
a3 = 0.32
# hardening low
taylor = 3.
s0 = 1180.
hard = 0.099 

#ノード番号
nel = np.zeros((mel, 8), dtype='int32')
#D行列
D = np.zeros((6, 6))
#要素応力　以後、テンソルは基本フォークト表記
s1 = np.zeros((mel, 6))
#ひずみ合計
gamma = np.zeros(mel)
#巨視的ひずみ
Ep = np.zeros(6)

#シュミットテンソル
eSchmid = np.zeros((mel, 24, 6))
wSchmid = np.zeros((mel, 24, 3))

#ノード位置
r = np.zeros((mem1**3, 3))

#オイラー角
phi = np.zeros(mel)
theta = np.zeros(mel)
psi = np.zeros(mel)

#要素体積
tvol = np.zeros(mel)

B_ = np.zeros((mel, 8, 6, 3))
C_ = np.zeros((mel, 8, 3, 3))


boundary_nodes = np.zeros(6*mem**2+2, dtype='int32')

#ひずみ増分
dep = np.zeros((mel, 6))
#スピン増分
dwp = np.zeros((mel, 3))
#スピン
dw = np.zeros((mel, 3))
#すべり系のひずみ和
dg = np.zeros((mel, 24))
#巨視的ひずみ
dEp = np.zeros(6)
dEp0 = np.zeros(6)
#巨視的応力
S = np.zeros(6)
#ノード変位
du = np.zeros((mem1**3, 3))
#すべり面法線ベクトル
n0ml = np.zeros((mel,24,3))
#微視的応力
s = np.zeros((mel, 6))
#応力-ひずみデータ
stress_strain_data = []
#lu分解したやつ
LU = 0
#円周率
pi = 3.1416

def initial():
    global s1, gamma, Ep
    dx = 1./mem
    
    for k in range(mem1):
        for j in range(mem1):
            for i in range(mem1):
                n = i + j*mem1 + k*mem1**2
                r[n] = [dx*i, dx*j, dx*k]
                
    
    for k in range(mem):
        for j in range(mem):
            for i in range(mem):
                m = i + j*mem + k*mem**2
                nel[m][0] = i + j*mem1 + k*mem1**2
                nel[m][1] = (i+1) + j*mem1 + k*mem1**2
                nel[m][2] = i + (j+1)*mem1 + k*mem1**2
                nel[m][3] = (i+1) + (j+1)*mem1 + k*mem1**2
                nel[m][4] = i + j*mem1 + (k+1)*mem1**2
                nel[m][5] = (i+1) + j*mem1 + (k+1)*mem1**2
                nel[m][6] = i + (j+1)*mem1 + (k+1)*mem1**2
                nel[m][7] = (i+1) + (j+1)*mem1 + (k+1)*mem1**2
                
    rambda = poas*young/(1 + poas)/(1 - poas*2)
    rigid = young/(2 + poas*2)
    for i in range(3):
        for j in range(3):
            D[i][j] = rambda
    D[0][0] = rambda + rigid*2
    D[1][1] = rambda + rigid*2
    D[2][2] = rambda + rigid*2
    D[3][3] = rigid
    D[4][4] = rigid
    D[5][5] = rigid


    s1 *= 0
    gamma *= 0
    Ep *=0

    index = 0
    for k in range(mem1):
        for j in range(mem1):
            for i in range(mem1):
                if(i==0 or j==0 or k==0 or i==mem or j==mem or k==mem):
                    n = i + j*mem1 + k*mem1**2
                    boundary_nodes[index] = n
                    index += 1
                    
    for i in range(mel):
        phi[i] = np.arccos(1.-2*random())   *1e-8
        theta[i] = random()*2.*math.pi      *1e-8
        psi[i] = random()*2.*math.pi        *1e-8


m0 = np.array([[1,1,1], [1,1,1], [1,1,1], [-1,1,1], [-1,1,1], [-1,1,1], [-1,-1,1], [-1,-1,1], [-1,-1,1], [1,-1,1], [1,-1,1], [1,-1,1], [-1,-1,-1], [-1,-1,-1], [-1,-1,-1], [1,-1,-1], [1,-1,-1], [1,-1,-1], [1,1,-1], [1,1,-1], [1,1,-1], [-1,1,-1], [-1,1,-1], [-1,1,-1]], dtype="float64")/3**0.5
n0 = np.array([[0,1,-1], [-1,0,1], [1,-1,0], [-1,0,-1], [0,-1,1], [1,1,0], [0,-1,-1], [1,0,1], [-1, 1, 0], [1,0,-1], [0,1,1], [-1,-1,0], [0,1,-1], [-1,0,1], [1,-1,0], [-1,0,-1], [0,-1,1], [1,1,0], [0,-1,-1], [1,0,1], [-1,1,0], [1,0,-1], [0,1,1], [-1, -1, 0]], dtype="float64")/2**0.5
n0_ = np.array([[-1,1,0], [0,-1,1], [1,0,-1], [-1,-1,0], [1,0,1], [0,1,-1], [1,-1,0], [0,1,1], [-1,0,-1], [1,1,0], [-1,0,1], [0,-1,-1], [1,0,-1], [-1,1,0], [0,-1,1], [0,-1,1], [-1,-1,0], [1,0,1], [-1,0,-1], [1,-1,0], [0,1,1], [0,-1,-1], [1,1,0], [-1,0,1]], dtype="float64")/2**0.5

def trans():
    Ce = np.zeros((24, 3, 3))
    Cw = np.zeros((24, 3, 3))
    
    for l in range(24):
        S0 = np.outer(m0[l], n0[l])
        S1 = np.outer(m0[l], n0_[l])
        S2 = np.outer(np.cross(n0[l], m0[l]), n0[l])
        S3 = np.outer(np.cross(n0_[l], m0[l]), n0_[l])
        S = S0 + a1*S1 + a2*S2 + a3*S3
        Ce[l] = (S + S.T)/2.
        Cw[l] = (S - S.T)/2.

    for m in range(mel):
        R = np.zeros((3, 3))
        R[0][0] = np.cos(phi[m])*np.cos(theta[m])*np.cos(psi[m]) - np.sin(theta[m])*np.sin(psi[m])
        R[0][1] = np.cos(phi[m])*np.sin(theta[m])*np.cos(psi[m]) + np.cos(theta[m])*np.sin(psi[m])
        R[0][2] = -np.sin(phi[m])*np.cos(psi[m])
        R[1][0] = -np.cos(phi[m])*np.cos(theta[m])*np.sin(psi[m]) - np.sin(theta[m])*np.cos(psi[m])
        R[1][1] = -np.cos(phi[m])*np.sin(theta[m])*np.sin(psi[m]) + np.cos(theta[m])*np.cos(psi[m])
        R[1][2] = np.sin(phi[m])*np.sin(psi[m])
        R[2][0] = np.sin(phi[m])*np.cos(theta[m])
        R[2][1] = np.sin(phi[m])*np.sin(theta[m])
        R[2][2] = np.cos(phi[m])
        
        for l in range(24):
            cee = R.T.dot(Ce[l]).dot(R)
            cww = R.T.dot(Cw[l]).dot(R)
            eSchmid[m][l][0] = cee[0][0]
            eSchmid[m][l][1] = cee[1][1]
            eSchmid[m][l][2] = cee[2][2]
            eSchmid[m][l][3] = cee[1][2]*2
            eSchmid[m][l][4] = cee[2][0]*2
            eSchmid[m][l][5] = cee[0][1]*2
            wSchmid[m][l][0] = cww[1][2]
            wSchmid[m][l][1] = cww[2][0]
            wSchmid[m][l][2] = cww[0][1]
            n0ml[m][l] = R.dot(n0[l])


seed = 0x12345678
def random():
    global seed
    mask = 0xffffffff
    seed = (seed^(seed<<13)) & mask
    seed = (seed^(seed>>17)) & mask
    seed = (seed^(seed<<15)) & mask
    return seed/2.**32


tk = np.zeros((mem1**3*3, mem1**3*3))

def tkmatrix():
    global tk, B_, C_, tvol
    tk_, B_, C_, tvol = fib.tkmatrix(r, nel, D, mem)
    tk = tk_.transpose((0,2,1,3)).reshape((mem1**3*3, mem1**3*3))
    for i in boundary_nodes:
        for j in range(3):
            n = i*3 + j
            tk[n][n] = 1.e20


def find():
    dU = np.zeros(3)
    de = np.zeros(6)
    f = np.zeros((mem1**3, 3))
    global dEp, dEp0, S, dwp, s, deex

    s1t = s1.T
    dwt = dw.T
    omega = np.array([  
        (s1t[4]*dwt[1] - s1t[5]*dwt[2])*2.,
        (s1t[5]*dwt[2] - s1t[3]*dwt[0])*2.,
        (s1t[3]*dwt[0] - s1t[4]*dwt[1])*2.,
        (s1t[1]-s1t[2])*dwt[0] + s1t[4]*dwt[2] - s1t[5]*dwt[1],
        (s1t[2]-s1t[0])*dwt[1] + s1t[5]*dwt[0] - s1t[3]*dwt[2],
        (s1t[0]-s1t[1])*dwt[2] + s1t[3]*dwt[1] - s1t[4]*dwt[0]]).T

    for m in range(mel):
        fp = B_[m].transpose((0,2,1)).dot(D.dot(dep[m]) - s1[m] - omega[m])*tvol[m]
        for i in range(8):
            f[nel[m][i]] += fp[i]
            
    #boundary-condition
    dEx = deex
    dEy = dEp[1] - poas*(deex-dEp[0])
    dEz = dEp[2] - poas*(deex-dEp[0])
    dGyz = dEp[3]
    dGzx = dEp[4]
    dGxy = dEp[5]
    for n in boundary_nodes:
        dU[0] = dEx*r[n][0] + dGxy*r[n][1] + dGzx*r[n][2]
        dU[1] = dEy*r[n][1] + dGyz*r[n][2]/2.
        dU[2] = dEz*r[n][2] + dGyz*r[n][1]/2.
        f[n] = dU*1.e20
    du = LU.solve(f.reshape(-1)).reshape((-1, 3))
    dt = (1. + poas)/young/2.
    Sigma = np.array([[S[0], S[3], S[5]],[S[3], S[1], S[4]],[S[5], S[4], S[2]]])
    
    for m in range(mel):
        du_m = np.zeros((8,3))
        for i in range(8):
            du_m[i] = du[nel[m][i]]

        de = B_[m].transpose((1,0,2)).reshape((6,24)).dot(du_m.reshape((-1)))
        dw[m] = C_[m].transpose((1,0,2)).reshape((3,24)).dot(du_m.reshape((-1)))
        s[m] = D.dot(de - dep[m]) + s1[m] + omega[m]
        
        yk = yield_(m)
        tau = eSchmid[m].dot(s[m])
        for l in range(24):
            sigma_l = Sigma.dot(n0ml[m][l]).dot(n0ml[m][l])
            tau[l] = (abs(tau[l]) + 1.732/2*a3*sigma_l)/(1 + a1/2)*np.sign(tau[l])
        xx = np.maximum(np.abs(dg[m]) + (np.abs(tau) - yk)*dt, 0)
        dg[m] = np.abs(xx) * np.sign(tau)
        dep[m] = eSchmid[m].T.dot(dg[m])
        dwp[m] = wSchmid[m].T.dot(dg[m])
        
    #巨視的ひずみ
    dEp = np.sum(dep, axis=0)/mel
    #巨視的応力
    S = np.sum(s, axis=0)/mel
    error = np.abs(dEp - dEp0).sum()
    dEp0 = dEp.copy()
    return error

def clear():
    global dep, dw, dg, dEp, dEp0
    dep *= 0.
    dw *= 0.
    dg *= 0.
    dEp *= 0.
    dEp0 *= 0.

def renew():
    global Ep, s1, dEp, r
    r += du
    Ep += dEp
    s1 = s.copy()
    dwr = dw - dwp
    for m in range(mel):
        gamma[m] += np.abs(dg[m]).sum()
        dphi = -dwr[m][0]*np.sin(theta[m]) + dwr[m][1]*np.cos(theta[m])
        if(abs(phi[m]<1.e-10)):
            phi[m] += dphi
        dpsi = (dwr[m][0]*np.cos(theta[m]) + dwr[m][1]*np.sin(theta[m]))/np.sin(phi[m])
        dtheta = -dpsi*np.cos(phi[m]) + dwr[m][2]
        theta[m] += dtheta
        phi[m] += dphi
        psi[m] += dpsi

def yield_(m):
    yk0 = s0/taylor**(1 + hard)
    ykmin = 5.
    gam = gamma[m] + np.abs(dg[m]).sum()
    yk = max(ykmin, yk0*gam**hard)
    return yk

def fepm():
    global LU
    num_divergence = 0
    errors = np.zeros((5000))
    initial()
    istep = 0
    while(True):
        trans()
        clear()
        tkmatrix()
        LU = splu(csc_matrix(tk))
        num_iter = 0
        while(True):
            error = find()
            errors[num_iter] = error
            num_iter += 1
            if(num_iter>50 or (errors[1]>errors[0]*0.9 and errors[3]>errors[0]*0.9 and errors[4]>errors[0]*0.9)):
                num_divergence += 1
                if(showProgress):
                    print("==========CAUTION::strain deverted==========")
                break
            if(error<1e-5):
                break
            
        renew()
        errors *= 0
        stress_strain_data.append([abs(Ep[0]), abs(S[0])])
        
        if(showProgress):
            length = len(stress_strain_data) - 1
            total_time = time.time()-calc_start_time
            remaining_time = total_time/(abs(Ep[0])+max_strain/50*0.5**length)*(max_strain - abs(Ep[0]))
            print(stress_strain_data[length][0], stress_strain_data[length][1],time.time()-calc_start_time, remaining_time)
            with open("res", mode="a") as file:
                file.write("\n{},{}".format(Ep, S))
        if(abs(Ep[0])>max_strain):
            if(showProgress):
                print("==================simulation fin==================")
                print("number of divergence: {}".format(num_divergence))
            return
        
        istep += 1

def test(_a1=0, _a2=0, _a3=0, _s0=360*3, _hard=0.25, _deex = 0.0005, _max_strain = 0.1, _showProgress=False):
    global a1, a2, a3, taylor, s0, hard, showProgress, deex, max_strain
    a1 = _a1
    a2 = _a2
    a3 = _a3
    s0 = _s0
    hard = _hard
    deex = _deex
    max_strain = _max_strain
    showProgress = True
    calc_start_time = time.time()
    fepm()
    return stress_strain_data


showProgress = False
calc_start_time = time.time()
if(__name__=="__main__"):
    showProgress = True
    fepm()
    print("calc time:{}".format(time.time()-calc_start_time))
    
    