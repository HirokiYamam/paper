# -*- coding: utf-8 -*-
import multiprocessing as mp
import time
import sys
import numpy as np
import os
import sub



#目的関数
tension_target = np.array([
    [0.01592827,	409.929078],
    [0.01592827,	409.929078],
    [0.025632911,	411.3475177],
    [0.025632911,	411.3475177],
    [0.03164557,	421.2765957],
    [0.03164557,	421.2765957],
    [0.040189873,	434.0425532],
    [0.040189873,	434.0425532],
    [0.048839662,	455.3191489],
    [0.048839662,	455.3191489]])


compression_target = np.array([
    [0.007805907,	703.5460993],
    [0.016139241,	748.9361702],
    [0.024578059,	778.7234043],
    [0.033122363,	800],
    [0.041666667,	821.2765957],
    [0.050843882,	836.8794326],
    [0.059599156,	852.4822695],
    [0.068248945,	863.8297872],
    [0.07721519,	879.4326241],
    [0.085232068,	890.7801418]])



#曲線curveと点bの縦軸方向の2乗距離を求める
def distance(curve, b):
    length = len(curve)
    for i in range(length):
        if(curve[i][0]>b[0]):
            break
    c = curve[i] - b
    d = c[0]**2 + c[1]**2
    return d


#評価関数
def evaluate(ssCurve, target):
    total = 0.
    for d in target:
        total += distance(ssCurve, d)
    mse = total / len(target)
    return mse**0.5
    

def fepm(x, send_rev, n):
    print("====process no {} start====".format(n))
    showProgress = True if n==0 else False
    result_tens = sub.test(_a1=x[0], _a2=x[1], _a3=x[2], _s0=x[3], _hard=x[4], _deex=0.00001, _max_strain=0.1, _showProgress=showProgress)
    result_comp = sub.test(_a1=x[0], _a2=x[1], _a3=x[2], _s0=x[3], _hard=x[4], _deex=-0.00001, _max_strain=0.1, _showProgress=showProgress)
    send_rev.send([result_tens, result_comp])
    print("====process no {} end====".format(n))

#遺伝子を作る
def generateChild(parent, N, mutation):
    children = np.zeros((N, 5))
    if mutation=="hard":    
        for i in range(N-1):
            children[i] = parent
            children[i][3] += parent[3]*((np.random.rand()-0.5)*0.05)
            children[i][4] += parent[4]*((np.random.rand()-0.5)*0.05)
    elif mutation == "a":
        for i in range(N-1):
            children[i] = parent
            children[i][0] += (np.random.rand()-0.5)*0.05
            children[i][1] += (np.random.rand()-0.5)*0.05
            children[i][2] += (np.random.rand()-0.5)*0.05
    else:
        print("something went wrong")
        sys.exit()
    
    children[N-1] = parent
    return children

#最適化処理のメイン
#epoch
#best_gene
def optimize(epoch, parent):

    print("epoch: {}".format(epoch))
    jobs = []
    pipes = []
    results = []
    evaluations = []

    #10世代おきに加工硬化則の最適化とa値の最適化を交互に行なう
    if int(epoch/10.)%2 == 0:
        mutation = "hard"
    else:
        mutation = "a"
    children = generateChild(parent, N, mutation)

    #並列処理
    for n in range(N):
        get_rev, send_rev = mp.Pipe(False)
        job = mp.Process(target=fepm, args=(children[n], send_rev, n))
        jobs.append(job)
        pipes.append(get_rev)
        job.start()
    
    #並列処理が終了するのを待つ
    for job in jobs:
        job.join()
    
    #値を回収する
    for pipe in pipes:
        result = pipe.recv()
        results.append(result)

    #評価値を計算
    for r in results:
        evaluation = evaluate(r[0], tension_target)/2 + evaluate(r[1], compression_target)/2
        evaluations.append(evaluation)


    print("evaluations:\n", evaluations)
    with open(path+"/evaluations.txt", mode="a") as file:
        file.write("\n{}".format(evaluations[-1]))

    #最も小さい評価値を探す
    minimum = 1.0e10
    index = 0
    for i in range(len(evaluations)):
        if(minimum>evaluations[i]):
            minimum = evaluations[i]
            index = i
    
    #親を交代
    parent = children[index].copy()

    print("next parent:\n", parent)
    print("process_time: {} min.".format((time.time()-start_time)/60))
    with open(path+"/best_gene.txt", mode="a") as file:
        file.write("gene:{},eval:{}\n".format(parent, evaluations[-1]))
    return parent

#ファイルパス
path = ""
N = 24 #並列数(=1世代の人数)
max_epoch = 400 # 最適化(optimize())の実行回数
init = np.array([0.1, 0.1, 0.1, 1500, 0.2])

start_time = time.time()

if __name__=="__main__":

    #結果を書き込むディレクトリを作成
    data_directry_num = 0
    while(os.path.exists("data{}".format(data_directry_num))):
        data_directry_num += 1
    os.makedirs("data{}".format(data_directry_num))
    path = "data{}".format(data_directry_num)
    parent = init

    #最適化をmax_epoch回数実行する
    for i in range(max_epoch):
        parent = optimize(i, parent)
    



