import os
cwd = os.getcwd()
import tensorflow as tf
from ctypes import *
import numpy as np
import csv
import pandas as pd
import datetime
import shutil
from multiprocessing import Pool, Lock, Value
import time

dt_now = datetime.datetime.now()
today = f'{dt_now.year-2000}_{dt_now.month}_{dt_now.day}'

output_dir = f'../data/1_DNN_learn'

# multsizeは並列処理の数
multisize = 12
dataset_size = 3300

# 各パラメータ
num_element = 10**3
M_base = 3.07
sigma_base = 300.0
hard_base = 0.5

# path of sofile
RDsofile = "./sofiles/taylorRD_211027.so"
DDsofile = "./sofiles/taylorDD_211027.so"
TDsofile = "./sofiles/taylorTD_211027.so"

def make_random_number(num_element, M_base, sigma_base, hard_base):
    phi = 2*np.pi*np.random.rand(num_element)
    theta = 2*np.pi*np.random.rand(num_element)
    psi = 2*np.pi*np.random.rand(num_element)
    taylor=np.full(num_element,M_base)
    sigma = abs(sigma_base*np.random.normal(loc = 1,scale = 0.2, size = num_element))
    hard = abs(hard_base*np.random.normal(loc = 1,scale = 0.2, size = num_element))
    return phi, theta, psi, taylor, sigma, hard

def fortran(argtypes,filename,phi,theta,psi,taylor,sigma,hard,result):
    add_np = np.ctypeslib.load_library(filename,".")
    add_np.main_.argtypes = argtypes #入力値の型指定
    add_np.main_.restype = c_void_p #戻り値の型(void限定)
    add_np.main_(phi,theta,psi,taylor,sigma,hard,result)
    return result

def save_data(output_dir, n, phi, theta, psi, taylor, sigma, hard, resultRD, resultDD, resultTD):
    input_data = np.zeros([1,6*num_element])
    input_data[0,:num_element] = phi
    input_data[0,num_element:2*num_element] = theta
    input_data[0,2*num_element:3*num_element] = psi
    input_data[0,3*num_element:4*num_element] = taylor
    input_data[0,4*num_element:5*num_element] = sigma
    input_data[0,5*num_element:6*num_element] = hard
    df_input = pd.DataFrame(np.array(input_data))
    input_data_name = f'{output_dir}/input/input{today}{n}.csv'
    df_input.to_csv(input_data_name)

    df_output = pd.DataFrame(np.array(resultRD))
    output_data_name = f'{output_dir}/output/RD/output{today}{n}.csv'
    df_output.to_csv(output_data_name)

    df_output = pd.DataFrame(np.array(resultDD))
    output_data_name = f'{output_dir}/output/DD/output{today}{n}.csv'
    df_output.to_csv(output_data_name)

    df_output = pd.DataFrame(np.array(resultTD))
    output_data_name = f'{output_dir}/output/TD/output{today}{n}.csv'
    df_output.to_csv(output_data_name)



def mainf(process_index, random_seed):
    np.random.seed(random_seed)
    argtype = np.ctypeslib.ndpointer(dtype=np.float64)
    argtypes = [argtype for i in range(7)]
    dt_now = datetime.datetime.now()
    phi, theta, psi, taylor, sigma, hard = make_random_number(num_element, M_base, sigma_base, hard_base)

    resultRD = np.zeros([20])
    resultRD = fortran(argtypes,RDsofile,phi,theta,psi,taylor,sigma,hard,resultRD)

    resultDD = np.zeros([20])
    resultDD = fortran(argtypes,RDsofile,phi,theta,psi,taylor,sigma,hard,resultDD)

    resultTD = np.zeros([20])
    resultTD = fortran(argtypes,RDsofile,phi,theta,psi,taylor,sigma,hard,resultTD)

    l.acquire()

    if not(any([i == 0 for i in resultRD]) | any([i == 0 for i in resultDD]) | any([i == 0 for i in resultTD])):
        save_data(output_dir, iter.value, phi, theta, psi, taylor, sigma, hard, resultRD, resultDD, resultTD)

        with iter.get_lock():
            iter.value = iter.value+1

    l.release()

    count.value+=1
    return iter.value
    print('------------------------------------------')

if __name__ == '__main__':
    start = time.time()
    date = datetime.datetime.now()
    l = Lock() 
    iter = Value('i',0) #Value : https://www.yoheim.net/blog.php?q=20170601
    count = Value('i',0)
    print("Calculating...")
    p = Pool(processes=multisize)
    a = list(range(dataset_size))
    b = list(range(dataset_size))
    np.random.shuffle(b)
    result_list = p.starmap(func=mainf, iterable=zip(a,b))                                                       
    p.close()
    total_time = time.time()-start
    total_time = datetime.timedelta(seconds=total_time)