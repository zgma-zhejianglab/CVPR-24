import os
import time
from time import sleep
import subprocess
import _thread
import  datetime
import fl_util

def exec_cmd(cmd):
    return subprocess.call(cmd, shell=True)

def exec_func():
    ip = '127.0.0.1'
    port='33333'
    gpu_list = ['0']

    ## configure the experimental settings
    method = 0
    ratio = 0.9
    dataset = 'fmnist'

    str_method = ['FedAvg', 'ReTrain', 'FedEraser', 'KNOT', 'Hier-FUN', 'Random']
    num_clients=21
    test_bs=100

    if dataset == 'fmnist':
        epochs, bs, lr, lu = 100, 64, 0.0001, 100
    elif dataset == 'c10':
        epochs, bs, lr, lu = 200, 32, 0.0005, 20
    elif dataset == 'c100':
        epochs, bs, lr, lu = 200, 32, 0.001, 30

    kill_server_cmd = """ps -ef|grep fl_server_module.py |grep -v grep|awk  '{print "kill -9 " $2}' |sh"""
    kill_client_cmd = """ps -ef|grep fl_client_module.py |grep -v grep|awk  '{print "kill -9 " $2}' |sh"""
    os.system(kill_server_cmd)
    os.system(kill_client_cmd)

    ## specify the code path, results path and python environment path, such as
    # source_code_path = 'xxx/zgma/Hier-FUN/'
    # exp_results_path = 'xxx/Hier-FUN/results/'
    # python_path = 'xxx/miniconda/envs/zgma/bin/python3'
    source_code_path = '???'
    exp_results_path = '???'
    python_path = '???'
    now_time = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    dir_name = exp_results_path+dataset+'-'+str(num_clients)+'-'+str(ratio)+"-"+str_method[method]+"-"+str(bs)+"-"+str(lr)+"-"+str(epochs)+"-"+now_time
    
    ## start the thread for the server
    fl_util.create_dir(dir_name)
    cmd = 'cd ' + source_code_path + ";" + python_path + ' -u fl_server_module.py --dataset ' + dataset \
        + ' --num_clients ' + str(num_clients) + ' --ip ' + ip  +  ' --rank 0' + ' --lu '+ str(lu)\
        + ' --port ' + str(port)  + ' --epochs ' + str(epochs) \
        + ' --dir ' + dir_name + ' --batch_size ' + str(bs) + ' --method ' + str(method) +' --ratio '+str(ratio)\
        + ' --test_batch_size ' + str(test_bs) + ' --GPU ' + str(0) +' --lr ' + str(lr) + ' > '+dir_name+'/server-resluts.txt'
    
    ## start the thread for each device
    _thread.start_new_thread(exec_cmd, (cmd,))
    for i in range(num_clients):
        gpu = int(i % len(gpu_list))
        cmd = 'cd ' + source_code_path  + ";" + python_path + ' -u fl_client_module.py --dataset ' + dataset  \
            + ' --num_clients ' + str(num_clients) + ' --ip ' + ip  +  ' --rank ' + str(i+1) + ' --lu '+ str(lu)\
            + ' --port ' + str(port) + ' --epochs ' + str(epochs) +' --ratio '+str(ratio)\
            + ' --dir ' + dir_name   + ' --batch_size ' + str(bs) + ' --method ' + str(method)\
            + ' --test_batch_size ' + str(test_bs) + ' --GPU ' + str(gpu_list[gpu]) +' --lr ' + str(lr)\
            + ' > '+dir_name+'/client-'+str(i+1)+'-resluts.txt'
        
        _thread.start_new_thread(exec_cmd, (cmd,))
    start_time = time.time()

    while True:
        end_time = time.time()
        if end_time - start_time >= 1500:
            break

exec_func()
