import torch
import fl_util
import torch.distributed as dist

## send data to the device with rank of 'dst'
def send_data(data, dst):
    flatten_data = fl_util.flatten_paras(data)
    dist.send(flatten_data.cpu(), dst=dst)

## receive data from the device with rank of 'src', and the data size is "data_size"
def recv_data(src, data_size):
    buf = torch.zeros([data_size]).float()
    dist.recv(buf, src=src)
    return buf
