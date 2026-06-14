#!/usr/bin/env python3
"""
梯形法数值积分 - MPI 并行版（带计时）
用于 B-2 性能测试
"""

from mpi4py import MPI
import time

def f(x):
    return x * x

def trapezoidal_serial(a, b, n):
    h = (b - a) / n
    integral = (f(a) + f(b)) / 2.0
    for i in range(1, n):
        x = a + i * h
        integral += f(x)
    integral *= h
    return integral

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # 固定问题规模：积分区间 [0,1]，总子区间数 10^7（1000万）
    a, b = 0.0, 1.0
    n_total = 10000000  # 10^7 个点
    
    # 开始计时（只计时计算部分）
    comm.Barrier()  # 同步所有进程
    start_time = time.time()
    
    # 隐式 Scatter：分发子区间
    h_global = (b - a) / size
    local_a = a + rank * h_global
    local_b = local_a + h_global
    
    local_n = n_total // size
    if rank == size - 1:
        local_n = n_total - (size - 1) * local_n
    
    # 局部计算
    local_integral = trapezoidal_serial(local_a, local_b, local_n)
    
    # Reduce 归约求和
    total_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)
    
    comm.Barrier()  # 同步所有进程
    end_time = time.time()
    
    if rank == 0:
        elapsed = end_time - start_time
        print(f"{size} {elapsed:.6f} {total_integral:.11f}")
