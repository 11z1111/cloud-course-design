#!/usr/bin/env python3
"""
梯形法数值积分 - MPI 并行版（阻塞通信）
基于 B-1 代码，添加计时功能
问题规模：n_total = 1000000
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
    # ============================================================
    # MPI 通信原语1：MPI.COMM_WORLD - 获取全局通信域
    # 数据流向：无数据传输，初始化 MPI 环境
    # ============================================================
    comm = MPI.COMM_WORLD
    
    # ============================================================
    # MPI 通信原语2：comm.Get_rank() - 获取当前进程编号
    # 数据流向：本地操作，无跨进程通信
    # ============================================================
    rank = comm.Get_rank()
    
    # ============================================================
    # MPI 通信原语3：comm.Get_size() - 获取总进程数
    # 数据流向：本地操作，无跨进程通信
    # ============================================================
    size = comm.Get_size()

    a, b = 0.0, 1.0
    n_total = 1000000

    # ============================================================
    # MPI 通信原语4：Scatter（隐式）- 分发子区间
    # 数据流向：每个进程根据 rank 自动计算自己的子区间
    #   rank0: [0.00, 0.25), rank1: [0.25, 0.50)
    #   rank2: [0.50, 0.75), rank3: [0.75, 1.00]
    # ============================================================
    h_global = (b - a) / size
    local_a = a + rank * h_global
    local_b = local_a + h_global

    local_n = n_total // size
    if rank == size - 1:
        local_n = n_total - (size - 1) * local_n

    # 局部计算（无通信）
    local_integral = trapezoidal_serial(local_a, local_b, local_n)

    # ============================================================
    # MPI 通信原语5：comm.reduce() - 阻塞归约求和
    # 数据流向：所有进程的 local_integral 求和到 root 进程（rank0）
    # ============================================================
    comm.Barrier()
    start_time = time.time()
    total_integral = comm.reduce(local_integral, op=MPI.SUM, root=0)
    comm.Barrier()
    end_time = time.time()
    elapsed = end_time - start_time

    if rank == 0:
        print(f"阻塞版 ({size} 进程): ∫({a}→{b}) x^2 dx = {total_integral:.11f}")
        print(f"运行时间: {elapsed:.6f} s")
