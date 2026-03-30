#!/bin/bash

echo "正在查找并杀死所有相关的训练进程..."

# 1. 先找到所有包含 scripts/train.py 的进程 PID（匹配相对路径和绝对路径）
# 匹配: scripts/train.py 或 nova_fastbev/scripts/train.py
all_pids=$(pgrep -u $USER -f "scripts/train.py")
pids=""

# 进一步过滤：确保是 nova_fastbev 项目的进程（通过工作目录判断）
if [ -n "$all_pids" ]; then
    for pid in $all_pids; do
        cwd=$(readlink -f /proc/$pid/cwd 2>/dev/null || echo "")
        if echo "$cwd" | grep -q "nova_fastbev"; then
            pids="$pids $pid"
        fi
    done
fi

if [ -n "$pids" ]; then
    echo "找到 nova_fastbev 训练主进程:$pids"
    
    # 2. 对每个主进程，先杀死其所有子进程（包括 spawn 的 worker 进程）
    for pid in $pids; do
        # 获取该进程的所有子进程（递归获取所有后代进程）
        children=$(pgrep -P $pid)
        if [ -n "$children" ]; then
            echo "杀死主进程 $pid 的子进程: $children"
            kill -9 $children 2>/dev/null
        fi
    done
    
    # 3. 最后杀死主进程本身
    echo "杀死训练主进程:$pids"
    kill -9 $pids 2>/dev/null
else
    echo "未找到正在运行的 nova_fastbev 训练主进程"
fi

# 3. 额外保险：查找所有占用 GPU 且与 nova_fastbev 训练相关的僵尸进程
# 使用 nvidia-smi 来找到占用 GPU 的进程
if command -v nvidia-smi &> /dev/null; then
    echo "检查 GPU 占用情况..."
    # 获取当前用户占用 GPU 的所有 Python 进程 PID
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
    
    if [ -n "$gpu_pids" ]; then
        for pid in $gpu_pids; do
            # 检查这个 PID 是否属于当前用户，且是否与训练相关
            # 通过命令行和工作目录双重判断
            cmd=$(ps -u $USER -p $pid -o cmd= 2>/dev/null)
            cwd=$(readlink -f /proc/$pid/cwd 2>/dev/null || echo "")
            
            # 匹配条件：命令行包含 scripts/train.py，或者工作目录在 nova_fastbev 且是 Python 进程
            if echo "$cmd" | grep -q "scripts/train.py"; then
                echo "杀死占用 GPU 的僵尸训练进程: $pid"
                echo "  命令: $cmd"
                kill -9 $pid 2>/dev/null
            elif echo "$cwd" | grep -q "nova_fastbev" && echo "$cmd" | grep -q "python"; then
                echo "杀死占用 GPU 的僵尸训练进程: $pid"
                echo "  命令: $cmd"
                echo "  工作目录: $cwd"
                kill -9 $pid 2>/dev/null
            fi
        done
    else
        echo "未发现占用 GPU 的进程"
    fi
fi

# 4. 等待一小段时间确保进程完全退出
sleep 1

echo "清理完成！"
echo "当前 GPU 使用情况："
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi