import subprocess
import os

def start_node(directory, script_name):
    """
    启动指定目录下的Python脚本
    :param directory: 目录路径
    :param script_name: 脚本文件名
    """
    script_path = os.path.join(directory, script_name)
    process = subprocess.Popen(['python', script_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1, universal_newlines=True)
    return process

def main():
    # 设置要启动的节点及其目录
    nodes = {
        'server': './server/federated_node.py',
        'client1': './client1/federated_node.py',
        'client2': './client2/federated_node.py'
    }

    processes = []

    # 启动每个节点
    for node, script in nodes.items():
        print(f"Starting {node}...")
        process = start_node(os.path.dirname(script), os.path.basename(script))
        processes.append((node, process))
        print(f"{node} started.")

    try:
        # 实时打印输出
        while processes:
            for node, process in processes:
                stdout_line = process.stdout.readline()
                if stdout_line:
                    print(f"{node} stdout: {stdout_line.strip()}")
                stderr_line = process.stderr.readline()
                if stderr_line:
                    print(f"{node} stderr: {stderr_line.strip()}")
                if process.poll() is not None and not stdout_line and not stderr_line:
                    processes.remove((node, process))
                    print(f"{node} process finished.")
                    break
    except KeyboardInterrupt:
        print("Terminating all nodes...")
        for node, process in processes:
            process.terminate()
            print(f"{node} terminated.")
    finally:
        for node, process in processes:
            process.kill()

if __name__ == '__main__':
    main()
