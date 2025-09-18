#!/usr/bin/env bash
set -euo pipefail

 # 使用当前环境的 Python
PY_BIN="$(which python)"
if [[ -z "$PY_BIN" ]]; then
  echo "!!! 未找到 python，请先激活对应环境后再运行本脚本。"
  exit 1
fi
DIR="$(cd "$(dirname "$0")" && pwd)"

FILES=( "federated_node_test.py" "federated_node_test2.py" "federated_node_test3.py" )
PORTS=( 5001 5002 5003 )
CLIENT_ID="127.0.0.1"

mkdir -p "$DIR/logs"

echo ">>> 清理老进程（如果有）：${PORTS[*]}"
if command -v lsof >/dev/null 2>&1; then
  for p in "${PORTS[@]}"; do
    if lsof -ti tcp:"$p" >/dev/null 2>&1; then
      lsof -ti tcp:"$p" | xargs kill -9 || true
    fi
  done
else
  echo "!!! 未找到 lsof，跳过按端口清理。"
fi

pids=()
for i in "${!FILES[@]}"; do
  f="${FILES[$i]}"
  port="${PORTS[$i]}"
  log="$DIR/logs/node$((i+1)).log"

  if [[ ! -f "$DIR/$f" ]]; then
    echo "!!! 找不到文件：$DIR/$f"
    exit 1
  fi

  echo ">>> 启动：$f --client-id $CLIENT_ID --port $port  (日志：$log)"
  # 关键：禁止 Flask debug 重启器，否则会双进程导致脚本抓不到真正的 PID
  env WERKZEUG_RUN_MAIN=1 FLASK_DEBUG=0 \
    "$PY_BIN" "$DIR/$f" --client-id "$CLIENT_ID" --port "$port" >"$log" 2>&1 &
  pids+=( "$!" )
done

cleanup() {
  echo
  echo ">>> 收到退出信号，正在关闭子进程：${pids[*]}"
  for pid in "${pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done
  wait || true
  echo ">>> 已全部关闭"
}
# 加上 EXIT，脚本异常退出也能清理
trap cleanup EXIT INT TERM

echo ">>> 全部已启动（日志在 ./logs/ 下）："
echo "    - http://127.0.0.1:5001"
echo "    - http://127.0.0.1:5002"
echo "    - http://127.0.0.1:5003"
echo ">>> tail -f logs/node1.log  可查看实时日志"
wait