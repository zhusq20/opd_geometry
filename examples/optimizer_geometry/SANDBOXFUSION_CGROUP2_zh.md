# SandboxFusion cgroup v2 构建与启动

官方 `server-20250609` 镜像的 `lite` runner 使用 cgroup v1 的
`cgcreate/cgexec`，在只提供 cgroup v2 的现代宿主机上会报
`Cgroup is not mounted`。本目录提供一个以官方镜像 digest 为基础的最小派生镜像：

- 直接使用 `cgroup.procs`、`memory.max`、`cpu.max` 和 `pids.max`；
- 每次执行使用独立 cgroup，并在结束或超时后杀死残留进程、删除 cgroup；
- 使用无 veth、无默认路由的 network namespace，禁止公网；
- 在新 PID namespace 内挂载 `/proc`，只提供最小 `/dev`；
- 每次执行使用独立的 mount/PID/network/IPC/UTS namespace，并只读暴露该执行自己的 cgroup controls；
- 以 uid 1000、空 capability、`no_new_privs` 运行生成代码；
- 在 Docker builtin seccomp 之上叠加仅作用于生成代码的 BPF，拒绝 user namespace、mount、setns、BPF、ptrace 等危险 syscall；
- 拒绝路径穿越与 symlink staging/fetch，对上传和取回文件分别设置总量上限，并禁用会绕过 lite 的 GPU/C#/Lean runner；
- 保留官方源文件哈希门禁：基础镜像内容不匹配时构建立即失败。

## 1. 宿主机要求

必须在 Linux amd64 宿主机执行，并且使用 rootful Docker 与 cgroup v2：

```bash
docker info --format \
  'CgroupVersion={{.CgroupVersion}} CgroupDriver={{.CgroupDriver}}'
```

`CgroupVersion` 必须为 `2`。构建和服务启动均需可用的 Docker daemon；服务仍需
经过白名单限定的 `SYS_ADMIN`/`NET_ADMIN` 等 capabilities；启动器会拒绝
`privileged=true`，并验证父 cgroup 只读、只有专用 subtree 可写。训练容器不需要 Docker socket
或任何 namespace 权限。

## 2. 构建并固定本地 image ID

以下命令必须在宿主机执行。项目代码始终使用 disk2 的真实路径；不要从可能优先
展示 disk3 文件的 `/mnt/data_from_server2/siqizhu4/iclr2027` 编辑或启动：

```bash
export HOST_ROOT=/mnt/data_from_server2/siqizhu4
export DISK2_ROOT=/mnt/disk2_from_server2/siqizhu4/iclr2027
export SLIME_REPO="$DISK2_ROOT/slime_opd_geometry"
export TRAIN_CACHE="$DISK2_ROOT/.training-cache/root-cache"
export SANDBOX_STATE="$HOME/.local/state/slime-opd-geometry/sandboxfusion"

test -f "$SLIME_REPO/examples/optimizer_geometry/run_single_task_rl.sh"
mkdir -p "$TRAIN_CACHE" "$SANDBOX_STATE"
chmod 700 "$SANDBOX_STATE"
cd "$SLIME_REPO"

SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
  bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh
```

脚本先针对固定基础镜像验证补丁，再执行两次禁用缓存的最终镜像构建。若 Docker
写入了不可复现的 atime/ctime 或目录 mtime 元数据，脚本会改为要求规范化后的运行时指纹
完全一致（镜像配置，以及所有补丁文件的内容、权限、属主、大小和 mtime；目录的权限与
属主），然后写入：

```text
$HOME/.local/state/slime-opd-geometry/sandboxfusion/sandboxfusion-image.env
```

其中 `SANDBOXFUSION_IMAGE=sha256:...` 是本机 Docker content-addressed image ID，
Compose 会按这个不可变 ID 启动，而不是按 mutable tag 启动。基础镜像很大，但第二次构建会
复用已经下载的官方 layers。

若需跨机器部署，可显式提供一个有写权限的临时 registry tag：

```bash
SANDBOXFUSION_PUSH_REF=registry.example.org/team/sandboxfusion:cgroup2-v1 \
SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
  bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh
```

脚本推送后会把 pin 改写为 `repository@sha256:...`，启动时仍只接受 digest。

## 3. 启动并运行主动探针

```bash
cd "$SLIME_REPO"

SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
SANDBOX_PREFLIGHT_MARKER="$SANDBOX_STATE/sandboxfusion_preflight.json" \
  bash examples/optimizer_geometry/start_sandboxfusion.sh
```

启动器会：

1. 验证 pin、镜像 ID、patch label 和完整 patch source hash；
2. 在宿主机 `/sys/fs/cgroup/sandboxfusion` 创建专用 delegated subtree；
3. 只把该 subtree 以读写方式挂入 SandboxFusion；
4. 让 API 控制面只加入无外部网关的 Docker internal bridge，并通过一个只读根、零 capability、
   固定上游的 TCP relay 把 endpoint 仅发布到 `127.0.0.1:8080`（Docker 28 不发布 internal network 上的端口）；
5. 主动验证执行、宿主文件不可见、禁公网、逐执行 namespace、只读 cgroup v2 控制值、实际内存 OOM、降权、seccomp/userns 拒绝和 timeout；
6. 通过训练实际使用的 `/submit` 跑一个最小 LiveCodeBench 样例，并确认客户端自定义提取 Python 与旧式 pickle 测试载荷均被拒绝；
7. 验证探针结束后没有遗留 execution cgroup、network namespace 或 overlay 目录。

只有全部通过才会生成 `safe=true` 的 marker：

```text
$HOME/.local/state/slime-opd-geometry/sandboxfusion/sandboxfusion_preflight.json
```

启动前旧 marker 会先被原子改写为 `safe=false`；任何一步失败都会关闭本次服务并保持 marker
无效。服务不会在崩溃或宿主重启后自动拉起；必须重新运行启动脚本和全部探针。不能改用
`SANDBOX_CONFIG=local`，也不能手工修改 marker。

## 4. 从训练容器访问

训练容器需共享宿主机网络，保持 SandboxFusion endpoint 仅发布到宿主 loopback。
运行中的容器不能追加 host network 或 bind mount；旧容器配置不一致时需确认没有
要保留的临时进程，再删除并重建。仍在宿主机 shell 中执行：

```bash
export HOST_ROOT=/mnt/data_from_server2/siqizhu4
export DISK2_ROOT=/mnt/disk2_from_server2/siqizhu4/iclr2027
export SLIME_REPO="$DISK2_ROOT/slime_opd_geometry"
export TRAIN_CACHE="$DISK2_ROOT/.training-cache/root-cache"
export SANDBOX_STATE="$HOME/.local/state/slime-opd-geometry/sandboxfusion"
export TRAIN_CONTAINER=iclr2027-code-grpo
export TRAIN_IMAGE=slimerl/slime:latest

test -f "$SLIME_REPO/examples/optimizer_geometry/run_single_task_rl.sh"
mkdir -p "$TRAIN_CACHE" "$SANDBOX_STATE"
chmod 700 "$SANDBOX_STATE"
test -s "$SANDBOX_STATE/sandboxfusion_preflight.json"

docker run -d \
  --name "$TRAIN_CONTAINER" \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e WANDB_API_KEY \
  -e WANDB_BASE_URL \
  -e FRESH_START=1 \
  -e M2RL_SANDBOX_PREFLIGHT_MARKER=/workspace/sandboxfusion-state/sandboxfusion_preflight.json \
  --mount "type=bind,src=${HOST_ROOT},dst=/workspace/dev" \
  --mount "type=bind,src=${DISK2_ROOT},dst=/workspace/dev/iclr2027" \
  --mount "type=bind,src=${TRAIN_CACHE},dst=/root/.cache" \
  --mount "type=bind,src=${SANDBOX_STATE},dst=/workspace/sandboxfusion-state,readonly" \
  -w /workspace/dev/iclr2027/slime_opd_geometry \
  "$TRAIN_IMAGE" \
  sleep infinity
```

第二个、更具体的 bind mount 会覆盖 `/workspace/dev/iclr2027`，所以仓库代码、
`outputs/`、训练 checkpoint 和 launcher 日志都会直接写入 disk2。第一个 mount
仍提供 `/workspace/dev/checkpoints` 等共享路径；SandboxFusion 状态目录在训练
容器中只读。

进入训练容器后：

```bash
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
export M2RL_SANDBOX_PREFLIGHT_MARKER=/workspace/sandboxfusion-state/sandboxfusion_preflight.json

test -s "$M2RL_SANDBOX_PREFLIGHT_MARKER"
python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8080/v1/ping").read().decode())'
```

预期返回 `"pong"`。训练容器只作为 HTTP 客户端，不需要 Docker daemon/socket。

从宿主机启动训练入口：

```bash
docker exec -it "$TRAIN_CONTAINER" bash -lc '
  cd /workspace/dev/iclr2027/slime_opd_geometry
  bash scripts/optimizer_geometry/start_code_grpo_in_container.sh
'
```

训练最终调用的 `run_single_task_rl.sh` 对应宿主机 disk2 上的：

```text
/mnt/disk2_from_server2/siqizhu4/iclr2027/slime_opd_geometry/examples/optimizer_geometry/run_single_task_rl.sh
```

容器主进程是 `sleep infinity`；断开 `docker exec` 或 SSH 不会自动删除容器。
`/root/.cache` 由 disk2 的 `TRAIN_CACHE` 持久化。通过 `apt`/`pip` 修改的容器系统
目录仍不会写回原镜像，需要通过 Dockerfile 固化。

## 5. 修改补丁后的规则

任何 `sandboxfusion_patch/` 文件变化都会改变 patch SHA-256；已有 pin 将被启动器拒绝，必须重新
运行构建脚本。更新官方基础镜像时，也必须重新审计 runner 并更新
`upstream-files.sha256`，不能只替换 `FROM` digest 绕过源文件哈希门禁。

专用 cgroup 的聚合默认上限为 32 GiB 和 4096 PID，服务控制面本身为 4 GiB/2048 PID。
宿主资源较小时可在构建后、启动前调整前两项（值必须是正整数，单位为 byte/PID）：

```bash
export SANDBOXFUSION_AGGREGATE_MEMORY_BYTES=$((16 * 1024 * 1024 * 1024))
export SANDBOXFUSION_AGGREGATE_PIDS=2048
SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
SANDBOX_PREFLIGHT_MARKER="$SANDBOX_STATE/sandboxfusion_preflight.json" \
  bash examples/optimizer_geometry/start_sandboxfusion.sh
```

实际值会进入安全 marker；不能把它们设置为 `max`。

## 6. `cgcreate: Cgroup is not mounted` 的含义

如果响应仍包含：

```text
sudo cgcreate -g memory:...
cgcreate: libcgroup initialization failed: Cgroup is not mounted
```

说明请求到达的仍是官方未打补丁的 cgroup v1 runner，或旧容器尚未被替换；它并不表示应在
cgroup v2 宿主机额外挂载一个 v1 `memory` hierarchy。检查当前容器使用的镜像后，重新执行构建和
启动脚本：

```bash
docker inspect -f '{{.Image}} {{.Config.Image}}' iclr2027-sandboxfusion
SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
  bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh
SANDBOXFUSION_PIN_FILE="$SANDBOX_STATE/sandboxfusion-image.env" \
SANDBOX_PREFLIGHT_MARKER="$SANDBOX_STATE/sandboxfusion_preflight.json" \
  bash examples/optimizer_geometry/start_sandboxfusion.sh
```

只有启动脚本输出全部主动探针通过且 marker 中 `safe` 为 `true`，训练才会继续。
