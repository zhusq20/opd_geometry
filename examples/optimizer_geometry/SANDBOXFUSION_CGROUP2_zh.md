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

```bash
cd /workspace/dev/iclr2027/slime_opd_geometry

bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh
```

脚本先针对固定基础镜像验证补丁，再执行两次禁用缓存的最终镜像构建。若 Docker
写入了不可复现的 atime/ctime 或目录 mtime 元数据，脚本会改为要求规范化后的运行时指纹
完全一致（镜像配置，以及所有补丁文件的内容、权限、属主、大小和 mtime；目录的权限与
属主），然后写入：

```text
data/m2rl/sandbox/sandboxfusion-image.env
```

其中 `SANDBOXFUSION_IMAGE=sha256:...` 是本机 Docker content-addressed image ID，
Compose 会按这个不可变 ID 启动，而不是按 mutable tag 启动。基础镜像很大，但第二次构建会
复用已经下载的官方 layers。

若需跨机器部署，可显式提供一个有写权限的临时 registry tag：

```bash
SANDBOXFUSION_PUSH_REF=registry.example.org/team/sandboxfusion:cgroup2-v1 \
  bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh
```

脚本推送后会把 pin 改写为 `repository@sha256:...`，启动时仍只接受 digest。

## 3. 启动并运行主动探针

```bash
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
data/m2rl/sandbox/sandboxfusion_preflight.json
```

启动前旧 marker 会先被原子改写为 `safe=false`；任何一步失败都会关闭本次服务并保持 marker
无效。服务不会在崩溃或宿主重启后自动拉起；必须重新运行启动脚本和全部探针。不能改用
`SANDBOX_CONFIG=local`，也不能手工修改 marker。

## 4. 从训练容器访问

训练容器需共享宿主机网络，保持 SandboxFusion endpoint 仅发布到宿主 loopback。原来的容器没有
`--network host`，运行中的容器不能切换到 host network，需退出后重建：

```bash
cd /workspace/dev/iclr2027
mkdir -p .training-cache/root-cache

docker run --rm \
  --name iclr2027-training \
  --gpus all \
  --network host \
  --ipc=host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PWD":/workspace/dev \
  -v "$PWD/.training-cache/root-cache":/root/.cache \
  -w /workspace/dev/slime_opd_geometry \
  -it slimerl/slime:latest /bin/bash
```

进入训练容器后：

```bash
export SANDBOXFUSION_BASE_URL=http://127.0.0.1:8080
export M2RL_SANDBOX_PREFLIGHT_MARKER=\
/workspace/dev/slime_opd_geometry/data/m2rl/sandbox/sandboxfusion_preflight.json

python3 -c 'import urllib.request; print(urllib.request.urlopen("http://127.0.0.1:8080/v1/ping").read().decode())'
```

预期返回 `"pong"`。训练容器只作为 HTTP 客户端，不需要 Docker daemon/socket。

`--rm` 会在退出时删除训练容器的 writable layer。`/workspace/dev` 中的文件由宿主 bind mount
持久化，不会丢；原来下载在 `/root/.cache`、安装在容器系统目录中的内容则会丢。若当前容器
已经下载了模型，退出前可先在当前容器内执行：

```bash
mkdir -p /workspace/dev/.training-cache/root-cache
cp -a /root/.cache/. /workspace/dev/.training-cache/root-cache/
```

上面的新启动命令会把该目录重新挂到 `/root/.cache`。通过 `apt`/`pip` 修改的系统环境应写入
训练镜像的 Dockerfile 后重建；仅 `docker run` 一个新容器并不会修改原镜像。

## 5. 修改补丁后的规则

任何 `sandboxfusion_patch/` 文件变化都会改变 patch SHA-256；已有 pin 将被启动器拒绝，必须重新
运行构建脚本。更新官方基础镜像时，也必须重新审计 runner 并更新
`upstream-files.sha256`，不能只替换 `FROM` digest 绕过源文件哈希门禁。

专用 cgroup 的聚合默认上限为 32 GiB 和 4096 PID，服务控制面本身为 4 GiB/2048 PID。
宿主资源较小时可在构建后、启动前调整前两项（值必须是正整数，单位为 byte/PID）：

```bash
export SANDBOXFUSION_AGGREGATE_MEMORY_BYTES=$((16 * 1024 * 1024 * 1024))
export SANDBOXFUSION_AGGREGATE_PIDS=2048
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
bash examples/optimizer_geometry/build_sandboxfusion_cgroup2.sh
bash examples/optimizer_geometry/start_sandboxfusion.sh
```

只有启动脚本输出全部主动探针通过且 marker 中 `safe` 为 `true`，训练才会继续。
