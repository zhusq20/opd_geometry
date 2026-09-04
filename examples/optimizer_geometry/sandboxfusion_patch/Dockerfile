FROM volcengine/sandbox-fusion@sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9 AS sandboxfusion-verified

# Refuse to apply the patch if the pinned image's source tree is not the exact
# revision that was audited when these replacements were written.
RUN --mount=type=bind,source=upstream-files.sha256,target=/tmp/sandboxfusion-upstream-files.sha256,ro \
    sha256sum -c /tmp/sandboxfusion-upstream-files.sha256

# Fail at build time if the fixed upstream image no longer supplies the exact
# runtime identity or namespace/mount tools used by the hardened lite chain.
RUN test "$(id -u app)" = "1000" \
    && for tool in chroot findmnt ip mknod mount umount unshare; do command -v "${tool}"; done

COPY sandbox/runners/cgroup_v2.py /root/sandbox/sandbox/runners/cgroup_v2.py
COPY sandbox/runners/file_security.py /root/sandbox/sandbox/runners/file_security.py
COPY sandbox/runners/isolation.py /root/sandbox/sandbox/runners/isolation.py
COPY sandbox/runners/base.py /root/sandbox/sandbox/runners/base.py
COPY sandbox/runners/types.py /root/sandbox/sandbox/runners/types.py
COPY sandbox/runners/jupyter.py /root/sandbox/sandbox/runners/jupyter.py
COPY sandbox/runners/__init__.py /root/sandbox/sandbox/runners/__init__.py
COPY sandbox/server/sandbox_api.py /root/sandbox/sandbox/server/sandbox_api.py
COPY sandbox/server/online_judge_api.py /root/sandbox/sandbox/server/online_judge_api.py
COPY scripts/cgroup2_exec.py /usr/local/libexec/sandboxfusion-cgroup2-exec
COPY scripts/prepare_cgroup2.py /usr/local/libexec/sandboxfusion-prepare-cgroup2
COPY scripts/restrict_exec.py /usr/local/libexec/sandboxfusion-restrict-exec

# The server runs with Miniconda Python; the shebang-driven helpers resolve
# /usr/bin/env python3 to the system Python and must remain compatible with it.
RUN chmod 0555 \
      /usr/local/libexec/sandboxfusion-cgroup2-exec \
      /usr/local/libexec/sandboxfusion-prepare-cgroup2 \
      /usr/local/libexec/sandboxfusion-restrict-exec \
    && /root/miniconda3/bin/python3 -m py_compile \
      /root/sandbox/sandbox/runners/cgroup_v2.py \
      /root/sandbox/sandbox/runners/file_security.py \
      /root/sandbox/sandbox/runners/isolation.py \
      /root/sandbox/sandbox/runners/base.py \
      /root/sandbox/sandbox/runners/types.py \
      /root/sandbox/sandbox/runners/jupyter.py \
      /root/sandbox/sandbox/runners/__init__.py \
      /root/sandbox/sandbox/server/sandbox_api.py \
      /root/sandbox/sandbox/server/online_judge_api.py \
    && /bin/python3 -m py_compile \
      /usr/local/libexec/sandboxfusion-cgroup2-exec \
      /usr/local/libexec/sandboxfusion-prepare-cgroup2 \
      /usr/local/libexec/sandboxfusion-restrict-exec \
    && rm -rf \
      /root/sandbox/sandbox/runners/__pycache__ \
      /root/sandbox/sandbox/server/__pycache__ \
      /usr/local/libexec/__pycache__

# The build script must complete the verifier target above before exporting
# this RUN-free target. Keeping validation RUN metadata out of the final stage
# lets BuildKit normalize the payload; the script separately checks any
# engine-specific atime/ctime differences.
FROM volcengine/sandbox-fusion@sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9 AS sandboxfusion-final

ARG PATCH_SHA256=unspecified
ARG SOURCE_DATE_EPOCH=0
LABEL org.opencontainers.image.title="SandboxFusion cgroup v2 patch" \
      org.opencontainers.image.description="Pinned SandboxFusion with fail-closed cgroup v2 and offline lite execution" \
      io.iclr2027.sandboxfusion.patch="cgroup2-v1" \
      io.iclr2027.sandboxfusion.patch-sha256="${PATCH_SHA256}" \
      io.iclr2027.sandboxfusion.base="volcengine/sandbox-fusion@sha256:dd7ff53d16132a8acad6d5da7f15154bb4a331381567a4cb21b3e97ce581f5f9"

COPY --chmod=0644 sandbox/runners/cgroup_v2.py /root/sandbox/sandbox/runners/cgroup_v2.py
COPY --chmod=0644 sandbox/runners/file_security.py /root/sandbox/sandbox/runners/file_security.py
COPY --chmod=0644 sandbox/runners/isolation.py /root/sandbox/sandbox/runners/isolation.py
COPY --chmod=0644 sandbox/runners/base.py /root/sandbox/sandbox/runners/base.py
COPY --chmod=0644 sandbox/runners/types.py /root/sandbox/sandbox/runners/types.py
COPY --chmod=0644 sandbox/runners/jupyter.py /root/sandbox/sandbox/runners/jupyter.py
COPY --chmod=0644 sandbox/runners/__init__.py /root/sandbox/sandbox/runners/__init__.py
COPY --chmod=0644 sandbox/server/sandbox_api.py /root/sandbox/sandbox/server/sandbox_api.py
COPY --chmod=0644 sandbox/server/online_judge_api.py /root/sandbox/sandbox/server/online_judge_api.py
COPY --chmod=0555 scripts/cgroup2_exec.py /usr/local/libexec/sandboxfusion-cgroup2-exec
COPY --chmod=0555 scripts/prepare_cgroup2.py /usr/local/libexec/sandboxfusion-prepare-cgroup2
COPY --chmod=0555 scripts/restrict_exec.py /usr/local/libexec/sandboxfusion-restrict-exec
