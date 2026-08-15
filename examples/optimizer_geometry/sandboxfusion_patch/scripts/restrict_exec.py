#!/usr/bin/env python3
"""Drop execution privileges, install seccomp, then exec untrusted code."""

from __future__ import annotations

import ctypes
import errno
import os
import platform
import sys
from pathlib import Path

# Classic BPF and seccomp constants from linux/filter.h and linux/seccomp.h.
BPF_LD_W_ABS = 0x20
BPF_JMP_JEQ_K = 0x15
BPF_JMP_JSET_K = 0x45
BPF_RET_K = 0x06
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
PR_SET_KEEPCAPS = 8
PR_CAPBSET_DROP = 24
PR_CAP_AMBIENT = 47
PR_CAP_AMBIENT_CLEAR_ALL = 4
SECCOMP_MODE_FILTER = 2
AUDIT_ARCH_X86_64 = 0xC000003E
LINUX_CAPABILITY_VERSION_3 = 0x20080522

# x86_64 syscall numbers. The build/start scripts reject other architectures.
DENIED_SYSCALLS = {
    101,  # ptrace
    155,  # pivot_root
    161,  # chroot
    165,  # mount
    166,  # umount2
    167,  # swapon
    168,  # swapoff
    169,  # reboot
    175,  # init_module
    176,  # delete_module
    179,  # quotactl
    212,  # lookup_dcookie
    246,  # kexec_load
    248,  # add_key
    249,  # request_key
    250,  # keyctl
    272,  # unshare
    298,  # perf_event_open
    300,  # fanotify_init
    304,  # open_by_handle_at
    308,  # setns
    310,  # process_vm_readv
    311,  # process_vm_writev
    312,  # kcmp
    313,  # finit_module
    320,  # kexec_file_load
    321,  # bpf
    323,  # userfaultfd
    425,  # io_uring_setup
    426,  # io_uring_enter
    427,  # io_uring_register
    428,  # open_tree
    429,  # move_mount
    430,  # fsopen
    431,  # fsconfig
    432,  # fsmount
    433,  # fspick
    442,  # mount_setattr
}
SYS_CLONE = 56
SYS_CLONE3 = 435
CLONE_NAMESPACE_FLAGS = (
    0x00000080  # CLONE_NEWTIME
    | 0x00020000  # CLONE_NEWNS
    | 0x02000000  # CLONE_NEWCGROUP
    | 0x04000000  # CLONE_NEWUTS
    | 0x08000000  # CLONE_NEWIPC
    | 0x10000000  # CLONE_NEWUSER
    | 0x20000000  # CLONE_NEWPID
    | 0x40000000  # CLONE_NEWNET
)


class SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint32),
    ]


class SockFprog(ctypes.Structure):
    _fields_ = [("length", ctypes.c_ushort), ("filter", ctypes.POINTER(SockFilter))]


class CapHeader(ctypes.Structure):
    _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]


class CapData(ctypes.Structure):
    _fields_ = [
        ("effective", ctypes.c_uint32),
        ("permitted", ctypes.c_uint32),
        ("inheritable", ctypes.c_uint32),
    ]


def statement(code: int, value: int) -> SockFilter:
    return SockFilter(code=code, jt=0, jf=0, k=value)


def jump(code: int, value: int, true_skip: int, false_skip: int) -> SockFilter:
    return SockFilter(code=code, jt=true_skip, jf=false_skip, k=value)


def checked_prctl(option: int, arg2: int = 0, arg3: int = 0, arg4: int = 0, arg5: int = 0) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(option, arg2, arg3, arg4, arg5) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def clear_capability_sets() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    header = CapHeader(version=LINUX_CAPABILITY_VERSION_3, pid=0)
    data = (CapData * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def drop_privileges(uid: int, gid: int) -> None:
    if uid <= 0 or gid <= 0:
        raise ValueError("sandbox uid and gid must be positive")
    if os.geteuid() != 0:
        raise PermissionError("privilege drop must begin as container root")

    # Prevent every future exec from granting privilege before changing any
    # identities. All subsequent operations only remove privilege.
    checked_prctl(PR_SET_NO_NEW_PRIVS, 1)
    os.setgroups([])

    cap_last = int(Path("/proc/sys/kernel/cap_last_cap").read_text(encoding="utf-8").strip())
    if cap_last < 0 or cap_last > 63:
        raise RuntimeError(f"unexpected kernel cap_last_cap: {cap_last}")
    for capability in range(cap_last + 1):
        checked_prctl(PR_CAPBSET_DROP, capability)

    checked_prctl(PR_SET_KEEPCAPS, 0)
    os.setresgid(gid, gid, gid)
    os.setresuid(uid, uid, uid)
    checked_prctl(PR_CAP_AMBIENT, PR_CAP_AMBIENT_CLEAR_ALL)
    clear_capability_sets()

    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
        if ":" in line
    )
    for field in ("CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb"):
        if int(status.get(field, "1").strip(), 16) != 0:
            raise RuntimeError(f"failed to clear {field}")
    if os.getresuid() != (uid, uid, uid) or os.getresgid() != (gid, gid, gid) or os.getgroups():
        raise RuntimeError("failed to establish the unprivileged execution identity")


def build_filter() -> list[SockFilter]:
    deny_eperm = SECCOMP_RET_ERRNO | errno.EPERM
    deny_enosys = SECCOMP_RET_ERRNO | errno.ENOSYS
    instructions = [
        statement(BPF_LD_W_ABS, 4),
        jump(BPF_JMP_JEQ_K, AUDIT_ARCH_X86_64, 1, 0),
        statement(BPF_RET_K, SECCOMP_RET_KILL_PROCESS),
        statement(BPF_LD_W_ABS, 0),
        # clone3 passes a pointer to its flags, which classic seccomp BPF cannot
        # inspect. ENOSYS makes glibc fall back to ordinary clone for threads.
        jump(BPF_JMP_JEQ_K, SYS_CLONE3, 0, 1),
        statement(BPF_RET_K, deny_enosys),
        # Ordinary fork/thread clone stays available; namespace creation does not.
        jump(BPF_JMP_JEQ_K, SYS_CLONE, 0, 3),
        statement(BPF_LD_W_ABS, 16),
        jump(BPF_JMP_JSET_K, CLONE_NAMESPACE_FLAGS, 0, 1),
        statement(BPF_RET_K, deny_eperm),
        statement(BPF_LD_W_ABS, 0),
    ]
    for syscall_number in sorted(DENIED_SYSCALLS):
        instructions.extend(
            [
                jump(BPF_JMP_JEQ_K, syscall_number, 0, 1),
                statement(BPF_RET_K, deny_eperm),
            ]
        )
    instructions.append(statement(BPF_RET_K, SECCOMP_RET_ALLOW))
    return instructions


def install_filter() -> None:
    if platform.machine() not in {"x86_64", "amd64"}:
        raise RuntimeError("the audited seccomp filter supports x86_64 only")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    instructions = build_filter()
    array = (SockFilter * len(instructions))(*instructions)
    program = SockFprog(length=len(instructions), filter=array)
    if libc.prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.byref(program), 0, 0) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: sandboxfusion-restrict-exec [--drop-to UID GID --] COMMAND [ARG ...]")
    command = sys.argv[1:]
    if command[0] == "--drop-to":
        if len(command) < 5 or command[3] != "--":
            raise SystemExit("usage: sandboxfusion-restrict-exec --drop-to UID GID -- COMMAND [ARG ...]")
        try:
            uid = int(command[1])
            gid = int(command[2])
        except ValueError as exc:
            raise SystemExit("sandbox uid and gid must be integers") from exc
        drop_privileges(uid, gid)
        command = command[4:]
    install_filter()
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
