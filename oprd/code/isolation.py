"""Linux submission restrictions; fail closed if seccomp is unavailable."""

import ctypes
import ctypes.util
import errno
import os
import resource

from oprd.code.common import MEMORY


def deny_network_and_process_creation():
    library = ctypes.util.find_library("seccomp")
    if not library:
        raise RuntimeError("libseccomp is required for network-off submissions")
    lib = ctypes.CDLL(library, use_errno=True)
    lib.seccomp_init.argtypes = [ctypes.c_uint32]
    lib.seccomp_init.restype = ctypes.c_void_p
    lib.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    lib.seccomp_syscall_resolve_name.restype = ctypes.c_int
    lib.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    lib.seccomp_load.argtypes = [ctypes.c_void_p]
    lib.seccomp_release.argtypes = [ctypes.c_void_p]
    context = lib.seccomp_init(0x7FFF0000)  # SCMP_ACT_ALLOW
    if not context:
        raise RuntimeError("seccomp_init failed")
    denied = []
    try:
        for name in (
            "socket",
            "socketpair",
            "connect",
            "bind",
            "listen",
            "accept",
            "accept4",
            "sendto",
            "sendmsg",
            "sendmmsg",
            "recvfrom",
            "recvmsg",
            "recvmmsg",
            "socketcall",
            "io_uring_setup",
            "io_uring_enter",
            "io_uring_register",
            "fork",
            "vfork",
            "clone",
            "clone3",
            "execve",
            "execveat",
            "ptrace",
            "process_vm_writev",
            "kill",
            "tgkill",
            "tkill",
            "mount",
            "umount2",
            "unshare",
            "setns",
            "bpf",
            "reboot",
        ):
            number = lib.seccomp_syscall_resolve_name(name.encode())
            if number >= 0:
                if lib.seccomp_rule_add(context, 0x00050000 | errno.EPERM, number, 0) != 0:
                    raise RuntimeError(f"Cannot restrict syscall {name}")
                denied.append(name)
        if lib.seccomp_load(context) != 0:
            raise RuntimeError("Cannot install seccomp restrictions")
    finally:
        lib.seccomp_release(context)
    return denied


def restrict_apps_submission(cpu):
    os.sched_setaffinity(0, {cpu})
    for name in ("AS", "DATA", "STACK"):
        resource.setrlimit(getattr(resource, "RLIMIT_" + name), (MEMORY, MEMORY))
    resource.setrlimit(resource.RLIMIT_CPU, (10, 11))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 1024**2, 16 * 1024**2))
    return deny_network_and_process_creation()
