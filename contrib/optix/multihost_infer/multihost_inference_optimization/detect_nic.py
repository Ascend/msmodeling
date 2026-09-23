#!/usr/bin/env python3

# -------------------------------------------------------------------------
# This file is part of the MindStudio project.
# Copyright (c) 2026 Huawei Technologies Co.,Ltd.
#
# MindStudio is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
# -------------------------------------------------------------------------

# -*- coding: utf-8 -*-
"""detect_nic.py - Look up a NIC name (nic_name) from an IP address, without relying on the ip / ifconfig commands.

A node may not have iproute2 installed (no `ip` command), but it always has Python, so
all NIC name detection goes through this script. Two detection paths are tried in order:

1. pyroute2 (if installed on the node): goes through netlink and can see every address
   on a given NIC (including secondary / alias addresses);
2. Pure standard library: enumerate NICs with socket.if_nameindex() (falling back to
   /proc/net/dev), then read each NIC's IPv4 address with ioctl(SIOCGIFADDR) and
   compare them one by one. Needs nothing but Python itself, so it works on nodes with
   no third-party libraries installed.

This script is deliberately **self-contained** (it imports no other module from the
package), so it can be uploaded to a remote node and run on its own:

    python3 detect_nic.py 192.168.1.100

On success stdout contains only the NIC name (so callers can use it directly); on
failure it writes to stderr and exits non-zero.
"""

import sys

# <linux/sockios.h>: get a NIC's IPv4 address by interface name
_SIOCGIFADDR = 0x8915
# <linux/if.h> IFNAMSIZ = 16 (including the trailing NUL), so a NIC name is at most 15 chars
_IFNAMSIZ = 16
_MAX_IFNAME_LEN = _IFNAMSIZ - 1


def _detect_via_pyroute2(target_ip):
    """Query via netlink, covering multiple addresses configured on one NIC. Raises ImportError when pyroute2 is missing."""
    from pyroute2 import IPRoute

    with IPRoute() as ipr:
        for msg in ipr.addr("dump"):
            if msg.get("address") != target_ip:
                continue
            link_info = ipr.link("get", index=msg.get("index"))
            return link_info[0].get_attr("IFLA_IFNAME")
    return None


def _iter_ifnames():
    """Enumerate local NIC names: socket.if_nameindex() first, falling back to parsing /proc/net/dev."""
    import socket

    try:
        return [name for _, name in socket.if_nameindex()]
    except (AttributeError, OSError):
        pass

    names = []
    try:
        with open("/proc/net/dev", "r", encoding="utf-8") as f:
            for line in f:
                # The first two lines are headers and contain no ':'; data lines look
                # like "  eth0: 1234 5 ..."
                if ":" not in line:
                    continue
                names.append(line.split(":", 1)[0].strip())
    except OSError:
        pass
    return names


def _ipv4_of(ifname):
    """Get a NIC's IPv4 address; returns None when the NIC has no IPv4 address or the query fails."""
    import fcntl
    import socket
    import struct

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        packed = struct.pack("256s", ifname.encode("utf-8")[:_MAX_IFNAME_LEN])
        # In the returned sockaddr_in, offsets 20..24 hold the 4-byte IPv4 address
        res = fcntl.ioctl(sock.fileno(), _SIOCGIFADDR, packed)
        return socket.inet_ntoa(res[20:24])
    except OSError:
        return None
    finally:
        sock.close()


def _detect_via_ioctl(target_ip):
    """Pure standard library path: read each NIC's IPv4 address and compare it with the target IP."""
    for ifname in _iter_ifnames():
        if _ipv4_of(ifname) == target_ip:
            return ifname
    return None


def get_ifname_by_ip(target_ip):
    """Return the name of the NIC bound to target_ip, or None when not found.

    Tries the pyroute2 path and then the pure standard library path: when either fails
    because of a missing dependency or insufficient permissions, the reason is logged to
    stderr and the next path is tried; None is returned only when all of them fail.
    """
    if not target_ip:
        return None
    for probe in (_detect_via_pyroute2, _detect_via_ioctl):
        try:
            ifname = probe(target_ip)
        except Exception as e:  # pyroute2 missing / no /proc / insufficient permissions, etc.
            print(f"[WARN] {probe.__name__} failed: {e}", file=sys.stderr)
            continue
        if ifname:
            return ifname
    return None


def main(argv):
    if len(argv) != 2:
        print(f"Usage: {argv[0] if argv else 'detect_nic.py'} <ip>", file=sys.stderr)
        return 2
    target_ip = argv[1].strip()
    ifname = get_ifname_by_ip(target_ip)
    if not ifname:
        print(f"[ERROR] no NIC found bound to IP {target_ip}", file=sys.stderr)
        return 1
    # stdout carries only the NIC name, so callers (including remote SSH invocations)
    # can use it directly
    print(ifname)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
