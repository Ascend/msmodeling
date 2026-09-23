#!/bin/bash

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

# Check local vLLM/Proxy service health, print HTTP code (200=healthy).
# Usage: bash check_health.sh <port> [health_path] [bind_ip]
#   health_path defaults to /health (vLLM); proxy uses /healthcheck
#   bind_ip defaults to 127.0.0.1; 0.0.0.0 or empty → 127.0.0.1
port="${1:?missing port}"
path="${2:-/health}"
ip="${3:-127.0.0.1}"
[ "$ip" = "0.0.0.0" ] && ip="127.0.0.1"
code=$(curl -s -o /dev/null -w '%{http_code}' --noproxy '*' --max-time 5 "http://${ip}:${port}${path}" 2>/dev/null)
echo "${code:-000}"
