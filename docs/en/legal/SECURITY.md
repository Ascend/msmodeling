# **msModeling Security Statement**

## System Security Hardening

- msModeling is a development, simulation, and performance evaluation tool, and must not be used as an online production service.
- To ensure security, you are advised to use the msModeling tool as a complete package. If you need to perform secondary development, you are responsible for monitoring and handling any security risks that may be introduced.
- msModeling runs from the CLI or the local Web UI by default. The Web UI binds to `127.0.0.1` by default. You are advised to start and use it locally instead of exposing it to the public network or untrusted networks.
- If you need to open an access entry through `0.0.0.0`, a remote address, Gradio share, or a reverse proxy, note the security risks introduced by remote communication, and configure access control, authentication, network isolation, and transport encryption yourself.
- This tool is a development-stage tool. It does not restrict the owner and permissions of the installation or of the file objects it processes. You need to assign appropriate owners and permissions based on the usage scenario and ensure that the file content processed by the tool is secure and trustworthy.
- You are advised to enable ASLR (level 2) on the system, also known as **full random address space layout randomization**. You can configure it as follows:

```sh
echo 2 > /proc/sys/kernel/randomize_va_space
```

## Recommended Runtime User

For security and to minimize permissions, msModeling should not be installed or run with a high-privilege account such as root. You are advised to run it with an ordinary user's permissions. If the tool depends on CANN (for example, NPU replay and HCCL communication benchmark in the performance database collection tool), use the CANN package installed by default for the same low-privilege user to ensure security. After running the `source` command, do not modify the environment variables involved in `set_env.sh` at will.

## File Permission Control

In a Linux or Unix environment, check the umask setting of the current user before installing or using msModeling. The recommended setting is `0027` or stricter, which ensures that installed files are not writable by other users or users in the same group, avoiding potential security risks.

When you provide input files to the tool as command input, you are advised to ensure that the owner of the provided files is the same as the owner of the tool process, and that the file permissions do not allow others (including the group and others) to modify them. Files written to the drive by the tool are not writable by others by default. You can control the permissions of the generated files as needed.

**File Permission Reference**

| Type | Maximum Linux Permissions (Reference) |
| --- | --- |
| User home directory | 750 (rwxr-x---) |
| Program files (including script files, library files, and so on) | 550 (r-xr-x---) |
| Program file directory | 550 (r-xr-x---) |
| Configuration file | 640 (rw-r-----) |
| Configuration file directory | 750 (rwxr-x---) |
| Log files (after recording is complete or when archived) | 440 (r--r-----) |
| Log files (being recorded) | 640 (rw-r-----) |
| Log file directory | 750 (rwxr-x---) |
| Debug file | 640 (rw-r-----) |
| Debug file directory | 750 (rwxr-x---) |
| Temporary file directory | 750 (rwxr-x---) |
| Maintenance and upgrade file directory | 770 (rwxrwx---) |
| Service data file | 640 (rw-r-----) |
| Service data file directory | 750 (rwxr-x---) |
| Directory of key components, private keys, certificates, and ciphertext files | 700 (rwx------) |
| Key components, private keys, certificates, and encrypted ciphertext | 600 (rw-------) |
| Encryption and decryption interfaces and scripts | 500 (r-x------) |

## Vulnerability Security Statement

- Huawei's rules on product vulnerability management follow the Vulnerability Handling Process. For details, see [Vulnerability Handling Process](https://www.huawei.com/cn/psirt/vul-response-process).
- If enterprise customers need to obtain vulnerability information, see [Security Notices](https://securitybulletin.huawei.com/enterprise/cn/security-advisory).
- When installing dependencies, use newer software packages that meet the requirements, and pay attention to and fix existing vulnerabilities, especially publicly disclosed high-risk vulnerabilities with a CVSS score greater than 7.

## Data Security Statement

During use, msModeling loads and saves data such as model configurations, simulation results, performance data, and log files. Before importing model configurations, YAML configurations, performance traces, or custom scripts, you need to ensure that the data sources are trustworthy and the environment is secure.

When you use a Hugging Face or ModelScope model ID to pull configurations, `trust_remote_code=True` in `transformers` may be triggered, causing remote Python code to be executed. msModeling does not provide security guarantees for remote model code. You are advised to use a reviewed local absolute path (secure local mode). If the model, configuration, or simulation data must be kept confidential, delete the related deliverables promptly after use to prevent unnecessary information leakage.

## Build Security Statement

This project installs Python dependencies through `uv` or `pip`. During the build or installation process, source code, dependency packages, or model configurations may be downloaded from sites such as GitCode, PyPI, official PyTorch sources, Hugging Face, and ModelScope. You can configure trusted mirror sources or intranet sources as needed, and control the permissions of the source code directory and build results yourself to reduce security risks.

## Runtime Security Statement

When an exception occurs during running, msModeling exits the process and prints error information. You are advised to locate the specific cause of the error based on the error messages, which may cover aspects such as file permissions, model parsing, configuration reading, data persistence, performance collection, and the dependency environment.

When the tool loads model configurations or large-scale simulation data, if the memory usage exceeds the system capacity limit, an error may occur and cause the process to exit unexpectedly. If the generated results exceed the free space remaining on the drive, the process may exit abnormally.

If the runtime environment depends on CANN or NPU-related dynamic libraries, you need to ensure that the content of environment variables such as `LD_LIBRARY_PATH` is secure and trustworthy before use, that the paths they point to do not involve symbolic links, and that the permissions and owners meet security expectations and cannot be tampered with by third parties. Otherwise, there is a risk of arbitrary code injection.

## Public Interface Statement

The msModeling project is developed in Python and provides a CLI entry point and Python APIs externally. All public interfaces are described in the user guide and quick start documentation. Scripting languages such as Python are released in source code form. You are advised to use the public interfaces described in the documentation directly and avoid calling internal modules or interfaces that are not explicitly public.

## Communication Security Hardening

- The msModeling Web UI binds to `127.0.0.1` by default. When the browser and the service run on the same machine, communication occurs within localhost.
- If you bind the Web UI to a non-local address or access it remotely through a proxy, port forwarding, shared links, or other methods, this is a non-default secure usage mode. You are responsible for monitoring the security risks introduced by remote communication.
- To mitigate security risks, you are advised to harden communication security with firewalls, iptables, access control lists, reverse proxy tools such as VPN and Nginx, and HTTPS, and limit the range of clients that can access the service.
- When collecting performance data of communication operators, the performance database collection tool uses `torchrun` to establish TCP communication. Use it only in a trusted network environment, and perform security hardening before using scripts for communication data collection. For details, see [Communication security hardening of the Ascend PyTorch repository](https://gitcode.com/Ascend/pytorch/blob/v2.7.1-26.1.0/docs/en/SECURITYNOTE.md#%E9%80%9A%E4%BF%A1%E5%AE%89%E5%85%A8%E5%8A%A0%E5%9B%BA). In addition, ensure that environment variables such as `MASTER_ADDR` and `MASTER_PORT` are configured as expected.

## Communication Matrix

**Communication Matrix Information**

| No. | Source Device | Source IP Address | Source Port | Destination Device | Destination IP Address | Destination Port (Listening) | Protocol | Port Description | Listening Port Configurable | Authentication Method | Encryption Method | Plane | Version | Special Scenario | Remarks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | PC where msModeling is installed | 127.0.0.1 | Random port assigned by the system | Local host | 127.0.0.1 | 2345 by default | HTTP | The msModeling Web UI binds to the local loopback address by default and is used for visual configuration and result viewing. | Changeable. Can be configured with `--port` or the `GRADIO_SERVER_PORT` environment variable. | Not applicable | Not applicable | Management plane | master | Local Web UI access | Not exposed to the public network by default |
| 2 | PC where msModeling is installed | IP address of the user's local client | Random port assigned by the system | Public network servers such as GitCode, PyPI, PyTorch, Hugging Face, and ModelScope | IP addresses of the corresponding public network servers | 443 | HTTPS | Public network addresses may be accessed when installing dependencies, cloning source code, downloading model configurations, or browsing online materials. | Not applicable | Not applicable | SSL/TLS | Management plane | master | Installation, build, model configuration retrieval, and online documentation browsing | Mirror sources or intranet sources can be configured based on security policies |
| 3 | PC where msModeling is installed | 127.0.0.1 or `MASTER_ADDR` specified by the user | Random port assigned by the system | Local host or multi-node communication peer | 127.0.0.1 or `MASTER_ADDR` specified by the user | Starts at 29500 or 29700 by default, range 1024-65535. | TCP | Used when the performance database collection tool runs HCCL communication benchmarks or distributed operator replay through `torchrun`. | Changeable. Can be configured with `MASTER_ADDR` and `MASTER_PORT`. | Not applicable | Not applicable | Service plane | master | Performance database collection scenario | An NPU/CANN environment is required. For multi-node scenarios, ensure the network is trustworthy |
| 4 | vLLM client | IP address of the vLLM client for communication | Random port assigned by the system | vLLM server started by the auto-optimization tool | IP address bound to the vLLM server | Fixed port configured based on the actual requirements of the live-network customer, corresponding to the `--port` field used when starting the service, 8000 by default | HTTP | The tool starts the inference service in the customer environment through the CLI. If the customer has not performed any configuration, the default port 8000 is used. Otherwise, the vLLM service is started on the port specified in the customer configuration. | Changeable. Can be configured with the `--port` field when starting the service. | Not applicable | Not applicable | Data plane | All versions | None | None |
