# Use Case: From Environment Preparation to Agent-Driven PD Disaggregation Optimization 

This document is intended for first-time OptiX users. It walks you through model preparation, image and container deployment, tool installation, agent-driven optimization, and result viewing in sequence.

**Scope**: Linux, Ascend NPU, vLLM Ascend, and OptiX agent mode. Obtain the tool from the [msmodeling official repository](https://gitcode.com/Ascend/msmodeling/). For the installation steps, see section 4.

**Before you start, ensure that the following conditions are met:**

- **Agent tool**: You can use msagent or OpenCode, which can read files, run commands, and continuously monitor tasks. For installation and startup, see section 4.5.
- **Model service and quota**: You have configured a usable model service account or API authentication and have sufficient quota. Analyzing logs, generating candidates, and multi-round optimization continuously consume quota. The actual cost and limits depend on the service that you use.
- **Network access**: The environment that runs the agent can access the selected model service and the optimization devices. In an intranet environment, you can use a reachable intranet model service, and a public network connection is not mandatory. The model, image, tool dependencies, and knowledge base must be downloadable or prepared offline in advance. The service selection and the transmission of configurations and logs must comply with the requirements of your organization.

If a usable agent tool, model service, or quota is not yet available, or if network restrictions prevent these conditions from being met, use the [PD disaggregation optimization guide for the Particle Swarm Optimization (PSO) algorithm](optix_pso_pd_disaggregation.md) instead. That process does not depend on an agent tool or on LLM API quota, but you still need to prepare the corresponding deployment and benchmark environment.

The main workflow is: **prepare the installation → start the agent → confirm the task → view the results**.

This guide applies to a trusted intranet test environment. Your administrator configures the accounts, keys, and network access permissions. The agent uses a model service approved by your organization, and whether configurations and logs leave your network depends on the service that you select.

Sections 1 through 4 cover the one-time environment preparation and the agent startup. Section 5 covers sending the task. In section 6, you confirm the task and then wait for the optimization. Section 7 covers viewing the results. The agent performs the optimization. Therefore, you do not need to start services, write configurations, or run benchmarks manually.

## 1. Preparing the Information Required for Optimization

This document uses two machines as an example: node A runs the agent, and both machines participate in the PD disaggregation optimization. All the following values are examples and **can be customized for your service**. If you change them, also replace the corresponding values in the subsequent commands and in the task description.

| Information to Provide | Example in This Document | How to Customize |
|---|---|---|
| Available machines and login method | A: `192.0.2.10`, B: `192.0.2.11`, SSH user `root`, port 22 | Replace them with the actual IP address, user, and port. |
| Devices available on each machine | Devices 0,1,2,3 are available on both A and B, 8 devices in total | List the specific device IDs that this task is allowed to use. Do not specify only the total number of devices. |
| Model | `Qwen/Qwen2.5-7B-Instruct` | You can replace it with a model that the deployment backend supports. |
| Workload | Input 1024 tokens and output 256 tokens, 200 requests, concurrency 16 | Fill in values according to the actual request scale. |
| Performance target | Improve throughput after the latency requirements are met | You can customize targets such as TTFT and TPOT. If you are not familiar with them, ask the agent to explain and confirm them first. |
| Time or run budget | At most 24 runs and 4 hours in total, including the retest | You can adjust it according to device idle time. The agent allocates the budget for each stage. |

**You do not need to determine the P/D ratio or the tuning parameters in advance.** The agent proposes candidates based on the model, the available devices, and the measurements. If you already have parameter experience or a fixed configuration, you can also provide it as an additional requirement.

The agent checks environment information such as the network. If it cannot determine the information about multiple NICs or cross-machine communication, it asks you or your administrator to confirm the actual NICs and communication IP addresses at one time before it starts the benchmark.

## 2. Preparing the Model

Download the model from [Hugging Face](https://huggingface.co/models) or [ModelScope](https://modelscope.cn/models), or reuse existing model files. This document uses `Qwen/Qwen2.5-7B-Instruct` as an example and places the complete model directory at `/data/models/Qwen2.5-7B-Instruct` on both host machines, keeping the model version consistent.

You can customize the model and the save path. If you change them, also replace the container mount path and the model path in the agent task description accordingly.

## 3. Downloading the Image and Starting the Containers: Running on Both Host Machines Separately

### 3.1 Checking the Host Machine

```bash
npu-smi info
docker version
```

Confirm that the devices and Docker work properly. Your administrator prepares the basic device environment in advance, and the agent can later read the device and network information.

### 3.2 Downloading the Matching Image

Select an image that matches the hardware, the driver, and the PD Connector from the [vLLM Ascend installation instructions](https://docs.vllm.ai/projects/ascend/en/main/getting_started/installation.html). Determine the complete image address or digest that matches your deployment environment, and then run the following on both host machines:

```bash
# Replace with the actual image tag that matches your deployment environment. Do not use latest
export IMAGE='quay.io/ascend/vllm-ascend:<verified full tag>'
docker pull "$IMAGE"
docker image inspect "$IMAGE" --format '{{json .RepoDigests}}'
```

Use the same image version on both machines, and confirm that the image supports PD disaggregation deployment.

### 3.3 Creating the Containers

The tool logs in to the host machines over SSH and operates the containers. The following example maps four devices on each machine. If the device IDs are different, modify the corresponding `--device` entries.

First create `/data/msmodeling` on both host machines. This is the location where the msmodeling tool is downloaded later, and its path in the container after mounting is `/workspace/msmodeling`. This directory can be empty for now on node B. You can customize the directory, but you must modify the mount command and the subsequent paths accordingly.

You can also customize the container name `optix-pd`. If you change it, also replace the container name in all `docker exec` commands and in the information that you provide to the agent. The host machine directory `/data/models` that this document mounts corresponds to `/models` in the container.

```bash
mkdir -p /data/msmodeling

docker run -d --name optix-pd \
  --network host --ipc host --shm-size 16g \
  --device /dev/davinci0 --device /dev/davinci1 \
  --device /dev/davinci2 --device /dev/davinci3 \
  --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -v /data/models:/models:ro \
  -v /data/msmodeling:/workspace/msmodeling \
  --entrypoint /bin/bash "$IMAGE" -lc 'sleep infinity'
```

For an RDMA deployment, you must also map devices such as `/dev/infiniband`, set the corresponding permissions, and configure the network parameters according to the selected image and network scheme, which your device administrator must confirm. The preceding basic container command does not replace the RDMA/HCCL configuration. With host networking, all services on the same host machine share the port space. Therefore, you must avoid the ports of existing services.

Check both containers:

```bash
docker exec optix-pd npu-smi info
docker exec optix-pd test -f /models/Qwen2.5-7B-Instruct/config.json
docker exec optix-pd bash -lc 'command -v vllm; vllm --help >/dev/null'
docker exec optix-pd python3 -c 'import vllm, vllm_ascend; print("vLLM imports OK")'
```

Pass criteria: The containers keep running, four devices are visible, the model is visible, and the vLLM command and the Ascend plugin are available. For the image mount requirements, see the [official quick start](https://docs.vllm.ai/projects/ascend/en/main/getting_started/quick_start.html), and the instructions for the corresponding version take precedence.

## 4. Preparing the Control Node: Running Only in the Container on Node A

Run all the following commands after you enter the container on node A by running `docker exec -it optix-pd bash`.

### 4.1 Preparing the Basic Commands and Passwordless SSH

The container requires `git`, `ssh`, `scp`, `curl`, `bash`, Python 3.10 or later, and venv support. If any of them are missing, install them according to the operating system of the image. For example, on Ubuntu, run `apt-get install git openssh-client curl python3-venv`.

```bash
# If a standard key or an SSH agent is already available, you do not need to generate a new key
ssh-keygen -t ed25519
ssh-copy-id -p 22 root@192.0.2.10
ssh-copy-id -p 22 root@192.0.2.11

ssh -o BatchMode=yes -p 22 root@192.0.2.10 \
  'docker exec optix-pd bash -lc "echo A_OK; command -v vllm"'
ssh -o BatchMode=yes -p 22 root@192.0.2.11 \
  'docker exec optix-pd bash -lc "echo B_OK; command -v vllm"'
```

Confirm the host fingerprint that your administrator verified when you connect for the first time. Use a key dedicated to this task, or an SSH agent loaded with that key, and do not reuse a personal or team-shared key. You are advised to use a dedicated operations account. The `root` account in the example is for the test instructions only. Therefore, if you change the account, also replace the account in the commands and in the task description. Docker permissions also grant high host machine operation capabilities. Therefore, your administrator must restrict the machines and sources that can access them, revoke the authorization after the task ends, and delete the dedicated key.

The tool does not read the alias, `IdentityFile`, or `ProxyJump` from the SSH configuration. Therefore, enter the actual address and use the default key path or an SSH agent. If a jump host is required, ask your administrator to provide a forwarding address that you can connect to directly.

Pass criteria: Both of the preceding checks complete without a password prompt and find vLLM in the container.

### 4.2 Downloading the msmodeling Tool

msmodeling is the tool repository that contains OptiX and the optimization Skill. The following downloads the master branch of the main repository to the mount directory that you created in the previous section:

```bash
git clone --branch master https://gitcode.com/Ascend/msmodeling.git /tmp/msmodeling-source
cp -a /tmp/msmodeling-source/. /workspace/msmodeling/
cd /workspace/msmodeling
```

The target directory must be empty the first time you run this command. If the tool already exists, do not copy it again. Run `git pull --ff-only origin master` in that directory to get the latest code. If there are local modifications, keep them and handle them first. After the update, run the installation commands in section 4.3 again.

### 4.3 Installing OptiX and the PD Plugin

If you already have `uv`, skip the first two lines. The following places uv itself in a separate small environment:

```bash
python3 -m venv /opt/uv-tools
/opt/uv-tools/bin/pip install uv
export PATH="/opt/uv-tools/bin:$PATH"

cd /workspace/msmodeling
uv sync && uv pip install --python .venv/bin/python -e contrib/optix/vllm_pd_simulator

uv run --no-sync msmodeling optix --help
.venv/bin/python -c 'from importlib.metadata import version; print(version("vllm_pd_simulator"))'
```

This document installs OptiX and AISBench in separate environments to avoid affecting the vLLM that already exists in the image. After the installation is complete, hand the environment over to the agent to run. After you update the code or run `uv sync` again, you must install the PD plugin again, because the sync operation removes packages that are not in the dependency list.

### 4.4 Downloading and Installing AISBench (Recommended)

AISBench is the benchmark tool that sends requests and collects throughput and latency statistics. Install it in a **separate environment in the container on node A**, not in the OptiX environment or in the vLLM environment of the image. For the installation method, see the [official AISBench repository](https://github.com/AISBench/benchmark).

```bash
uv venv --python 3.10 /opt/aisbench-venv
uv pip install --python /opt/aisbench-venv/bin/python 'setuptools<81' \
  ais_bench_benchmark==3.1.20260630
/opt/aisbench-venv/bin/ais_bench --help

# Let OptiX find the AISBench in the separate environment. It must remain in effect in the current terminal and in the terminal that starts the Agent
export OPTIX_DEPLOY_PATH=/opt/aisbench-venv
```

Pinning the version makes the result reproducible. You can customize the installation path, and if you change it, adjust the commands and environment variables accordingly. The agent prepares the benchmark configuration. Therefore, you do not need to write it manually.

### 4.5 Installing and Starting the Agent (Choose Either One)

This document consistently installs and runs the agent **in the container on node A**, sharing the same directory and terminal environment as OptiX. You do not need to install the agent on node B. This way, the agent can directly read the configuration, run the tool, and manage both machines. Do not start the agent on the host machine and then carry over the container paths.

You can use the msagent provided by your organization, or install and configure a reachable model service according to the [OpenCode official documentation](https://dev.opencode.ai/v2/docs). The agent must be able to read files, run commands, and continuously monitor tasks. This document does not guarantee compatibility with arbitrary agents. If a usable tool already exists, skip the installation.

Open a terminal in the container on node A, and start the agent after you enter the tool environment:

```bash
cd /workspace/msmodeling
source .venv/bin/activate
export OPTIX_DEPLOY_PATH=/opt/aisbench-venv
opencode
```

If you use msagent, replace the last line with the startup command that your organization provides. Then send the task description in the next section. If the tool does not load the Skill automatically, read `SKILL.md` at the path in the task. If the knowledge base is missing, the agent initializes it according to the Skill, and it must be able to access the corresponding knowledge repository. In an offline environment, ask your administrator to prepare it in advance, and you do not need to initialize the submodules yourself.

## 5. Telling the Agent What to Do

After you start the agent, copy the task description below and modify the machines, model, load, target, and budget according to your actual situation. You can customize all the example values. If you already have a fixed ratio or specific parameter requirements, add them at the end of the task description.

```text
Use .agents/skills/optix-assistant in this repository to run Agent mode optimization for PD disaggregation, and use AISBench for the benchmark.

Machines: A is 192.0.2.10 and B is 192.0.2.11, SSH user root, port 22.
The container name is optix-pd on both host machines, and devices 0, 1, 2, and 3 are available on each machine.
Model: /models/Qwen2.5-7B-Instruct in both containers, served model name pd-demo.
Load: input 1024 tokens and output 256 tokens, 200 requests, fixed concurrency 16.
Target: improve throughput while TTFT≤2000ms and TPOT≤100ms.
Budget: at most 24 measurement runs and 4 hours in total, including the baseline and the final retest.

Check the environment, prepare the configuration, and arrange the optimization yourself. Show the configuration and the plan for my confirmation first.
Report the progress during execution, and provide the optimal ratio, the parameters, the comparison with the baseline, and the deployment reproduction files when you finish.
```

You can customize the served model name, and the agent ensures that the service side and the benchmark use the same name. If you are not familiar with the latency target, you can change the target line to “Measure the baseline first, and confirm the target with me after you explain the results.”

The agent completes the remaining preparation and execution, and it asks for all the missing information at once if the information is insufficient. Select interactive mode as prompted to confirm the plan round by round. If you want the agent to proceed automatically within the budget, add “use fully automatic mode” to the task description.

## 6. Confirming the Task and Waiting for the Agent to Optimize

After the agent checks the environment, it shows the configuration and the execution plan. Verify the following items and confirm them in the conversation:

- Whether the machines, the available devices, and the model are correct.
- Whether the request length, the number of requests, and the concurrency meet the requirements of this service.
- Whether the performance target and the time and run budget meet your expectations.

After you confirm, **the agent performs the service startup, the baseline measurement, the ratio and parameter adjustment, the benchmark, the retest, and the result export**. You do not need to open another terminal to start services, modify configurations, or run benchmark commands. In interactive mode, the agent prompts you when it needs you to confirm the next candidate.

This case uses AISBench to measure the performance of the complete PD deployment and selects a candidate based on the end-to-end throughput, latency, and success rate. You do not need to prepare a separate environment for measuring the P and D sides individually.

During execution, you can talk to the agent directly:

| What You Want to Do | What You Can Say |
|---|---|
| Check the progress. | “Report the current progress, the best result, and the remaining budget.” |
| Pause. | “Clean up the current task, pause, and keep the results.” |
| Resume. | “Continue the previous optimization. Check the environment and the remaining budget first.” |
| Handle an exception. | “Analyze this failure and tell me the cause and whether administrator assistance is needed.” |

The agent keeps the task records. If you switch to another session, give the previous task directory to the new session to continue. You do not need to edit the records or clean up the service processes yourself.

## 7. Viewing the Results and Letting the Agent Reproduce Them if Needed

When the optimization finishes, the agent provides a summary and the report path. Open `final_report.md` and focus on the following:

| Item | What to Check |
|---|---|
| Optimal candidate | The P/D ratio, the device allocation, and the service parameters |
| Performance | Throughput, first token latency (TTFT), time per output token (TPOT), and request success rate |
| Comparison with the baseline | Whether the performance improves under the same workload and whether it meets the target |
| Retest conclusion | Whether the optimal candidate has been independently retested and whether the results are stable |
| Completion scope | The actual elapsed time, the number of measurements, and any unfinished items and their causes |

The criteria for completion are as follows: a successful complete PD measurement, independent retest conclusions for the baseline and the winning candidate, a report, and reproducible deployment files. Whether the performance meets the target is stated separately, and no improvement percentage is assumed.

The report should list the recommended P/D ratio, the throughput, TTFT, and TPOT of the baseline and the recommended candidate, the retest conclusion, and any unfinished items, one by one. This document does not provide example performance values, and the actual data depends on this measurement.

The deployment files are saved in `best_deployment/`, and the agent provides the exact path. If the retest is not complete, the report must state this clearly. Do not treat the best result of a single run as a stable gain.

When you need to reproduce the result, tell the agent: “Reproduce the optimal candidate in the original environment based on the exported results. Confirm that the resources are available before you start.” The agent prepares the specific plan, and after you confirm it, performs the deployment and verification. You do not need to copy scripts manually or start services machine by machine.

If you encounter a problem that cannot be resolved automatically, ask the agent to organize an error summary, the environment information, and the related logs, and then contact your administrator or support personnel. Do not send SSH private keys or access tokens in the conversation.
