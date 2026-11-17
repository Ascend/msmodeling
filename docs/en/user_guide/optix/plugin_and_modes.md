<a id="3-plugin-support"></a>

## 1 Plugin Support

Measured serving optimization supports user-defined search parameter configuration and benchmark tools. You only need to adapt the plugin mode and register the corresponding plugin. Built-in plugins are released with the `optix` package and can be selected directly through `-e`/`-b`. You can also integrate custom plugins through the plugin mode.

**Built-in Plugins**:

| Plugin Name | Type | Plugin Description |
| --- | --- | --- |
| vllm | Serving framework (`-e`) | vLLM serving framework adaptation |
| mindie | Serving framework (`-e`) | MindIE serving framework adaptation |
| ais_bench | Benchmark tool (`-b`) | AISBench benchmark tool adaptation |
| vllm_benchmark | Benchmark tool (`-b`) | vllm_benchmark benchmark tool adaptation |

**Extended plugins** (located in `contrib/optix/`, install as needed):

| Plugin Name | Type | Provider | Plugin Description | Plugin Directory |
| --- | --- | --- | --- | --- |
| custom_vllm | Serving framework (`-e`) | contrib example | Starts local and remote vLLM containers through Docker + SSH to demonstrate custom serving framework integration. | contrib/optix/vllm_msserviceprofiler/ |
| multihost_infer | Serving framework (`-e`) | Slim & agile commando steam | vLLM multi-node hybrid deployment plugin based on the multi-node Data Parallel (DP) plus headless solution. The master node provides APIs externally, and worker nodes only act as worker nodes. | contrib/optix/multihost_infer/ |
| vllm_pd | Serving framework (`-e`) | Slim & agile commando steam | vLLM Prefill-Decode (PD) disaggregated deployment cluster plugin. It manages multi-node start and stop and health checks remotely through SSH. | contrib/optix/vllm_pd_simulator/ |
| evalscopeperf | Benchmark tool (`-b`) | Slim & agile commando steam | Wraps the `evalscope perf` command as a benchmark tool to demonstrate custom benchmark tool integration. | contrib/optix/evalscopeplugin/ |

> Plugins are registered through Python entry points. For details about custom plugin development and usage, see [Procedure for Developing the Custom Plugin](../msmodeling_optix_plugin_user_guide.md).

<a id="5-quick-start"></a>

## 2 Quick Start

1. **Modify the configuration file**: Complete [Environment Preparation and Installation](../msmodeling_optix_user_guide.md#4-environment-preparation-and-installation), and modify the configuration file `config.toml` according to your actual situation, including the optimization parameters, benchmark tool parameters, and serving parameters. For details, see [Configuration File Description](configuration.md#7-configuration-file-description).

   > **Note:** If you are not sure which parameters and parameter ranges to adjust, you can use the PSO mode of the **`optix-assistant` optimization skill** (a unified entry point already routed in `AGENTS.md`). After you enter the hardware, model, load, and optimization objective, it returns the recommended parameters and search ranges. Then check them against the actual deployment environment (device memory, number of devices, latency requirements, and so on) and let it write them to `config.toml`.

2. **Start optimization**: By default, the tool performs `vLLM` serving parameter optimization based on `AISBench` (PSO mode). For other usages, see [Command Parameters](commands_and_results.md#8-command-parameters). To switch to agent mode, see [Optimization Modes: PSO and Agent](#6-optimization-modes-pso-and-agent).

    ```bash
    msmodeling optix
    ```

3. **View the results**: The optimization time depends on the model size, the data set size, and the number of optimization runs. When the optimization ends, a `data_storage_*.csv` file is generated and saved in the **`result/store` subdirectory in the current working directory** (you can change it with `[data_storage].store_dir`). For details, see [Output File Description](commands_and_results.md#9-output-file-description).

<a id="6-optimization-modes-pso-and-agent"></a>

## 3 Optimization Modes: PSO and Agent

Measured serving optimization provides two optimization modes. Select a mode through the top-level `optimizer_strategy` field in the configuration file. The default is `pso`:

| Optimization Mode | Configuration Value | Optimization Method | Feature | Applicable Scenario |
| --- | --- | --- | --- | --- |
| PSO mode (default) | `optimizer_strategy = "pso"` | The built-in PSO algorithm of the tool automatically generates parameter combinations and iteratively optimizes them. | Fully automatic operation without manual intervention. The search scale is directly controlled by `n_particles` and `iters`. | Routine automatic optimization and unattended long-duration optimization |
| Agent mode | `optimizer_strategy = "agent"` | The AI agent generates candidate parameters round by round. The tool validates the candidates, runs real-world tests, and writes back the results. The agent decides the direction of the next round based on the results of each round. | Each candidate has an explainable basis for selection. It can absorb historical experience and failure information, and supports manual confirmation or modification of candidates. | Scenarios that require direction decisions based on tuning experience, or scenarios in which you want to participate in candidate review |

The two modes share the optimization parameters, benchmark tool parameters, and serving parameters in [Configuration File Description](configuration.md#7-configuration-file-description). The differences are mainly in how the search is controlled:

- In PSO mode, `n_particles` and `iters` control the search scale. In agent mode, the `[agent_optimizer]` configuration section controls the number of rounds, the number of trials per round, and the time budget.
- Agent mode requires `use_request_rate_calibration = false`. In this case, `CONCURRENCY` participates in the search, and `REQUESTRATE` is fixed to the maximum stress rate, so that the relationship between concurrency and latency can be observed trial by trial. If you keep the default `true`, `CONCURRENCY` in candidates is fixed to `max`, and the concurrency search dimension becomes invalid.

<a id="how-agent-mode-works"></a>

### 3.1 How Agent Mode Works

For the installation and usage steps of PD disaggregation agent mode, see [From Environment Preparation to Agent-Driven PD Disaggregation Optimization](../../quick_start/optix_agent_pd_disaggregation.md).

In agent mode, the tool does not run PSO internally. Instead, it reads the candidate file `candidates.round-N.json` generated by the AI agent from the run directory corresponding to `[agent_optimizer].run_dir`, validates the candidates, runs real-world tests one by one, and writes back `results.round-N.json` and `progress.round-N.json`. It exits after the current round is complete only when the candidate file for the next round has not been generated yet. Convergence exit requires both the trend condition (the improvement over consecutive `convergence_rounds` rounds is less than 1%) and the coverage condition (the number of completed rounds is not less than `min_rounds` and the number of completed trials is not less than `min_trials`). It is also subject to the `max_rounds` upper limit, and trial execution in each round is subject to the `time_limit_minutes` time limit.

The fields of the `[agent_optimizer]` configuration section (all optional):

| Parameter | Default Value | Description |
| --- | --- | --- |
| run_id | `"default"` | Optimization run identifier. Candidate and result files are organized in the run directory by this identifier. |
| run_dir | Empty | Run directory. If left empty, it is automatically resolved to `.agent_optimizer/runs/{run_id}`. |
| max_rounds | 32 | Maximum number of optimization rounds. The optimization terminates naturally when this number is reached. |
| max_trials | 12 | Maximum number of trials executed per round. Excess trials are not executed. |
| time_limit_minutes | 1440 | Time limit (minutes) for executing trials in a single round. The default is 1440. The global budget is accumulated by the agent across rounds. |
| convergence_rounds | 6 | Number of consecutive rounds without significant improvement required to trigger convergence detection. |
| min_rounds | 4 | Minimum number of optimization rounds to complete before convergence. |
| min_trials | 4 | Minimum number of trials to complete before convergence. In agent mode, each round runs 1 trial, which is equivalent to `min_rounds`. The larger value takes effect. |

You are advised to use agent mode through the **`optix-assistant` optimization skill**. This skill provides supporting scripts such as configuration preflight (`config_preflight`), context collection (`collect_context`), configuration parameter check (`check_search_space`), and optimal configuration export (`export_best_config`), and follows a unified behavior contract for candidate generation, failure analysis, and candidate comparison rules. Typical usage process:

1. Complete the installation according to [Environment Preparation and Installation](../msmodeling_optix_user_guide.md#4-environment-preparation-and-installation), and confirm that the deployment stack is available.
2. Install and start the AI Agent tool. You are advised to use [MindStudioAgent](https://gitcode.com/Ascend/msagent), or install OpenCode and configure the model service according to the [OpenCode official documentation](https://dev.opencode.ai/v2/docs). After you enter the msmodeling repository directory, enter the following in the Agent conversation: “Please read the `.agents/skills/optix-assistant/SKILL.md` in this repository and use optix-assistant for agent mode optimization.” Then select the fully automatic or interactive execution method as prompted.
3. After the configuration preflight and context collection are complete, the agent reads the method card according to the target routing. The base configuration is registered to TOML through `config_writer`, and `context.search_space` is updated in each round, while candidate `params` contain only the values. After validation, the agent runs real-world tests. After the process exits in each round, copy `context.json` to `context.round-N.json`, and then prepare the next round. After convergence, export the configuration from the context archive of the round to which the best result belongs. If the archive is missing or damaged, the export is rejected.
