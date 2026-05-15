# RFC: Support Ulysses for Diffusers Model Proposal

## Metadata

| Item | Details |
|:-----|:--------|
| **Status** | Approved |
| **Author** | jia_ya_nan |
| **Creation Date** | 2025-12-25 |
| **Related Links** | [support ulysses for diffuser model](https://gitcode.com/Ascend/msit/pull/4907) |

---

## 1. Overview

This proposal aims to support ulysses parallelism in the diffuser model

## 2. Detailed Design

- Split input for Dit.
- Add communication in attention.

### 2.1 Proposed solution

#### Part 1: Split input for Dit

It is judged to be segmented on the h or w dimensions. Finally, the result of Dit needs to be all_gather on the segmentation dimension to get a complete output.
To achieve this, we modify the code as follows:

1. add ulysses-size in Parallel_Config.
2. add `process_input` in `video_generation.py`
3. add `all_gather` after dit forward if use ulysses.

#### Part 2: Add communication in attention

To support ulysses parallelism in the attention layer, we need to modify the forward function of the attention layer.

1. add `get_sp_group` to manager communication.
2. add `all_to_all` in attention layer.

As for the communication in attention layer, we assume the input size is [b, s, head_num, head_dim], p is the ulysses-size.
After `all_to_all`, we get each part is [b, s * p, head_num, head_dim / p].
So the communication cost is `input_len / p * (p  - 1)`.

## 3. Plan

**TODO LIST**

- [ ] Add extra checks in ut to make sure ulysses parallelism works.
