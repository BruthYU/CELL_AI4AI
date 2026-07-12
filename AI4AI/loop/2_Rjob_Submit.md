## 以下为我基于Codex提交训练、推理、评估RJOB时使用的Prompt

- 训练Prompt
```text
阅读RJOB_SUBMISSION_RULES.md，使用以下路径作为config-path，job-name自己拟定, 提交训练（若指定多个path则依次执行）：
[指定YAML文件1]
[指定YAML文件2]
```

- 推理Prompt
```text
阅读RJOB_SUBMISSION_RULES.md，使用以下路径作为config-path，job-name自己拟定，同时判断出需要使用的submit_inference/run_inference脚本，提交推理（若指定多个path则依次执行）：
[指定YAML文件1]
[指定YAML文件2]
```

- 评估Prompt
```text 
阅读RJOB_SUBMISSION_RULES.md，使用以下文件夹路径作为input_dir，job-name自己拟定，同时判断出需要使用的submit_evaluate/run_evalute脚本，提交评估（若指定多个文件夹则依次执行）：
[指定文件夹1]
[指定文件夹2]

注意：评估的指定文件夹中只能存放一对h5ad文件 （相同数据集的不同split的评估结果请存放于不同文件夹）
```

## 示例
```text
阅读RJOB_SUBMISSION_RULES.md，使用以下路径作为config-path，job-name自己拟定, 提交训练（若指定多个path则依次执行）：
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/AI4AI/CELL_AI4AI/benchmark/workspace/20260701_121737_JIT_pbmc
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/AI4AI/CELL_AI4AI/benchmark/workspace/20260702_083428_JIT_replogle


阅读RJOB_SUBMISSION_RULES.md，使用以下路径作为config-path，job-name自己拟定，同时判断出需要使用的submit_inference/run_inference脚本，提交推理（若指定多个path则依次执行）：
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/nemo_cellflow/config/jit_tahoe.yaml

阅读RJOB_SUBMISSION_RULES.md，使用以下文件夹路径作为input_dir，job-name自己拟定，同时判断出需要使用的submit_evaluate/run_evalute脚本，提交评估（若指定多个文件夹则依次执行）：
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/AI4AI/CELL_AI4AI/benchmark/workspace/20260701_121737_JIT_pbmc
/mnt/shared-storage-gpfs2/beam-gpfs02/yulang/master/AI4AI/CELL_AI4AI/benchmark/workspace/20260702_083428_JIT_replogle
```