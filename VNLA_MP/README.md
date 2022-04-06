# Multi-priority VNLA directory

This directory contains files for multi-priority VNLA [default: **explict** tasks]. To run it, replace the original VNLA folder under `root_dir/code/tasks/` with this folder, and change its name from `VNLA_MP` to `VNLA`.

To run it on **implicit** tasks, replace `train.py` with `train_implicit.py` and change rename it as `train.py`. Similarly, replace `eval.py` with `eval_implicit.py` and rename it as `eval.py`.

## Notes on file organisation on the server
If you have access to my files on the SoC server, this folder corresponds to `~/vnla/code/tasks/VNLA`. There are many more files in that folder (which aren't really needed to run an experiment). I'll summarise what these files are here:
- `datasets`, `noroom_exmp_datasets`, `ori_noroom_datasets`, `original_datasets`, `v3_datasets`: these folders contain various versions of datasets. By right they shouldn't be in this folder, but I happened to generate some of them here and they were put here as backups.
- `v2_mp_dataset_generator.py`: script to generate explicit multi-priority datasets.
- `v3_mp_dataset_generator.py`: script to generate implicit multi-priority datasets. (All the "v3" stuff is related to implicit tasks)
- `output_xx` folders: `output` folders for different runs of the programme. Each training run generates an `output` that contains the checkpointed trained model and evaluation results. Some notable ones: `output_mp04` explicit tasks, `output_ori` original tasks, `output_v3_bs100` implicit tasks.
- `ori_xxxx` files: just put here for easy comparison sometimes. They are the original VNLA files. 
- `intermediate_scores.txt`: each time the agent is evaluated (val set or test set), it appends the evaluation metrics stats here. Recreate it each time you run the programme. Then you can use the `vnla/multi-priority/metrics_analysis.py` script in this repo to analyse how these metrics change over iterations.