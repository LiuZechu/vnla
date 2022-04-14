# CP4101 B.Comp. Dissertation (Final Year Project)

**AY2021/2022**

**By Liu Zechu**

See report [here](https://github.com/LiuZechu/vnla/blob/master/FYP_Final_Report_LiuZechu.pdf).

See video demo [here](https://youtu.be/NuvTGVC_Fyw).

## Overview
This GitHub repo contains the code accompanying my FYP, which is an extension on _Vision-based Navigation with Language-based Assistance (VNLA) via Imitation Learning with Indirect Intervention_ by Khanh Nguyen, Debadeepta Dey, Chris Brockett, and Bill Dolan (access the original paper [here](https://arxiv.org/abs/1812.04155)). VNLA's GitHub repo can be accessed at <https://github.com/debadeepta/vnla>. This work is also partly based on Yuxuan's extension on VNLA at <https://github.com/HanYuxuanHYX/NUS-CP3106>.

## File Organisation in This Repo
- `VNLA_OC`: VNLA modified with Object Co-occurrence
- `VNLA_MP`: Multi-priority VNLA
- `multi-priority`: miscellaneous files related to multi-priority agent (e.g. dataset generation scripts, results analysis scripts)
- `object_cooccurrence`: miscellaneous files related to object co-occurrence (e.g. scripts to generate COMs)
- `FYP_Final_Report_LiuZechu.pdf`: my FYP Final Report

## File Organisation on SoC Server
If you have access to my files on SoC server, this section may be helpful for you.

For files in the `~/vnla/code/tasks/VNLA`/`VNLA_XX` folders, please refer to `README.md` files in `VNLA_MP` and `VNLA_OC` in this repo for more details.

On the server, the directory at `~/vnla/data/asknav` contains different versions of datasets. Here's what the prefixes mean: `imp`: implicit tasks, `mixed075`: mixes original and explicit multi-priority tasks in the ratio 0.75:0.25, `ori`: original tasks, `ori_trans`: original datasets transformed to suit the format that the new agent can process, `mp or [no prefix]`: explicit tasks. There are also some scripts to manipulate datasets. Note: when running an experiment on a particular version of datasets, remove the prefix (e.g. `imp_asknav_test_seen.json` --> `asknav_test_seen.json`).

Also, the directory at `~/vnla/data/noroom` on the server contains two versions with different prefixes: `ori`: original noroom datasets, `ex`: explicit multi-priority **test** datasets without room labels. Remove the prefixes when running.

The folder `~/vnla/code/tasks` has many `VNLA` or `VNLA_xx` folders:
- `VNLA`: [multi-priority agent](https://github.com/LiuZechu/vnla/tree/master/VNLA_MP)
- `VNLA_ORI`: original 
- `VNLA_ORI_COOC`: [original + modified with COM](https://github.com/LiuZechu/vnla/tree/master/VNLA_OC)
- `VNLA_QA`: Yuxuan's version
- `VNLA_QA_COOC`: Yuxuan's version + modified with COM
To run one of them, rename it as `VNLA`.

## Running Experiments
1. Clone the original VNLA repo [here](https://github.com/debadeepta/vnla).
2. Follow the steps to:
    * [download data](https://github.com/debadeepta/vnla/tree/master/data),
    * [set up simulator](https://github.com/debadeepta/vnla/tree/master/code).
3. Copy the `VNLA_XX` folder in this repo to the original repo, replacing the original `VNLA` directory under `root_dir/code/tasks/` with this folder.
    * If you want to run multi-priority VNLA, use `VNLA_MP`. Rename it as `VNLA`. Also, replace `root_dir/data/asknav/train_vocab.txt` with `multi-priority/train_vocab.txt` from this repo. Check `README.md` in `VNLA_MP` for more details.
    * If you want to run VNLA with object co-occurrence, use `VNLA_OC`. Rename it as `VNLA`. Check `README.md` in `VNLA_OC` for more details.
4. Run experiments at `root_dir/code/tasks/VNLA/scripts/`.

```
training:
$ bash train_main_results.sh [learned|none] [gpu id]
example: $ bash train_main_results.sh learned 0

evaluation:
$ bash eval_main_results.sh [learned|none] [seen|unseen] [gpu id]
example: 
(for Test Seen) 
$ bash eval_main_results.sh learned seen
(for Test Unseen)
$ bash eval_main_results.sh learned unseen

no-room experiment:
training:
$ bash train_noroom.sh noroom_learned [gpu id]
evaluation:
$ bash eval_noroom.sh [noroom_learned|asknav_learned] [seen|unseen]
```

Note: features not mentioned in my final report may not be implemented and thus errors may occur.


Should you have any questions regarding this repo or my FYP, please contact me at <liuzechu2013@gmail.com>.

