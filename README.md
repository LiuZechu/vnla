# CP4101 B.Comp. Dissertation (Final Year Project)

**AY2021/2022**

**By Liu Zechu**

See report [here](https://github.com/LiuZechu/vnla/blob/master/FYP_Final_Report_LiuZechu.pdf).

## Overview
This GitHub repo contains the code accompanying my FYP, which is an extension on _Vision-based Navigation with Language-based Assistance (VNLA) via Imitation Learning with Indirect Intervention_ by Khanh Nguyen, Debadeepta Dey, Chris Brockett, and Bill Dolan (access the original paper [here](https://arxiv.org/abs/1812.04155)). VNLA's GitHub repo can be accessed at <https://github.com/debadeepta/vnla>. This work is also partly based on Yuxuan's extension on VNLA at <https://github.com/HanYuxuanHYX/NUS-CP3106>.

## File Organisation
- `VNLA_OC`: VNLA modified with Object Co-occurrence
- `VNLA_MP`: Multi-priority VNLA
- `multi-priority`: miscellaneous files related to multi-priority agent (e.g. dataset generation scripts, results analysis scripts)
- `object_cooccurrence`: miscellaneous files related to object co-occurrence (e.g. scripts to generate COMs)
- `FYP_Final_Report_LiuZechu.pdf`: my FYP Final Report

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

