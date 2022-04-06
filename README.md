# CP4101 B.Comp. Dissertation (Final Year Project)

**AY2021/2022**

**Liu Zechu**

See report [here](https://github.com/LiuZechu/vnla/blob/master/FYP_Final_Report_LiuZechu.pdf).

## Overview
This GitHub repo contains the code accompanying my FYP, which is an extension on _Vision-based Navigation with Language-based Assistance (VNLA) via Imitation Learning with Indirect Intervention_ by Khanh Nguyen, Debadeepta Dey, Chris Brockett, and Bill Dolan (access the original paper [here](1812.04155)). VNLA's GitHub repo can be accessed at (https://github.com/debadeepta/vnla)[https://github.com/debadeepta/vnla]. This work is also partly based on Yuxuan's extension on VNLA at [https://github.com/HanYuxuanHYX/NUS-CP3106](https://github.com/HanYuxuanHYX/NUS-CP3106).

## File Organisation
- `VNLA_OC`: files for VNLA agent modified with Object Co-occurrence
- `VNLA_MP`: files for Multi-priority VNLA agent
- `multi-priority`: files related to works on multi-priority agent
- `object_cooccurrence`: files related to works on object co-occurrence
- `FYP_Final_Report_LiuZechu.pdf`: my FYP report

## Running Experiments
1. Clone the original VNLA repo [here](https://github.com/debadeepta/vnla).
2. Follow the steps to:
a) [download data](https://github.com/debadeepta/vnla/tree/master/data),
b) [set up simulator](https://github.com/debadeepta/vnla/tree/master/code).
3. Copy the `VNLA_XX` folder in this repo to the original repo, replacing the `VNLA` directory under `root_dir/code/tasks/`.
4. Run experiments at `root_dir/code/tasks/VNLA/scripts/`

```bash
training:
$ bash train_main_results.sh [learned|none] [gpu id]
example: $ bash train_main_results.sh learned 0

evaluation:
$ bash eval_main_results.sh [learned|none] [seen|unseen] [gpu id]
example: $ bash eval_main_results.sh learned seen

no room experiment:
$ bash train_noroom.sh noroom_learned [gpu id]
$ bash eval_noroom.sh noroom_learned [seen|unseen]
```

Note that features not mentioned in my final report may not be implemented and thus errors may occur.

