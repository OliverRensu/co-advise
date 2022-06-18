# Co-advise
This is the official implementation of CVPR2022 paper "Co-advise: Cross Inductive Bias Distillation"
## Prepare
Install RedNet following https://github.com/d-li14/involution

Install the rest packages following https://github.com/microsoft/Swin-Transformer

## Training

```shell
bash train.sh
```
Noted that mix precision should be closed, because involution is not supported.
## Pretrained Model
The checkpoints can be found at [Goolge Drive](https://drive.google.com/drive/folders/1qZosjeFctKBdBpcyOJ6k1jjcsJYxDGOu?usp=sharing)

## Citation
```
@InProceedings{Ren_coadvise_2022_CVPR,
    author    = {Sucheng Ren, Zhengqi Gao, Tianyu Hua, Zihui Xue, Yonglong Tian, Shengfeng He, Hang Zhao},
    title     = {Co-advise: Cross Inductive Bias Distillation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
}
```

If you have any questions, feel free to email Sucheng Ren :) (oliverrensu@gmail.com)