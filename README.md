# On Balancing Bias and Variance in Unsupervised Multi-Source-Free Domain Adaptation
This repository contains the code for implementing [On Balancing Bias and Variance in Unsupervised Multi-Source-Free Domain Adaptation](https://arxiv.org/pdf/2202.00796.pdf).

## Requirements
- python == 3.8.8
- pytorch == 1.10.0
- torchvision == 0.11.1
- numpy, scipy, sklearn, argparse, tqdm, PIL

## Datasets
### Office
- Please manually download the three datasets: [Office](https://drive.google.com/file/d/0B4IapRTv9pJ1WGZVd1VDMmhwdlE/view), [Office-Caltech](https://github.com/jindongwang/transferlearning/tree/master/data#office-caltech10), and [Office-Home](https://drive.google.com/file/d/0B81rNlvomiwed0V1YUxQdC1uOTg/view).
- Please create a directory named "data", and move `gen_list.py` inside the directory.
- To generate the image list file for each office dataset,
```
python ./data/gen_list.py
```

## Usage
- Take one dataset [Office] as an example.
- To train the source models,
```
python train_source.py --dset office --s 0 --max_epoch 100 --trte val --gpu_id 0 --output ckps/source/
```
- Please complete the training of all source models before starting domain adaptation.
- To adapt the source models to target,
```
python adapt.py --dset office --t 0 --max_iterations 20 --gpu_id 0 --output_src ckps/source/
```

## Reference
The implementation is based on this repo: [DECISION](https://github.com/driptaRC/DECISION).

## Citation
If this code is helpful for your research, please consider citing our paper.
```

@inproceedings{shen2023balancing,
  title={On Balancing Bias and Variance in Unsupervised Multi-Source-Free Domain Adaptation},
  author={Shen, Maohao and Bu, Yuheng and Wornell, Gregory W},
  booktitle={International Conference on Machine Learning},
  pages={30976--30991},
  year={2023},
  organization={PMLR}
}
```




