# LiTS17 [^1]
## Getting the data
1. Download it from [https://competitions.codalab.org/competitions/17094](https://competitions.codalab.org/competitions/17094)

2. Create a new folder `LITS` at the main directory of this project

3. Uncompress `Training_Batch1.zip` and `Training_Batch2.zip` and move all the segmentation and volume NIfTI files to a new folder `train` inside `LITS`.


## Creating basic training dataset
See [docs/create_basic_training_dataset.md](docs/create_basic_training_dataset.md)


## Using LiTS17CropMGR and calculating min_crop_mean
See [docs/calculate_min_crop_mean.md](docs/calculate_min_crop_mean.md)

[^1]: P. Bilic et al., “The liver tumor segmentation benchmark (LiTS),” arXiv e-prints, p. arXiv:1901.04056, Jan. 2019. [Online]. Available: [https://arxiv.org/abs/1901.04056](https://arxiv.org/abs/1901.04056)
