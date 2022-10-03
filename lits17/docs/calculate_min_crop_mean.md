# How to calculate min_crop_mean for LiTS17CropMGR

This example shows the code to calculate a proper `min_crop_mean` for a LiTS17 crop
dataset.

1. Create your lesion dataset of `368x368x<original number of scans>`.
   ``` python
    import matplotlib.pyplot as plt
	import numpy as np
	import torch

	from monai.transforms import ForegroundMask

	from lits17.processors import LiTS17MGR, LiTS17CropMGR
	from skimage.exposure import equalize_adapthist

    mgr = LiTS17MGR('/media/giussepi/TOSHIBA EXT/LITS/train',
	              	saving_path='/media/giussepi/TOSHIBA EXT/LiTS17Lesion368x368x-2-Pro',
                    target_size=(368, 368, -2), only_liver=False, only_lesion=True)
    mgr()
   ```
2. Manually remove the files without the desired lesion label. The files ids to
   remove are `[32, 34, 38, 41, 47, 87, 89, 91, 105, 106, 114, 115, 119]`.

3. Split your crops dataset into train 60%, val 20%, and test 20%

	``` python
	mgr.split_processed_dataset(.20, .20, shuffle=True)
	```

4. Create the crop dataset using a `min_crop_mean` of zero to disable that filter
   for now.

	``` python
    LiTS17CropMGR(
        '/media/giussepi/TOSHIBA EXT/LiTS17Lesion368x368x-2-Pro',
        patch_size=(160,160,32),
        patch_overlapping=(.75, .75, .75), only_crops_with_masks=True, min_mask_area=625e-6,
        foregroundmask_threshold=.59, min_crop_mean=0, crops_per_label=4, adjust_depth=True,
        centre_masks=True,
        saving_path='/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro-4PositiveCrops32x160x160'
    )()
	```

5. Use the following code to plot crops along with their masks and foreground
   masks. Do not forget to jot down the foreground mask mean for good and bad
   foreground masks.


   ``` python
    train, val, test = LiTS17CropDataset.get_subdatasets(
        '/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro-4PositiveCrops32x160x160/train',
        '/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro-4PositiveCrops32x160x160/val',
        '/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro-4PositiveCrops32x160x160/test'
    )
    for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
        print(f'{db_name}: {len(dataset)}')

        for data_idx in range(len(dataset)):
            data = dataset[data_idx]
            if len(data['image'].shape) == 4:
                img_ids = [np.random.randint(0, data['image'].shape[-3])]

                # uncomment these lines to only plot crops with masks
                # if 1 not in data['mask'].unique():
                #     continue
                # else:
                #     # selecting an idx containing part of the mask
                #     img_ids = data['mask'].squeeze().sum(axis=-1).sum(axis=-1).nonzero().squeeze()

                foreground_mask = ForegroundMask(threshold=.59, invert=True)(data['image'])
                std, mean = torch.std_mean(data['image'], unbiased=False)
                fstd, fmean = torch.std_mean(foreground_mask, unbiased=False)

				# once you have chosen a good mean, uncomment the following
                # lines and replace .63 with your chosen mean to verify that
                # only good crops are displayed.
                # if fmean < .63:
                #     continue

                print(f"SUM: {data['image'].sum()}")
                print(f"STD MEAN: {std} {mean}")
                print(f"SUM: {foreground_mask.sum()}")
                print(f"foreground mask STD MEAN: {fstd} {fmean}")

                for img_id in img_ids:
                    fig, axis = plt.subplots(1, 3)
                    axis[0].imshow(
                        equalize_adapthist(data['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
                        cmap='gray'
                    )
                    axis[0].set_title('Img')
                    axis[1].imshow(data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
                    axis[1].set_title('mask')
                    axis[2].imshow(foreground_mask.detach().numpy().squeeze()
                                   .transpose(1, 2, 0)[..., img_id], cmap='gray')
                    axis[2].set_title('foreground_mask')
                    plt.show()
                    plt.clf()
                    plt.close()
            else:
				break
   ```

6. We chose values considering foreground masks covering a good portion of
   the crop area. Our results were the following:

   ``` python
	good_mean = [.94, .69, .75, .67, .64, .63, ]

	bad_mean = [.59, .56, .46, .33, .27, .25, ]
   ```

7. Based on the previous notes we chose .63 as our `min_crop_mean`. To verify the results
   just run the same code to plot the crops but now uncomment the following
   lines and replace .63 by your custom value:
   ``` python
    # if fmean < .63:
    #      continue
   ```

8. One you are sure about your `min_crop_mean` value re-run the step 4 using
   the value you just found.
   ``` python
    LiTS17CropMGR(
        '/media/giussepi/TOSHIBA EXT/LiTS17Lesion368x368x-2-Pro',
        patch_size=(160,160,32),
        patch_overlapping=(.75, .75, .75), only_crops_with_masks=True, min_mask_area=625e-6,
        foregroundmask_threshold=.59, min_crop_mean=.63, crops_per_label=4, adjust_depth=True,
        centre_masks=True,
        saving_path='/media/giussepi/TOSHIBA EXT/LiTS17Lesion-Pro-4PositiveCrops32x160x160'
    )()
   ```
