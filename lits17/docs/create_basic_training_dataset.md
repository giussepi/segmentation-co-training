# Creating basic training dataset
The `LiTS17MGR` has options to create multiclass or binary datasets. Review its definition to use it properly. In the next lines we show an example to create a binary dataset (train, validation and test sub-datasets) using the lesion label.

``` python
    mgr = LiTS17MGR('LITS/train',
                    saving_path='LiTS17Lesion-Pro',
                    target_size=(368, 368, -1), only_liver=False, only_lesion=True)
    print(mgr.get_insights())
    print(mgr.get_lowest_highest_bounds())
    mgr()
    mgr.perform_visual_verification(68, scans=[40, 64], clahe=True)  # ppl 68 -> scans 64
    # after manually removing files without the desired label and less scans than 32
    # (000, 001, 054 had 29 scans) we ended up with 230 FILES @ LiTS17 only lesion and
    # 256 files @ LiTS17 only liver
    mgr.split_processed_dataset(.20, .20, shuffle=True)

    # getting subdatasets and plotting some crops #############################
    train, val, test = LiTS17Dataset.get_subdatasets(
        'LiTS17Lesion-Pro/train',
        'LiTS17Lesion-Pro/val',
        'LiTS17Lesion-Pro/test'
    )
    for db_name, dataset in zip(['train', 'val', 'test'], [train, val, test]):
        print(f'{db_name}: {len(dataset)}')
        data = dataset[0]
        print(data['image'].shape, data['mask'].shape)
        print(data['label'], data['label_name'], data['updated_mask_path'], data['original_mask'])
        print(data['image'].min(), data['image'].max())
        print(data['mask'].min(), data['mask'].max())

        if len(data['image'].shape) == 4:
            img_id = np.random.randint(0, data['image'].shape[-3])
            fig, axis = plt.subplots(1, 2)
            axis[0].imshow(
                equalize_adapthist(data['image'].detach().numpy()).squeeze().transpose(1, 2, 0)[..., img_id],
                cmap='gray'
            )
            axis[1].imshow(data['mask'].detach().numpy().squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
            plt.show()
        else:
            num_crops = dataset[0]['image'].shape[0]
            imgs_per_row = 4
            for ii in range(0, len(dataset), imgs_per_row):
                fig, axis = plt.subplots(2, imgs_per_row*num_crops)
                # for idx, d in zip([*range(imgs_per_row)], dataset):
                for idx in range(imgs_per_row):
                    d = dataset[idx+ii]
                    for cidx in range(num_crops):
                        img_id = np.random.randint(0, d['image'].shape[-3])
                        axis[0, idx*num_crops+cidx].imshow(
                            equalize_adapthist(d['image'][cidx].detach().numpy()
                                               ).squeeze().transpose(1, 2, 0)[..., img_id],
                            cmap='gray'
                        )
                        axis[0, idx*num_crops+cidx].set_title(f'CT{idx}-{cidx}')
                        axis[1, idx*num_crops+cidx].imshow(d['mask'][cidx].detach().numpy(
                        ).squeeze().transpose(1, 2, 0)[..., img_id], cmap='gray')
                        axis[1, idx*num_crops+cidx].set_title(f'Mask{idx}-{cidx}')

                fig.suptitle('CTs and Masks')
                plt.show()
```
