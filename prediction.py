
def predict():
    subjects_list = get_subjects()

    transforms = (
        ZNormalization(),
        RandomNoise(std_range=(0, 0.25)),
        RandomAffine(scales=(0.9, 1.1), degrees=10),
        RandomFlip(axes=(0,)),
    )
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(subjects_list, transform)

    img = subjects_dataset[0]['mri']['data'].numpy()

    patch_size = 96, 96, 96
    patch_overlap = 2, 2, 2
    batch_size = 4
    sample = subjects_dataset[0]
    sampler = LabelSampler(sample, patch_size)
    patch = sampler.extract_patch(sample, patch_size)

    writer = SummaryWriter('runs/test')
    #grid = torchvision.utils2.make_grid(patch['mri']['data'][0, ...].max(dim=0)[0])
    #writer.add_image('test', patch['mri']['data'][0, :, 50, :], 0, dataformats='HW')
    writer.add_figure('new', matplotlib_imshow(patch))
    writer.close()

    print()
    plt.imshow(patch['mri']['data'][0, ...].mean(dim=0))
    plt.show()
    plt.imshow(np.max(patch['label']['data'][0, ...].numpy(), axis=0))
    plt.show()

    return
    grid_sampler = GridSampler(img[0, :, :, :], patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)

    patch = next(iter(patch_loader))


    #print(patch['image'].shape)
    plt.imshow(patch['image'][0, 0, :, 50, :])
    plt.show()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)



    #unet = model.UNet3D(in_channels=1, out_channels=1,
    #                    final_sigmoid=True,
    #                    f_maps=64)

    # use multiple gpus
    # if torch.cuda.device_count() > 1:
    #    print("Let's use", torch.cuda.device_count(), "GPUs!")
    #    # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
    #    unet = nn.DataParallel(unet)

    #unet.to(device)

    #input = patch['image'].to(device)


def evaluate():

    subjects_list = get_subjects()

    transforms = (
        ZNormalization(),
        RandomNoise(std_range=(0, 0.25)),
        RandomAffine(scales=(0.9, 1.1), degrees=10),
        RandomFlip(axes=(0,)),
    )
    transform = Compose(transforms)
    subjects_dataset = ImagesDataset(subjects_list, transform)

    img = subjects_dataset[0]['mri']['data'].numpy()

    patch_size = 96, 96, 96
    patch_overlap = 4, 4, 4
    batch_size = 6
    CHANNELS_DIMENSION = 1

    net = unet.model.ResidualUNet3D(in_channels=1, out_channels=1,
                            final_sigmoid=True,
                            f_maps=64)


    checkpoint = torch.load('/mnt/share/raheppt1/pytorch_models/model2.pt')
    net.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print(epoch)
    loss = checkpoint['loss']
    net.eval()

    # Select GPU with CUDA_VISIBLE_DEVICES=x python main.py
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Assuming that we are on a CUDA machine, this should print a CUDA device:
    print(device)
    net.to(device)

    patch_overlap = 4, 4, 4
    grid_sampler = GridSampler(img[0,:,:,:], patch_size, patch_overlap)
    patch_loader = DataLoader(grid_sampler, batch_size=batch_size)
    aggregator = GridAggregator(img[0,:,:,:], patch_overlap)

    with torch.no_grad():
        for patches_batch in tqdm(patch_loader):
            input_tensor = patches_batch['image'].to(device)
            locations = patches_batch['location']
            logits = net(input_tensor)  # some unet

            sigmoid_fnc = torch.nn.Sequential(
                torch.nn.Sigmoid())
            logits = sigmoid_fnc(logits)

            #plt.imshow(logits[0, 0, :, 50, :].cpu().detach())
            #plt.show()
            aggregator.add_batch(logits, locations)

    output_array = aggregator.output_array
    print(output_array.shape)
    plt.imshow(np.max(img[0, :, :, :], axis=2), cmap='gray')
    plt.imshow(np.mean(output_array, axis=2), alpha = 0.6)
    plt.show()
