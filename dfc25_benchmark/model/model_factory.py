import segmentation_models_pytorch as smp


def creatModel(args=None,
               in_channels=6,
               out_channels=4,
               activation=None,
               encoder_weights="imagenet",
               encoder_name='mit_b4'):
    if args.encoder_name is not None:
        encoder_name = args.encoder_name
    model = smp.UPerNet(
        classes=out_channels,
        in_channels=in_channels,
        activation=activation,
        encoder_weights=encoder_weights,
        encoder_name=encoder_name,
    )
    return model
