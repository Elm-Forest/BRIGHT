import segmentation_models_pytorch as smp


def creatModel(args=None,
               in_channels=6,
               out_channels=4,
               activation=None,
               encoder_weights="imagenet",
               encoder_name='mit_b4'):
    model = smp.DeepLabV3Plus(
        classes=out_channels,
        in_channels=in_channels,
        activation=activation,
        encoder_weights=encoder_weights,
        encoder_name=encoder_name,
    )
    return model
