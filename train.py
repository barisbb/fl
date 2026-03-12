from utils.pytorch_models import ResNet50
from models.poolformer import create_poolformer_s12
from models.ConvMixer import create_convmixer
from models.MLPMixer import create_mlp_mixer
from utils.clients import GlobalClient
from utils.pytorch_utils import seed_everything


def train():
    seed_everything(42)

    csv_paths = [
        "Finland",
        "Ireland",
        "Serbia",
        "Austria",
        "Belgium",
        "Lithuania",
        "Portugal",
        "Switzerland",
    ]

    epochs = 3
    communication_rounds = 40
    channels = 10
    num_classes = 19

    # model = create_poolformer_s12(in_chans=channels, num_classes=num_classes)
    # model = create_mlp_mixer(channels, num_classes)
    # model = create_convmixer(channels=channels, num_classes=num_classes, pretrained=False)
    model = ResNet50("ResNet50", channels=channels, num_cls=num_classes, pretrained=False)

    global_client = GlobalClient(
        model=model,
        lmdb_path="",
        val_path="",
        csv_paths=csv_paths,
        fedawa_weight_opt_steps=20,
        fedawa_weight_opt_lr=0.05,
        fedawa_reg_coeff=1.0,
    )

    global_model, global_results = global_client.train(
        communication_rounds=communication_rounds,
        epochs=epochs,
    )
    print(global_results)


if __name__ == "__main__":
    train()
