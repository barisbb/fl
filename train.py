from utils.pytorch_models import ResNet50
from models.poolformer import create_poolformer_s12
from models.ConvMixer import create_convmixer
from models.MLPMixer import create_mlp_mixer
from utils.clients import GlobalClient
from utils.pytorch_utils import start_cuda
from utils.pytorch_utils import seed_everything


def train():
    seed_everything(42)

    # Office-Caltech10 domains = clients
    csv_paths = ["amazon", "caltech", "dslr", "webcam"]

    epochs = 2
    communication_rounds = 40
    channels = 3
    num_classes = 10

    office_caltech_root = "/home/baris/fedavg/office_caltech_10"   # change this

    # model = create_poolformer_s12(in_chans=channels, num_classes=num_classes)
    # model = create_mlp_mixer(channels, num_classes)
    # model = create_convmixer(channels=channels, num_classes=num_classes, pretrained=False)
    model = ResNet50("ResNet50", channels=channels, num_cls=num_classes, pretrained=False)

    global_client = GlobalClient(
        model=model,
        lmdb_path=office_caltech_root,
        val_path="",
        csv_paths=csv_paths,
        num_classes=num_classes,
    )

    global_model, global_results = global_client.train(
        communication_rounds=communication_rounds,
        epochs=epochs
    )
    print(global_results)


if __name__ == '__main__':
    train()
