import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from utils.pytorch_models import ResNet50
from models.poolformer import create_poolformer_s12
from models.ConvMixer import create_convmixer
from models.MLPMixer import create_mlp_mixer
from utils.clients import GlobalClient
from utils.pytorch_utils import start_cuda
from utils.pytorch_utils import seed_everything


def train():
	seed_everything(42)
	csv_paths = ["Finland","Ireland","Serbia","Austria", "Belgium", "Lithuania", "Portugal", "Switzerland"] #this means that there are 3 clients that includes the images of a specific country. You can add Austria, Belgium, Lithuania, Portugal, Switzerland
	epochs = 2
	communication_rounds = 40
	channels = 10
	num_classes = 19
	#model = create_poolformer_s12(in_chans=channels, num_classes=num_classes)
	#model = create_mlp_mixer(channels, num_classes)
	#model = create_convmixer(channels=channels, num_classes=num_classes, pretrained=False)
    #model = create_poolformer_s12(in_chans=channels, num_classes=num_classes)
	model = ResNet50("ResNet50", channels=channels, num_cls=num_classes, pretrained=False)
	global_client = GlobalClient(
		model=model,
		lmdb_path="",
		val_path="",
		csv_paths=csv_paths,
	)
	global_model, global_results = global_client.train(communication_rounds=communication_rounds, epochs=epochs)
	print(global_results)


if __name__ == '__main__':
    train()