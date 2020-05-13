import argparse

parser = argparse.ArgumentParser(description='Train Image Classifier.')

parser.add_argument('data_directory', action="store")

parser.add_argument('--save_dir',
                    action="store",
                    default=".",
                    dest='save_dir',
                    type=str
                    )

parser.add_argument('--save_name',
                    action="store",
                    default="checkpoint",
                    dest='save_name',
                    type=str
                    )

parser.add_argument('--categories_json',
                    action="store",
                    default="cat_to_name.json",
                    dest='categories_json',
                    type=str
                    )

parser.add_argument('--arch',
                    action="store",
                    default="vgg16",
                    dest='arch',
                    type=str
                    )

parser.add_argument('--gpu',
                    action="store_true",
                    dest="use_gpu",
                    default=False)

hp = parser.add_argument_group('hyperparameters')

hp.add_argument('--learning_rate',
                action="store",
                default=0.001,
                type=float)

hp.add_argument('--hidden_units', '-hu',
                action="store",
                dest="hidden_units",
                default=[3136, 784],
                type=int,
                nargs='+')

hp.add_argument('--epochs',
                action="store",
                dest="epochs",
                default=1,
                type=int,
                help='Epochs')

parser.parse_args()
