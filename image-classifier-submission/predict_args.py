import argparse


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('path_to_image', action="store")

    parser.add_argument('checkpoint_file', action="store")

    parser.add_argument('--save_dir',
                        action="store",
                        default=".",
                        dest='save_dir',
                        type=str)

    parser.add_argument('--top_k',
                        action="store",
                        default=5,
                        dest='top_k',
                        type=int)

    parser.add_argument('--category_names',
                        action="store",
                        default="cat_to_name.json",
                        dest='categories_json',
                        type=str)

    parser.add_argument('--gpu',
                        action="store_true",
                        dest="use_gpu",
                        default=False)

    return parser.parse_args()
