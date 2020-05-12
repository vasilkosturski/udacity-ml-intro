import argparse

parser = argparse.ArgumentParser(description='Image Recognition app')

parser.add_argument('--top_k', action="store",
                    dest="top_k", type=int)

parser.add_argument('--gpu', action="store_true",
                    default=False)

parser.add_argument('positional', action="store")

print(parser.parse_args())
