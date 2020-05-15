import train_args
import sys


def main():
    args = train_args.get_args()
    print(args.data_directory)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)


