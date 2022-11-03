import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='train')

    parser.add_argument("--model_name", default="VGG16", help="model name for loading")
    args = parser.parse_args()
    return args