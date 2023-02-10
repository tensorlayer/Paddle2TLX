# coding: utf-8
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import paddle
import numpy as np
import paddle.nn as nn
import paddle.optimizer as opt


# EPOCH_NUM = 10
EPOCH_NUM = 4
BATCH_SIZE = 8  # 64
BATCH_NUM = 4  # 100
IMAGE_SHAPE = [3, 224, 224]
CLASS_NUM = 1000


class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random(size=IMAGE_SHAPE).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class ModelTrainPaddle(object):
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name

    def train(self):
        loss_fn = nn.CrossEntropyLoss()
        adam = opt.Adam(learning_rate=0.001, parameters=self.model.parameters())

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = paddle.io.DataLoader(dataset,
                                         batch_size=BATCH_SIZE,
                                         shuffle=True,
                                         drop_last=True,
                                         num_workers=0)  # 2
        print(f"{self.model_name} begin...")
        for epoch_id in range(EPOCH_NUM):
            # for batch_id, (image, label) in enumerate(loader()):
            for batch_id, (image, label) in enumerate(loader):  # TODO
                out = self.model(image)
                if isinstance(out,list):
                    out = out[0]
                loss = loss_fn(out, label)
                loss.backward()
                adam.step()
                adam.clear_grad()
                print("Epoch {} batch {}: loss = {}".format(epoch_id, batch_id, np.mean(loss.numpy())))
                # for key in self.model.state_dict().keys():
                #     print(key, self.model.state_dict()[key].shape)
                # for key in self.model.state_dict().keys():
                #     print(key, self.model.state_dict()[key].shape)
                exit()
        print(f"{self.model_name} end...")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--model_name", default="alexnet", help="model name for loading")
    parser.add_argument(
        "--input_dir_pd",
        "-idp",
        default="../paddlers/paddlers",
        help="define project folder path for paddle")
    args = parser.parse_args()
    from examples.models_clas_pd import PaddleClassificationModel

    pd_project_path = args.input_dir_pd
    ModelPD = PaddleClassificationModel(pd_project_path, args.model_name)
    pd_model = ModelPD.pd_model
    Train = ModelTrainPaddle(pd_model, args.model_name)
    Train.train()
