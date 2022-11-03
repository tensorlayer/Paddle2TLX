# coding: utf-8
import os
import random
import paddle
# paddle.disable_static()  # TODO
import paddle.nn.functional as F
import numpy as np
from utils.load_image import load_image
import paddle.nn as nn
import paddle.optimizer as opt


EPOCH_NUM = 10
BATCH_SIZE = 8  # 64
BATCH_NUM = 4  # 100
IMAGE_SHAPE = [3, 224, 224]
CLASS_NUM = 1000





class RandomDataset(paddle.io.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        images = np.random.random(size=IMAGE_SHAPE).astype('float32')
        labels = np.random.randint(0, CLASS_NUM - 1, (1,)).astype('int64')
        return images, labels

    def __len__(self):
        return self.num_samples


def CatDogGenerator(mode="train"):
    base_dir = f"./data2cats/{mode}"
    images_paths = os.listdir(base_dir)
    label = -1  # not defined
    images = []
    labels = []
    if mode == "train":
        for i, image_path in enumerate(images_paths):
            label_name = image_path.split(".")[0]
            if label_name == "cat":
                label = 0
            elif label_name == "dog":
                label = 1
            image_path = os.path.join(base_dir, image_path)
            img = load_image(image_path)
            images.append(img)
            labels.append(label)
            assert len(images) == len(labels)
    else:
        pass
    index_list = list(range(len(images)))

    def data_generator():
        if mode == 'train':
            random.shuffle(index_list)
            images_list = []
            labels_list = []
            for idx in index_list:
                images_list.append(images[idx])
                label = np.reshape(labels[i], [1]).astype('int64')
                labels_list.append(label)
                if len(images_list) == BATCH_SIZE:
                    yield np.array(images_list), np.array(labels_list)
                    images_list, labels_list = [], []
            if len(images_list) > 0:
                yield np.array(images_list), np.array(labels_list)

    return data_generator


class ModelTrainPaddle(object):
    def __init__(self, model):
        self.model = model

    def train(self):


        self.model.train()
        # loss_fn = nn.CrossEntropyLoss()
        opt = opt.Adam(learning_rate=0.001, parameters=self.model.parameters())

        # create data loader
        # train_dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        # train_dataset = CatDogDataset(BATCH_NUM * BATCH_SIZE, "train")
        train_dataset = CatDogGenerator("train")
        # train_loader = paddle.io.DataLoader(train_dataset,
        #                                     batch_size=BATCH_SIZE,
        #                                     shuffle=True,
        #                                     drop_last=True,
        #                                     num_workers=0)  # 2
        for epoch_id in range(EPOCH_NUM):
            # for batch_id, (images, labels) in enumerate(train_loader()):
            for batch_id, (images, labels) in enumerate(train_dataset()):
                images = paddle.to_tensor(images)
                labels = paddle.to_tensor(labels)
                preds = self.model(images)
                # loss = loss_fn(preds, labels)
                loss = F.cross_entropy(preds, labels)
                avg_loss = paddle.mean(loss)
                acc = paddle.metric.accuracy(input=preds, label=labels)
                print("Epoch {} batch {}: loss = {}, acc = {}".format(epoch_id + 1,
                                                                      batch_id + 1,
                                                                      avg_loss.numpy()[0],
                                                                      acc.numpy()[0]))
                avg_loss.backward()
                opt.step()
                opt.clear_grad()


from models_pd.pd_hardnet import hardnet85
from models_pd.pd_vgg import vgg16

if __name__ == '__main__':
    args = parse_args()
    model = vgg16(pretrained=False, num_classes=2)
    if args.model_name == "hardnet85":
        model = hardnet85(pretrained=False)
    elif args.model_name == "hardnet85":



    Train = ModelTrainPaddle(model)
    Train.train()