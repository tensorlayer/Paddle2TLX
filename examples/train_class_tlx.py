# coding: utf-8
import os
os.environ['TL_BACKEND'] = 'paddle'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import tensorlayerx as tlx
import numpy as np
import tensorlayerx.optimizers as opt


# EPOCH_NUM = 10
EPOCH_NUM = 4
BATCH_SIZE = 8  # 64
BATCH_NUM = 4  # 100
IMAGE_SHAPE = [3, 224, 224]
CLASS_NUM = 1000


class RandomDataset(tlx.dataflow.Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __getitem__(self, idx):
        image = np.random.random(size=IMAGE_SHAPE).astype('float32')
        label = np.random.randint(0, CLASS_NUM - 1, (1,)).astype('int64')
        return image, label

    def __len__(self):
        return self.num_samples


class ModelTrainTLX(object):
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
    def train(self):
        # loss_fn = nn.CrossEntropyLoss()
        loss_fn = tlx.losses.softmax_cross_entropy_with_logits
        # print(f"model.parameters()={model.parameters()}")
        adam = opt.Adam(lr=0.001)

        # create data loader
        dataset = RandomDataset(BATCH_NUM * BATCH_SIZE)
        loader = tlx.dataflow.DataLoader(dataset,
                                      batch_size=BATCH_SIZE,
                                      shuffle=True,
                                      drop_last=True,
                                      num_workers=0)  # 2
        print(f"{self.model_name} begin...")
        for epoch_id in range(EPOCH_NUM):
            for batch_id, (image, label) in enumerate(loader):
                out = self.model(image)
                if isinstance(out, list):
                    out = out[0]
                # print(f"type(out)={type(out)}")
                loss = loss_fn(tlx.convert_to_tensor(out), label)
                # loss.backward()
                # adam.step()
                # adam.clear_grad()
                grads = adam.gradient(loss, self.model.trainable_weights)
                adam.apply_gradients(zip(grads, self.model.trainable_weights))
                print("Epoch {} batch {}: loss = {}".format(epoch_id, batch_id, np.mean(loss.numpy())))
                # for key in self.model.state_dict().keys():
                #     print(key, self.model.state_dict()[key].shape)
                exit()
        print(f"{self.model_name} end...")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='train')
    parser.add_argument("--model_name", default="alexnet", help="model name for loading")
    parser.add_argument(
        "--output_dir_tlx",
        "-odt",
        default="../paddleclas/paddlers",
        help="path to save the converted model of tensorlayerx")
    args = parser.parse_args()

    from examples.models_clas_tlx import TLXClassificationModel

    tlx_project_path = args.output_dir_tlx
    ModelTLX = TLXClassificationModel(tlx_project_path, args.model_name)
    tlx_model = ModelTLX.tlx_model
    Train = ModelTrainTLX(tlx_model, args.model_name)
    Train.train()
