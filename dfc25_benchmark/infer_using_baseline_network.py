import sys

from model.model_factory import creatModel

sys.path.append('/home/chenhrx/project/BRIGHT/dfc25_benchmark')  # change this to the path of your project
import ttach as tta
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from make_data_loader import MultimodalDamageAssessmentDatset_Inference

from PIL import Image

import argparse
from datetime import datetime


class Inference:
    def __init__(self, args):
        self.args = args

        # Load dataset
        dataset = MultimodalDamageAssessmentDatset_Inference(args.val_dataset_path, args.val_data_name_list,
                                                             suffix='.tif')
        self.val_loader = DataLoader(dataset, batch_size=1, num_workers=1, drop_last=False)

        # Load model
        # self.model = UNet(in_channels=6, out_channels=4)
        self.model = creatModel(args, in_channels=6, out_channels=4)
        # self.model = SiamCRNN()

        self.model.eval()
        self.color_map = {
            0: (255, 255, 255),  # No damage - white
            1: (70, 181, 121),  # Intact - green
            2: (228, 189, 139),  # Damaged - orange
            3: (182, 70, 69)  # Destroyed - red
        }

        self.now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(os.path.join(self.args.inferece_saved_path, self.now_str)):
            os.makedirs(os.path.join(self.args.inferece_saved_path, self.now_str))
            os.makedirs(os.path.join(self.args.inferece_saved_path, self.now_str, 'raw'))
            os.makedirs(os.path.join(self.args.inferece_saved_path, self.now_str, 'color'))

        if args.existing_weight_path is not None:
            if not os.path.isfile(args.existing_weight_path):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.existing_weight_path))
            print("Loading weights...")
            checkpoint = torch.load(args.existing_weight_path, map_location=torch.device('cpu'))
            try:
                self.model.load_state_dict(checkpoint, strict=True)
                print('Pretrained Loading success!')
            except:
                new_state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
                try:
                    self.model.load_state_dict(new_state_dict, strict=False)
                    model_dict = {}
                    state_dict = self.model.state_dict()
                    for k, v in checkpoint.items():
                        if k in state_dict:
                            model_dict[k] = v
                    state_dict.update(model_dict)
                    self.model.load_state_dict(state_dict)
                    print('loading success after replace module')
                except Exception as inst:
                    print('pass loading weights')
                    print(inst)
            print('Loaded existing weights from {}'.format(args.existing_weight_path))
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.VerticalFlip(),
                tta.Rotate90(angles=[0, 90, 180, 270]),
                # tta.Scale(scales=[0.8, 1, 1.2]),
                # tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        self.model = tta.SegmentationTTAWrapper(self.model, transforms, merge_mode='mean')
        self.model = self.model.cuda().eval()

    def run_inference(self):
        print('Starting inference...')
        with torch.no_grad():
            for i, data in enumerate(tqdm(self.val_loader)):
                pre_change_imgs, post_change_imgs, file_name = data
                pre_change_imgs = pre_change_imgs.cuda()
                post_change_imgs = post_change_imgs.cuda()
                file_name = file_name[0]  # Get the filename as a string

                input_data = torch.cat([pre_change_imgs, post_change_imgs], dim=1)  # if use UNet
                output = self.model(input_data)  # if use UNet

                # _, output = self.model(pre_change_imgs, post_change_imgs) # If use SiamCRNN

                output = torch.argmax(output, dim=1).squeeze().cpu().numpy().astype(np.uint8)

                self.save_prediction_map(output, file_name)

    def save_prediction_map(self, prediction, file_name):
        """Saves the raw and colored prediction maps"""
        color_map_img = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.uint8)

        for cls, color in self.color_map.items():
            color_map_img[prediction == cls] = color

        raw_output_path = os.path.join(self.args.inferece_saved_path, self.now_str, 'raw',
                                       file_name + '_building_damage.png')  # upload this to leaderboard
        color_output_path = os.path.join(self.args.inferece_saved_path, self.now_str, 'color',
                                         file_name + '_building_damage.png')  # this is for your visualization
        Image.fromarray(prediction).save(raw_output_path)
        Image.fromarray(color_map_img).save(color_output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference on BRIGHT dataset")

    parser.add_argument('--model_name', default="uper")
    parser.add_argument('--encoder_name', default="mit_b4")
    parser.add_argument('--val_dataset_path', type=str, default='K:/dataset/dfc25track2/dfc25_track2_trainval/val')
    parser.add_argument('--val_data_list_path', type=str, default='dataset/splitname/val_setlevel.txt')
    parser.add_argument('--val_data_name_list', type=list)
    parser.add_argument('--existing_weight_path', type=str, default='pretained/uper_mitb4_66.pth')
    parser.add_argument('--inferece_saved_path', type=str, default='results')

    args = parser.parse_args()

    # Load test data list
    with open(args.val_data_list_path, "r") as f:
        args.val_data_name_list = [data_name.strip() for data_name in f]

    inference = Inference(args)
    inference.run_inference()
