import os
import cv2
import pandas as pd
import numpy as np
import jpeg4py as jpeg
from PIL import Image
from torch import from_numpy
from torch.utils.data import Dataset


class CxrDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if all([c in self.df.columns for c in self.cfg['classes']]):
            label = self.df.iloc[index][self.cfg['classes']].to_numpy().astype(np.float32)    
        else:
            label = np.zeros(len(self.cfg['classes']))

        path = self.df.iloc[index]["path"]
        path = os.path.join(self.cfg['data_dir'], path)
        resized_path = path.replace(".jpg", f"_resized_{self.cfg['size']}.jpg")

        if os.path.exists(resized_path):
            img = jpeg.JPEG(resized_path).decode()
            if os.path.exists(path):
                os.remove(path)
            assert img.shape == (self.cfg['size'], self.cfg['size'], 3)
        else:
            img = jpeg.JPEG(path).decode()
            img = cv2.resize(img, (self.cfg['size'], self.cfg['size']), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(resized_path, img)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']   
            img = np.moveaxis(img, -1, 0)

        return img, label 


class CxrBalancedDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        class_name = self.cfg['classes'][index%len(self.cfg['classes'])]
        df = self.df[self.df[class_name] == 1].sample(1).iloc[0]

        label = df[self.cfg['classes']].to_numpy().astype(np.float32)    

        path = df["path"]
        path = os.path.join(self.cfg['data_dir'], path)
        resized_path = path.replace(".jpg", f"_resized_{self.cfg['size']}.jpg")

        if os.path.exists(resized_path):
            img = jpeg.JPEG(resized_path).decode()
            assert img.shape == (self.cfg['size'], self.cfg['size'], 3)
        else:
            img = jpeg.JPEG(path).decode()
            img = cv2.resize(img, (self.cfg['size'], self.cfg['size']), interpolation=cv2.INTER_LANCZOS4)
            cv2.imwrite(resized_path, img)

        if self.transform:
            transformed = self.transform(image=img)
            img = transformed['image']   
            img = np.moveaxis(img, -1, 0)

        return img, label


class CxrStudyIdDataset(Dataset):
    def __init__(self, cfg, df, transform=None):
        self.cfg = cfg
        self.df = df.groupby("study_id")
        self.study_ids = list(self.df.groups.keys())
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        df = self.df.get_group(self.study_ids[index])
        if len(df) > 4:
            df = df.sample(4)
        if all([c in df.columns for c in self.cfg['classes']]):
            label = df[self.cfg['classes']].iloc[0].to_numpy().astype(np.float32)    
        else:
            label = np.zeros(len(self.cfg['classes']))

        imgs = []
        for i in range(len(df)):
            resized_dir = '/mnt/nfs_share/yangjz/dataset/MIMIC'
            path = df.iloc[i]["fpath"]
            resized_path = os.path.join(resized_dir, path)
            path = os.path.join(self.cfg['data_dir'], path)
            # resized_path = path.replace(".jpg", f"_resized_{self.cfg['size']}.jpg")

            def create_directories(path):
                path = os.path.dirname(path)
                try:
                    os.makedirs(path, exist_ok=True)
                except Exception as e:
                    print(f"创建路径时出错: {e}")

            create_directories(resized_path)

            # if os.path.exists(resized_path):
            #     img = jpeg.JPEG(resized_path).decode()
            #     if os.path.exists(path):
            #         os.remove(path)
            #     assert img.shape == (self.cfg['size'], self.cfg['size'], 3)
            # else:
            #     img = jpeg.JPEG(path).decode()
            #     img = cv2.resize(img, (self.cfg['size'], self.cfg['size']), interpolation=cv2.INTER_LANCZOS4)
            #     cv2.imwrite(resized_path, img)

            if os.path.exists(resized_path):
                img = Image.open(resized_path)
                assert img.size == (self.cfg['size'], self.cfg['size'])
                img = np.array(img)
            else:
                if not os.path.exists(path):
                    # 全部为0的图片
                    img = np.zeros((3, self.cfg['size'], self.cfg['size']))
                    # 写入文件
                    # not_exist_file = '/mnt/nvme_share/yangjz/Python/PythonProject/competition/CheXFusion/output/log/not_exist_file.txt'
                    # with open(not_exist_file, 'a') as f:
                    #     f.write(f"{path}\n")

                    imgs.append(img)
                    continue
                else:
                    img = Image.open(path)
                    img = img.resize((self.cfg['size'], self.cfg['size']), Image.LANCZOS)
                    img.save(resized_path)
                    img = np.array(img)

            if self.transform:
                # 转为三通道
                img = np.stack([img, img, img], axis=-1)
                transformed = self.transform(image=img)
                img = transformed['image']   
                img = np.moveaxis(img, -1, 0)
            imgs.append(img)

        img = np.stack(imgs, axis=0)    
        img = np.concatenate([img, np.zeros((4-len(df), 3, self.cfg['size'], self.cfg['size']))], axis=0).astype(np.float32)
        return img, label

