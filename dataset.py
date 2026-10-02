import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CatsDogsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        # 1. 保存图片文件夹路径
        self.root_dir = root_dir

        # 2. 保存图片预处理方式
        self.transform = transform

        # 3. 获取文件夹里的所有 jpg 图片文件名
        self.files =  os.listdir(root_dir)

    def __getitem__(self, index):
        # 1. 根据 index 获取图片文件名
        file_name = self.files[index]

        # 2. 拼接出完整图片路径
        img_path = os.path.join(self.root_dir, file_name)

        # 3. 打开图片，并统一转换成 RGB 三通道
        img = Image.open(img_path).convert("RGB")

        # 4. 根据文件名生成标签
        # cat.xxx.jpg → 0
        # dog.xxx.jpg → 1
        if file_name.startswith("cat"):
            label = 0
        elif file_name.startswith("dog"):
            label = 1

        # 5. 如果设置了 transform，就对图片进行预处理
        if self.transform is not None:
            img = self.transform(img)

        # 6. 返回一张图片和它对应的标签
        return img, label

    def __len__(self):
        # 返回数据集中的图片总数
        return len(self.files)

# =========================
# Kaggle测试集Dataset
# =========================
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = os.listdir(root_dir)

        # 按图片编号进行排序
        # 例如：
        # 1.jpg
        # 2.jpg
        # 3.jpg
        # ...
        self.files.sort(
            key=lambda x: int(
                os.path.splitext(x)[0]
            )
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # 获取图片文件名
        file_name = self.files[index]
        # 完整图片路径
        img_path = os.path.join(self.root_dir,file_name)
        # 打开图片
        img = Image.open(img_path).convert("RGB")
        # 图片预处理
        if self.transform is not None:
            img = self.transform(img)
        # 例如：
        # "123.jpg" → 123
        image_id = int(os.path.splitext(file_name)[0])

        return img, image_id

# 测试 Dataset 是否能够正常工作
if __name__ == "__main__":

    project_root = Path(__file__).resolve().parent

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    dataset = CatsDogsDataset(
        root_dir=project_root / "data" / "raw" / "train",
        transform=transform
    )

    print("数据集图片总数：", len(dataset))

    img, label = dataset[0]

    print("第 0 张图片 shape：", img.shape)
    print("第 0 张图片 label：", label)
    print("第 0 张图片文件名：", dataset.files[0])
