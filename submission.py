from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from dataset import TestDataset
from model import CatsDogsCNN


PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "data" / "raw" / "test"
MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
SUBMISSION_PATH = OUTPUT_DIR / "submission.csv"


def main():
    # =========================
    # 1. 设置设备
    # =========================
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("当前使用设备：", device)

    # =========================
    # 2. 测试集图片预处理
    # =========================
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # =========================
    # 3. 创建测试集
    # =========================
    test_dataset = TestDataset(
        root_dir=TEST_DIR,
        transform=test_transform
    )

    print("测试集图片数量：", len(test_dataset))

    # =========================
    # 4. 创建 DataLoader
    # =========================
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # =========================
    # 5. 创建模型
    # =========================
    model = CatsDogsCNN()

    # =========================
    # 6. 加载最佳模型参数
    # =========================
    state_dict = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model = model.to(device)

    # 切换为测试模式
    model.eval()

    # =========================
    # 7. 保存预测结果
    # =========================
    all_ids = []
    all_probs = []

    # =========================
    # 8. 批量预测测试集
    # =========================
    with torch.no_grad():
        for imgs, image_ids in test_dataloader:
            # 图片放到GPU
            imgs = imgs.to(device)

            # 模型预测
            outputs = model(imgs)

            # logits → 概率
            probs = torch.softmax(outputs, dim=1)

            # 第0列 = cat概率
            # 第1列 = dog概率
            dog_probs = probs[:, 1]

            # 保存图片编号
            all_ids.extend(image_ids.tolist())

            # GPU Tensor → CPU → Python list
            all_probs.extend(dog_probs.cpu().tolist())

    # =========================
    # 9. 创建 submission
    # =========================
    submission = pd.DataFrame({
        "id": all_ids,
        "label": all_probs
    })

    # =========================
    # 10. 创建outputs文件夹
    # =========================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # 11. 保存CSV
    # =========================
    submission.to_csv(
        SUBMISSION_PATH,
        index=False
    )

    print("预测完成！")
    print("submission数量：", len(submission))

    print("\n前10行结果：")
    print(submission.head(10))

    print(f"\nsubmission.csv 已保存到：{SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
