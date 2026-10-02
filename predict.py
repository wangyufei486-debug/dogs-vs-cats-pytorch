import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from model import CatsDogsCNN


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"
DEFAULT_IMAGE_PATH = PROJECT_ROOT / "data" / "predict" / "dog.png"


def parse_args():
    parser = argparse.ArgumentParser(description="使用训练好的模型预测单张猫狗图片")
    parser.add_argument(
        "image",
        nargs="?",
        type=Path,
        default=DEFAULT_IMAGE_PATH,
        help="待预测图片路径（默认：data/predict/dog.png）",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="模型权重路径（默认：checkpoints/best_model.pth）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # =========================
    # 1. 设置设备
    # =========================
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    print("当前使用设备：", device)

    # =========================
    # 2. 创建模型
    # =========================
    model = CatsDogsCNN()

    # =========================
    # 3. 加载训练好的模型参数
    # =========================
    state_dict = torch.load(
        args.model,
        map_location=device,
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)

    # 切换到预测模式
    model.eval()

    # =========================
    # 4. 图片预处理
    # =========================
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # =========================
    # 5. 读取需要预测的图片
    # =========================
    img = Image.open(args.image).convert("RGB")

    # =========================
    # 6. 图片预处理
    # =========================
    img = transform(img)

    print("预处理后的图片形状：", img.shape)

    # =========================
    # 7. 增加batch维度
    # =========================
    img = img.unsqueeze(0)
    print("增加batch维度后：", img.shape)

    # 把图片放到GPU
    img = img.to(device)

    # =========================
    # 8. 模型预测
    # =========================
    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)

    # =========================
    # 9. 获取猫狗概率
    # =========================
    cat_prob = probs[0][0].item()
    dog_prob = probs[0][1].item()

    print(f"\n猫的概率：{cat_prob:.2%}")
    print(f"狗的概率：{dog_prob:.2%}")

    # =========================
    # 10. 最终预测结果
    # =========================
    prediction = probs.argmax(dim=1).item()

    if prediction == 0:
        print("预测结果：Cat ")
    else:
        print("预测结果：Dog ")


if __name__ == "__main__":
    main()
