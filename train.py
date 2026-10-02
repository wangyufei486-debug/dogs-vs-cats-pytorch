from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from dataset import CatsDogsDataset
from model import CatsDogsCNN


PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "data" / "raw" / "train"
LOG_DIR = PROJECT_ROOT / "logs"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
MODEL_PATH = CHECKPOINT_DIR / "best_model.pth"


def main():
    # =========================
    # 1. 设置设备
    # =========================
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("当前使用设备：", device)

    # =========================
    # 2. 图片预处理
    # =========================
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    # =========================
    # 3. 创建完整数据集
    # =========================
    dataset = CatsDogsDataset(
        root_dir=TRAIN_DIR,
        transform=transform
    )

    print("完整数据集数量：", len(dataset))

    # =========================
    # 4. 划分训练集和验证集
    # =========================
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    print("训练集数量：", len(train_dataset))
    print("验证集数量：", len(val_dataset))

    # =========================
    # 5. 创建 DataLoader
    # =========================
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )

    # =========================
    # 6. 创建模型
    # =========================
    model = CatsDogsCNN()
    model = model.to(device)

    # =========================
    # 7. 损失函数
    # =========================
    loss_fn = nn.CrossEntropyLoss()

    # =========================
    # 8. 优化器
    # =========================
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # =========================
    # 9. TensorBoard
    # =========================
    writer = SummaryWriter(LOG_DIR)

    # =========================
    # 10. 创建模型保存文件夹
    # =========================
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # =========================
    # 11. 设置训练轮数
    # =========================
    epochs = 10
    # 保存目前最好的验证集准确率
    best_val_acc = 0.0

    # =========================
    # 12. 正式开始训练
    # =========================
    for epoch in range(epochs):
        print(f"\n---------- 第 {epoch + 1} 轮训练开始 ----------")

        # =====================================
        # 训练阶段
        # =====================================
        model.train()

        total_train_loss = 0.0
        total_train_correct = 0

        for imgs, labels in train_dataloader:
            # 把图片和标签放到 GPU / CPU
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = loss_fn(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss.item() 是当前 batch 的平均 loss
            # 乘当前 batch 图片数量，方便最后计算整个训练集平均 loss
            total_train_loss += loss.item() * imgs.size(0)

            # outputs.shape = [batch_size, 2]
            # 取两个类别中得分最大的那个
            preds = outputs.argmax(dim=1)

            # 预测正确的图片数量
            total_train_correct += (preds == labels).sum().item()

        # 整个训练集平均 loss
        train_loss = total_train_loss / len(train_dataset)
        # 整个训练集准确率
        train_acc = total_train_correct / len(train_dataset)

        # =====================================
        # 验证阶段
        # =====================================
        model.eval()

        total_val_loss = 0.0
        total_val_correct = 0

        with torch.no_grad():
            for imgs, labels in val_dataloader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                outputs = model(imgs)
                loss = loss_fn(outputs, labels)
                total_val_loss += loss.item() * imgs.size(0)

                preds = outputs.argmax(dim=1)

                total_val_correct += (preds == labels).sum().item()

        # 验证集平均 loss
        val_loss = total_val_loss / len(val_dataset)
        # 验证集准确率
        val_acc = total_val_correct / len(val_dataset)

        # =====================================
        # 打印这一轮结果
        # =====================================
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_acc:.4f}")

        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val Accuracy: {val_acc:.4f}")

        # =====================================
        # TensorBoard记录
        # =====================================
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_acc, epoch)

        # =====================================
        # 保存最佳模型
        # =====================================
        if val_acc > best_val_acc:
            best_val_acc = val_acc

            torch.save(
                model.state_dict(),
                MODEL_PATH
            )

            print("保存当前最佳模型！")

    # =========================
    # 13. 训练结束
    # =========================
    writer.close()

    print("\n训练结束！")
    print("最佳验证集准确率：", best_val_acc)


if __name__ == "__main__":
    main()
