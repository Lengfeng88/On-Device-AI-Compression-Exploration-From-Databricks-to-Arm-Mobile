# Databricks notebook source
# MAGIC %md
# MAGIC ## 03a: Train MobileNetV3-Small on CIFAR-10 (NAS Model)

# COMMAND ----------
import torch, torchvision, mlflow, time, os, tempfile
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# === Cleanup ===
while mlflow.active_run():
    mlflow.end_run()

mlflow.set_experiment("/Shared/Arm_Compression")
device = torch.device("cpu")

# === Data ===
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# === Model ===
model = torchvision.models.mobilenet_v3_small(num_classes=10)
model.features[0][0] = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
model = model.to(device)

# === Optimizer & Loss ===
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

# === Evaluation Function ===
def evaluate(model):
    model.eval(); correct = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            correct += (model(x).argmax(1) == y).sum().item()
    return 100 * correct / len(testset)

# === Training + Save in ONE MLflow run ===
with mlflow.start_run(run_name="NAS_MobileNetV3_Trained") as run:
    # Train
    for epoch in range(5):
        model.train()
        running_loss = 0.0
        for i, (x, y) in enumerate(trainloader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        acc = evaluate(model)
        print(f"Epoch {epoch+1}: Loss={running_loss/len(trainloader):.3f}, Acc={acc:.2f}%")
        mlflow.log_metrics({"train_loss": running_loss/len(trainloader), "val_acc": acc}, step=epoch)
    
    # ✅ CRITICAL: Save model INSIDE a "models" subfolder
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = os.path.join(tmp_dir, "models")
        os.makedirs(models_dir, exist_ok=True)  # ←← 创建 models/ 目录
        
        model_path = os.path.join(models_dir, "mobilenetv3_cifar10.pth")
        torch.save(model.state_dict(), model_path)
        
        # Log the entire "models" folder
        mlflow.log_artifact(model_path, "models")  # ←← 路径必须匹配！

    mlflow.set_tag("method", "NAS")
    print(f"✅ Model saved to MLflow. Run ID: {run.info.run_id}")