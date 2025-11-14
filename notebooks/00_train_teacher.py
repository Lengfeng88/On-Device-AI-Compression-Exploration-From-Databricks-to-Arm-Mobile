# Databricks notebook source
# MAGIC %md
# MAGIC ## 00. Train Teacher Model (ResNet-18)

# COMMAND ----------
import torch, torchvision, mlflow
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# === Set up a unified experiment ===
mlflow.set_experiment("/Shared/Arm_Compression")

device = torch.device("cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
trainset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64)

model = torchvision.models.resnet18(num_classes=10)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

model.train()
for epoch in range(10):
    for x, y in trainloader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} done")

# ✅ Create input_example (CIFAR-10 format: 32x32x3)
input_example = torch.randn(1, 3, 32, 32).numpy()  # batch_size=1, C=3, H=32, W=32

# Save to MLflow and tag it 
with mlflow.start_run(run_name="Teacher_ResNet18") as run:
    mlflow.pytorch.log_model(
        model,
        "teacher_model",
        input_example=input_example  
    )
    mlflow.set_tag("role", "teacher")
print(f"✅ Teacher model has been saved，Run ID: {run.info.run_id}")