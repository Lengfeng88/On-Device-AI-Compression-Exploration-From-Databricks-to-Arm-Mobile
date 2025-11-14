# Databricks notebook source
# MAGIC %md
# MAGIC ## 3. NAS (MobileNetV3-Small)

# COMMAND ----------
import torch, torchvision, mlflow, time, os, tempfile
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pyspark.sql import SparkSession, Row
from mlflow import MlflowClient

# === Force clean up unclosed run ===
while mlflow.active_run():
    mlflow.end_run()

mlflow.set_experiment("/Shared/Arm_Compression")
device = torch.device("cpu")
spark = SparkSession.builder.getOrCreate()

# === Data ===
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
testset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# === Load fine-tuned MobileNetV3 from MLflow ===
# ⚠️ Replace with your actual Run ID from 03a_train_mobilenetv3.py
nas_run_id = "b5834b41bba1481f9990e1a35084991f"  

# Load model
model = torchvision.models.mobilenet_v3_small(num_classes=10)
# Adjust first conv for 32x32 input
model.features[0][0] = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
with tempfile.TemporaryDirectory() as tmp:
    local_artifact = mlflow.artifacts.download_artifacts(run_id=nas_run_id, dst_path=tmp)
    model_file = os.path.join(local_artifact, "models", "mobilenetv3_cifar10.pth")
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
model = model.to(device).eval()

# === Evaluation function ===
def evaluate(model): 
    model.eval(); c=0; t=0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            c += (model(x).argmax(1) == y).sum().item()
            t += y.size(0)
    return 100 * c / t

def latency(model):
    model.eval()
    x = torch.randn(1,3,32,32).to(device)
    for _ in range(10): _ = model(x)
    times = [(time.time_ns()) for _ in range(50)]
    for _ in range(50): _ = model(x)
    return sum((time.time_ns() - t)/1e6 for t in times) / 50

acc = evaluate(model)
lat = latency(model)

# === Get model size ===
with tempfile.TemporaryDirectory() as tmp:
    local_artifact = mlflow.artifacts.download_artifacts(run_id=nas_run_id, dst_path=tmp)
    model_file = os.path.join(local_artifact, "models", "mobilenetv3_cifar10.pth")
    size_mb = os.path.getsize(model_file) / (1024**2)

# === Robust writing to Delta Lake (compatible with Databricks Free Edition Unity Catalog) ===
def robust_write_to_delta(spark, data_dict, table_name="model_benchmarks"):
    """
    Automatically detect the current catalog, create a default schema (if it does not exist), and write it to the Delta Lake table.
    """
    # Get the current catalog
    current_catalog = spark.sql("SELECT current_catalog()").collect()[0][0]
    print(f"✅ Current Catalog: {current_catalog}")
    
    # Ensure the default schema exists.
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{current_catalog}`.`default`")
    
    # Construct a complete table name
    full_table_name = f"`{current_catalog}`.`default`.`{table_name}`"
    
    # Create a DataFrame and write it.
    df = spark.createDataFrame([Row(**data_dict)])
    df.write.mode("append").option("mergeSchema", "true").saveAsTable(full_table_name)
    
    print(f"✅ Already written to the table: {full_table_name}")
    return full_table_name

# === MLflow + Delta Lake ===
with mlflow.start_run(run_name="NAS") as run:
    mlflow.log_metrics({"accuracy": acc, "latency_ms": lat, "model_size_mb": size_mb})
    mlflow.set_tag("method", "NAS")

robust_write_to_delta(spark, {
    "method": "NAS",
    "accuracy": acc,
    "latency_ms": lat,
    "model_size_mb": size_mb
})

print(f"✅ NAS | Acc: {acc:.2f}% | Latency: {lat:.2f} ms | Size: {size_mb:.2f} MB")