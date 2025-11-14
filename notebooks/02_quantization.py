# Databricks notebook source
# MAGIC %md
# MAGIC ## 02. QAT: Quantization-Aware Training (FP32 Evaluation Only)

# COMMAND ----------
import torch, torchvision, mlflow, time, os, tempfile
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pyspark.sql import SparkSession, Row
from mlflow import MlflowClient

# === Clean up residual run files ===
while mlflow.active_run():
    mlflow.end_run()

# === Set up a unified experiment ===
mlflow.set_experiment("/Shared/Arm_Compression")
exp = mlflow.get_experiment_by_name("/Shared/Arm_Compression")
exp_id = exp.experiment_id
device = torch.device("cpu")
spark = SparkSession.builder.getOrCreate()

# === Hard-coded valid Distillation Run ID ===
distill_run_id = "0403e79be845489e914a04a53fca67a8" 
print(f"✅ Using the distilled model: Run ID={distill_run_id}")

# === Loading the distilled model（FP32）===
with tempfile.TemporaryDirectory() as tmp:
    local_path = mlflow.artifacts.download_artifacts(
        run_id=distill_run_id,
        artifact_path="models",
        dst_path=tmp
    )
    model_file = os.path.join(local_path, "student_distilled.pth")
    model = torchvision.models.mobilenet_v2(num_classes=10)
    model.load_state_dict(torch.load(model_file, map_location="cpu"))
    model = model.to(device)

# === Enable QAT configuration (simulate quantization only during training) ===
model.train()
model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
torch.quantization.prepare_qat(model, inplace=True)

# === Data and fine-tuning ===
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
trainset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

# === QAT fine-tuning（3 epoch）===
with mlflow.start_run(run_name="QAT") as run:
    for epoch in range(3):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    
    # ✅ Save the model after QAT fine-tuning (still FP32, but with added quantization information)
    with tempfile.TemporaryDirectory() as tmp:
        qat_model_path = os.path.join(tmp, "qat_model_fp32.pth")
        torch.save(model.state_dict(), qat_model_path)
        mlflow.log_artifact(qat_model_path, "models")

# === Evaluation: FP32 model fine-tuned using QAT (Safe!) ===
testset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

def evaluate(model):
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return 100 * correct / total

def measure_latency(model, trials=50):
    model.eval()
    dummy_input = torch.randn(1, 3, 32, 32).to(device)
    for _ in range(10): _ = model(dummy_input)
    times = []
    for _ in range(trials):
        start = time.time()
        _ = model(dummy_input)
        times.append((time.time() - start) * 1000)  # ms
    return sum(times) / len(times)

accuracy = evaluate(model)
latency = measure_latency(model)

# === Getting the model size ===
with tempfile.TemporaryDirectory() as tmp:
    art = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, dst_path=tmp)
    model_file = os.path.join(art, "models", "qat_model_fp32.pth")
    model_size_mb = os.path.getsize(model_file) / (1024 ** 2)

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
mlflow.log_metrics({"accuracy": accuracy, "latency_ms": latency, "model_size_mb": model_size_mb})
mlflow.set_tag("method", "QAT")

robust_write_to_delta(spark, {
    "method": "QAT",
    "accuracy": accuracy,
    "latency_ms": latency,
    "model_size_mb": model_size_mb
})

print(f"✅ QAT completed | Acc: {accuracy:.2f}% | Latency: {latency:.2f} ms | Size: {model_size_mb:.2f} MB")