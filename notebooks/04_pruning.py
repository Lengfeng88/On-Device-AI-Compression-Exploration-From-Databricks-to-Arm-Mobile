# Databricks notebook source
# MAGIC %md
# MAGIC ## 04. Pruning + Fine-tuning (FP32 Only)

# COMMAND ----------
import torch, torchvision, mlflow, time, os, tempfile
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pyspark.sql import SparkSession, Row
import torch.nn.utils.prune as prune

# === Setting up the experiment and loading the distillation model ===
mlflow.set_experiment("/Shared/Arm_Compression")
device = torch.device("cpu")
spark = SparkSession.builder.getOrCreate()

# Hardcode the distillation Run ID (replace with your valid ID)
distill_run_id = "0403e79be845489e914a04a53fca67a8"
print(f"✅ Loading the distilled model: Run ID={distill_run_id}")

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

# === Structured pruning (L1 Unstructured, prune 30% weight) ===
model.train()
for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.3)
        prune.remove(module, 'weight')  

# === Fine-tuning recovery accuracy ===
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,)*3, (0.5,)*3)])
trainset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

with mlflow.start_run(run_name="Pruning") as run:
    for epoch in range(2):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
    
    # Saving the pruned model (FP32)
    with tempfile.TemporaryDirectory() as tmp:
        pruned_path = os.path.join(tmp, "pruned_model_fp32.pth")
        torch.save(model.state_dict(), pruned_path)
        mlflow.log_artifact(pruned_path, "models")

# === Assessment (FP32, Safety) ===
testset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

def evaluate(model):
    model.eval(); correct = 0; total = 0
    with torch.no_grad():
        for x, y in testloader:
            x, y = x.to(device), y.to(device)
            _, pred = torch.max(model(x).data, 1)
            total += y.size(0); correct += (pred == y).sum().item()
    return 100 * correct / total

def measure_latency(model, trials=50):
    model.eval()
    dummy = torch.randn(1, 3, 32, 32).to(device)
    for _ in range(10): _ = model(dummy)
    times = [(time.time_ns() - time.time_ns()) for _ in range(trials)]
    for _ in range(trials):
        start = time.time()
        _ = model(dummy)
        times.append((time.time() - start) * 1000)
    return sum(times[-trials:]) / trials

accuracy = evaluate(model)
latency = measure_latency(model)

# === Getting the model size ===
with tempfile.TemporaryDirectory() as tmp:
    art = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, dst_path=tmp)
    model_file = os.path.join(art, "models", "pruned_model_fp32.pth")
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
mlflow.set_tag("method", "Pruning")

robust_write_to_delta(spark, {
    "method": "Pruning",
    "accuracy": accuracy,
    "latency_ms": latency,
    "model_size_mb": model_size_mb
})

print(f"✅ Pruning completed | Acc: {accuracy:.2f}% | Latency: {latency:.2f} ms | Size: {model_size_mb:.2f} MB")