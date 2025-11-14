# Databricks notebook source
# MAGIC %md
# MAGIC ## 1. Distillation

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
trainset = torchvision.datasets.CIFAR10("./data", train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10("./data", train=False, download=True, transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
testloader = DataLoader(testset, batch_size=64, shuffle=False)

# === Automatically load teacher models from MLflow ===
client = MlflowClient()
exp = mlflow.get_experiment_by_name("/Shared/Arm_Compression")
exp_id = exp.experiment_id
teacher_runs = client.search_runs(experiment_ids=[exp.experiment_id], filter_string='tags.role = "teacher"')
if not teacher_runs:
    raise RuntimeError("❌ Teacher model not found! Please run 00_train_teacher.py first and ensure that it is used. mlflow.set_tag('role', 'teacher')")

teacher = mlflow.pytorch.load_model(f"runs:/{teacher_runs[0].info.run_id}/teacher_model", map_location="cpu")
teacher = teacher.to(device).eval()

# === Student models ===
student = torchvision.models.mobilenet_v2(num_classes=10)

# === Distillation training ===
optimizer = torch.optim.Adam(student.parameters(), lr=0.001)
criterion_ce = torch.nn.CrossEntropyLoss()
criterion_kl = torch.nn.KLDivLoss(reduction="batchmean")
temp, alpha = 4, 0.7

with mlflow.start_run(run_name="Distillation") as run:
    student.train()
    for epoch in range(5):
        for x, y in trainloader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                t_logits = teacher(x)
            s_logits = student(x)
            loss_kl = criterion_kl(
                torch.log_softmax(s_logits / temp, dim=1),
                torch.softmax(t_logits / temp, dim=1)
            ) * (temp ** 2)
            loss_ce = criterion_ce(s_logits, y)
            loss = alpha * loss_kl + (1 - alpha) * loss_ce
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    
    # Save student models to MLflow
    with tempfile.TemporaryDirectory() as tmp:
        model_path = os.path.join(tmp, "student_distilled.pth")
        torch.save(student.state_dict(), model_path)
        mlflow.log_artifact(model_path, "models")

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

acc = evaluate(student)
lat = latency(student)

# === Get model size ===
with tempfile.TemporaryDirectory() as tmp:
    local_artifact = mlflow.artifacts.download_artifacts(run_id=run.info.run_id, dst_path=tmp)
    model_file = os.path.join(local_artifact, "models", "student_distilled.pth")
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
mlflow.log_metrics({"accuracy": acc, "latency_ms": lat, "model_size_mb": size_mb})
mlflow.set_tag("method", "Distillation")

# Use robust write functions
robust_write_to_delta(spark, {
    "method": "Distillation",
    "accuracy": acc,
    "latency_ms": lat,
    "model_size_mb": size_mb
})

print(f"✅ Distillation | Acc: {acc:.2f}% | Latency: {lat:.2f} ms | Size: {size_mb:.2f} MB")