# COMMAND ----------
import pandas as pd

# Reading data from Delta Lake
df = spark.table("model_benchmarks").toPandas()
df = df.round(2)
# Use the built-in visualization (Databricks supports)
display(df)