# Week 5 — Spark DataFrames: Data Cleaning, Transformation & Aggregation

## Overview

This repository/notebook contains the solution to the Week 5 assignment for the **Data Engineering — Celebal Summer Internship (CSI)**, covering Apache Spark fundamentals and hands-on PySpark DataFrame operations. The assignment moves from conceptual questions (why Spark over MapReduce, in-memory computing, shuffles) to practical, runnable PySpark code covering data cleaning, filtering, aggregation, and building a small end-to-end pipeline.

| | |
|---|---|
| **Student** | Sumit Kumar Singh |
| **Registration / Intern ID** | CT_CSI_DE_1082 |
| **Department** | Data Engineering |
| **College / University** | DIT University |
| **Course** | Data Engineering003 — Celebal Summer Internship (CSI) |
| **Faculty / Mentor** | HR Tarun Sir |
| **Date of Submission** | 20 July 2026 |
| **Subject** | Apache Spark / PySpark |

---

## Environment & Setup

All code was run using **PySpark** in a notebook environment (Google Colab). To reproduce:

```bash
pip install pyspark
```

```python
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Week5").getOrCreate()
```

Every question below assumes `spark` is already initialized, and (where relevant) a small sample DataFrame is created inline so each snippet is runnable on its own, without needing an external file.

---

## Table of Contents

1. [Q1 — Limitations of MapReduce](#q1--limitations-of-traditional-mapreduce)
2. [Q2 — In-Memory Computing for Iterative ML](#q2--in-memory-computing-for-iterative-ml)
3. [Q3 — Removing Duplicate Rows](#q3--removing-duplicate-rows-on-specific-columns)
4. [Q4 — Filter + Group + Average](#q4--filter-group-by-and-average)
5. [Q5 — `.na.drop()` vs `.na.fill()`](#q5--nadrop-vs-nafill)
6. [Q6 — Count per City with Threshold](#q6--count-per-city-above-a-threshold)
7. [Q7 — DataFrame Immutability](#q7--immutability-and-data-cleaning)
8. [Q8 — Range + Categorical Filter](#q8--range-and-categorical-filter)
9. [Q9 — Nulls Before Aggregation](#q9--handling-nulls-before-aggregation)
10. [Q10 — Cast & Rename Column](#q10--casting-and-renaming-a-column)
11. [Q11 — The Shuffle Process](#q11--the-shuffle-process)
12. [Q12 — Multi-condition Null/Empty Filter](#q12--removing-rows-with-null-or-empty-values)
13. [Q13 — Multiple Aggregates with `.agg()`](#q13--multiple-statistics-with-agg)
14. [Q14 — Risks of `inferSchema=True`](#q14--risks-of-inferschematrue)
15. [Q15 — Final Processing Pipeline](#q15--final-processing-pipeline)
16. [Conclusion](#conclusion)
17. [References](#references)

---

## Q1 — Limitations of traditional MapReduce

**Question:** What are the key limitations of traditional MapReduce that make Spark a preferred choice for modern big data processing?

**Answer:**

- **Disk I/O overhead** — MapReduce writes intermediate results to disk after every Map and Reduce phase, causing heavy read/write latency. Spark keeps data in memory (RDDs/DataFrames) across operations.
- **Poor support for iterative algorithms** — ML algorithms (like gradient descent) need multiple passes over data. MapReduce re-reads/re-writes to disk each iteration; Spark caches data in RAM.
- **No interactive/ad-hoc querying** — MapReduce is batch-only. Spark supports interactive shells, SQL, and near real-time streaming.
- **Rigid programming model** — MapReduce only offers Map and Reduce; Spark offers a rich set of transformations (map, filter, join, groupBy, window functions, etc.).
- **Higher latency for small jobs** — JVM startup and scheduling overhead per job is high in MapReduce; Spark's DAG scheduler optimizes execution across the whole pipeline.

---

## Q2 — In-memory computing for iterative ML

**Question:** Explain how Spark uses In-Memory Computing to speed up iterative machine learning algorithms compared to disk-based systems.

**Answer:**

In disk-based systems (like classic MapReduce), each iteration of an algorithm (e.g., k-means, logistic regression) reads input from disk, processes it, writes output back to disk, and the next iteration reads it again. This repeated I/O dominates runtime.

Spark instead loads the dataset into memory once (via `.cache()` or `.persist()`) as an RDD/DataFrame. Each iteration reuses this in-memory copy, only recomputing the transformation logic, not re-reading from disk. This can make iterative ML **10–100x faster**.

```python
df = spark.createDataFrame(data, ["feature", "value"])
df.cache()  # persists in memory after first action

for i in range(iterations):
    # each iteration reuses cached in-memory data
    result = df.groupBy("feature").avg("value")
    result.show()
```

---

## Q3 — Removing duplicate rows on specific columns

**Question:** Write a code snippet to remove all duplicate rows from a DataFrame based on a specific set of columns: `user_id` and `transaction_date`.

**Answer:**

```python
df_clean = df.dropDuplicates(["user_id", "transaction_date"])
df_clean.show()
```

- `dropDuplicates()` without arguments checks all columns; passing a list checks only those specific columns.
- Even if other columns (like `amount`) differ, rows are still considered duplicates if `user_id` and `transaction_date` match.
- Useful when the same event might get logged twice with slightly different metadata.

This keeps only the first occurrence of each unique combination of `user_id` and `transaction_date`, and removes the rest.

---

## Q4 — Filter, group by, and average

**Question:** Given a DataFrame `df_sales`, write a query to filter for rows where the region is `'West'` and then group by `product_category` to find the average `sale_amount`.

**Answer:**

```python
from pyspark.sql.functions import avg

result = (df_sales
    .filter(df_sales.region == "West")
    .groupBy("product_category")
    .agg(avg("sale_amount").alias("avg_sale_amount")))

result.show()
```

- `.filter()` happens first to reduce the dataset before grouping — filtering early is more efficient than filtering after aggregation.
- `.groupBy().agg()` is the standard pattern for category-wise statistics.
- Chaining filter → groupBy → agg keeps the code readable and mirrors SQL's `WHERE` → `GROUP BY` → aggregate order.

---

## Q5 — `.na.drop()` vs `.na.fill()`

**Question:** What is the difference between `.na.drop()` and `.na.fill()`? Provide a code example of filling null values in a `status` column with the string `'Unknown'`.

**Answer:**

`.na.drop()` removes rows that contain null values (by default, any null in any column, though a subset of columns can be specified). It is useful when incomplete rows should be discarded entirely.

`.na.fill()` replaces null values with a specified default value while keeping the row in the dataset. It can accept a single value applied to all compatible columns, or a dictionary mapping specific column names to fill values.

```python
df_filled = df.na.fill({"status": "Unknown"})
df_filled.show()
```

---

## Q6 — Count per city above a threshold

**Question:** Write a query to find the total count of records for each city in a DataFrame, but only for cities where the count is greater than 100.

**Answer:**

```python
from pyspark.sql.functions import col

result = (df.groupBy("city")
    .count()
    .filter(col("count") > 100))

result.show()
```

- `.count()` after `groupBy()` creates a new column literally named `count`.
- The filter on `count > 100` happens **after** aggregation — this is different from filtering rows before grouping.
- This pattern is essentially SQL's `HAVING` clause.

---

## Q7 — Immutability and data cleaning

**Question:** How does the immutability of Spark DataFrames affect how you perform "data cleaning" steps like dropping columns or renaming them?

**Answer:**

Spark DataFrames are immutable, meaning no transformation modifies the original DataFrame object in place. Operations such as `.drop("column")` or `.withColumnRenamed("old", "new")` do not change `df` directly — instead, each one returns a brand-new DataFrame representing the transformed data.

Because of this, cleaning steps must be chained together or explicitly reassigned to a variable (often the same variable name) to persist the change:

```python
df = df.drop("temp_col").withColumnRenamed("old_name", "new_name")
```

This immutability also enables Spark's **lazy evaluation** model: each transformation simply adds a new node to a logical execution plan (DAG), which the Catalyst optimizer can analyze and optimize as a whole before any actual computation happens, rather than executing each step eagerly.

---

## Q8 — Range and categorical filter

**Question:** Write a Spark command to filter a dataset for rows where the age is between 18 and 30 (inclusive) and the subscription is `'Premium'`.

**Answer:**

```python
from pyspark.sql.functions import col

result = df.filter(
    (col("age").between(18, 30)) & (col("subscription") == "Premium")
)
result.show()
```

- `&` is used (not `and`) because Spark Column expressions need element-wise boolean operators.
- Each condition is wrapped in parentheses due to Python operator precedence rules.
- `.between(18, 30)` is used instead of writing two separate comparisons manually.

---

## Q9 — Handling nulls before aggregation

**Question:** When cleaning a dataset, why is it often better to handle null values before performing mathematical aggregations like `sum()` or `avg()`?

**Answer:**

Null values represent missing or unknown data, and mathematical aggregation functions handle them inconsistently depending on context — for example, arithmetic on a null-containing derived column can itself produce null, and averages/sums computed over columns with unhandled nulls can be skewed or misleading (e.g., a wrong denominator when computing an average).

Functions like `sum()` and `avg()` silently ignore nulls in Spark, but if enough rows have nulls, the aggregate becomes misleading — e.g., `avg()` computed on 3 valid rows out of 10 gives a very different result than intended if the 7 nulls actually should have been zeros. Cleaning nulls first (via `.na.fill()` or `.na.drop()`) ensures the aggregation reflects a deliberate, correct decision rather than silently skewing the result.

---

## Q10 — Casting and renaming a column

**Question:** Write the code to revise a column named `raw_timestamp` by casting it to a `TimestampType` and renaming it to `event_time`.

**Answer:**

```python
from pyspark.sql.functions import col
from pyspark.sql.types import TimestampType

df = (df.withColumn("event_time", col("raw_timestamp").cast(TimestampType()))
        .drop("raw_timestamp"))
```

- `.cast(TimestampType())` converts a string into a queryable date/time type, enabling date arithmetic later (extracting year, filtering by date range, etc.).
- `withColumn()` adds the new column; `.drop()` removes the old one — together they simulate a "rename with transformation."
- Always verify with `.printSchema()` that the cast actually worked as expected.

---

## Q11 — The Shuffle process

**Question:** Explain the "Shuffle" process that occurs during a grouping operation. Why is it considered a wide transformation?

**Answer:**

When a grouping operation such as `groupBy` is executed, Spark needs to bring together all records that share the same key so an aggregation can be computed per key. Since the data is distributed across partitions on different executors, rows with the same key may live on different nodes.

The shuffle process redistributes data across the cluster: on the **map side**, each executor writes its data to disk, organized/partitioned by key; then, on the **reduce side**, each output partition reads (pulls) the relevant records for its assigned keys over the network from all the other partitions. This involves disk writes, network transfer, and serialization/deserialization, making shuffles the most expensive operations in Spark.

It is called a **wide transformation** because each output partition can depend on data from many (or all) input partitions — unlike a narrow transformation such as `filter` or `map`, where each output partition depends on only a single input partition. Wide transformations mark **stage boundaries** in Spark's execution DAG.

---

## Q12 — Removing rows with null or empty values

**Question:** Write a code snippet that identifies and removes rows where the `email` column contains null values OR the `username` is an empty string.

**Answer:**

```python
from pyspark.sql.functions import col

df_clean = df.filter(~(col("email").isNull() | (col("username") == "")))
df_clean.show()
```

- `isNull()` checks for missing values; `!= ""` (or `== ""` negated) checks for a specific "empty but not null" case — these are two different data-quality problems.
- Real-world data often has "hidden" missing values represented as empty strings rather than actual nulls.
- Combining both conditions with `|` (then negating the whole expression with `~`) ensures a stricter, more reliable clean dataset.

---

## Q13 — Multiple statistics with `.agg()`

**Question:** How do you use the `.agg()` function to calculate multiple statistics at once, such as the min, max, and mean of the `price` column?

**Answer:**

```python
from pyspark.sql.functions import min, max, mean

result = df.agg(
    min("price").alias("min_price"),
    max("price").alias("max_price"),
    mean("price").alias("mean_price")
)
result.show()
```

Passing multiple aggregate expressions to a single `.agg()` call computes all of them in **one pass** over the data, which is more efficient than calling separate actions for each statistic. `.alias()` is important — without it, output columns get default names like `min(price)`.

---

## Q14 — Risks of `inferSchema=True`

**Question:** In the context of cleaning a dataset, what is the risk of using `inferSchema=true` when your source data contains messy or inconsistent date formats?

**Answer:**

With `inferSchema=True`, Spark samples a portion of the data to guess each column's data type automatically. If a date column contains inconsistent formats (for example, a mix of `DD/MM/YYYY` and `YYYY-MM-DD`, or some rows with invalid/garbage values), Spark's inference logic may fall back to treating the entire column as `StringType` rather than a proper `DateType`/`TimestampType`.

Because inference is based only on a sample rather than the full dataset, problems that exist in the un-sampled rows may not be detected at load time, and can silently surface later as parsing errors, incorrect sort/filter behaviour, or unexpected nulls after casting.

For messy data, it is safer to explicitly define a `StructType` schema and parse/cast date columns deliberately using a known format string, rather than relying on automatic inference.

---

## Q15 — Final processing pipeline

**Question:** Write a final processing pipeline that: (1) filters out duplicates, (2) fills null prices with 0, (3) groups by `store_id` to calculate total revenue.

**Answer:**

```python
from pyspark.sql.functions import sum as _sum

final_df = (df
    .dropDuplicates()
    .na.fill({"price": 0})
    .groupBy("store_id")
    .agg(_sum("price").alias("total_revenue")))

final_df.show()
```

- Order matters: dedupe first (avoids inflating totals from duplicate rows), then fill nulls (avoids `sum()` silently ignoring missing prices), then aggregate.
- This reflects a real-world "cleaning pipeline" pattern: **dedupe → handle missing values → aggregate**.
- Chaining all steps together in one pipeline is more efficient than running them as separate intermediate DataFrame variables.

---

## Conclusion

This assignment covers the fundamentals of Apache Spark and its advantages over traditional MapReduce, including in-memory computation and DAG-based execution. It demonstrates practical PySpark DataFrame operations for data cleaning (handling nulls and duplicates), filtering, schema modification, and aggregation using `groupBy` and `agg()`. Together, these exercises build a complete, working understanding of how to design and implement an end-to-end Spark data-cleaning and aggregation pipeline, from raw, messy data to a clean, analysis-ready dataset.

## References

- Apache Spark Official Documentation — https://spark.apache.org/docs/latest/
- PySpark API Reference — https://spark.apache.org/docs/latest/api/python/
- Celebal Technologies — Data Engineering003 Course Portal (CSI Learning Portal)
