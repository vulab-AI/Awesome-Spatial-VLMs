from openai import OpenAI
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

api_key ="your-openai-api-key"

client = OpenAI(api_key=api_key)

# 配置文件夹 !!!!
json_folders = "gpt5_jsonls"
output_folders = "gpt5_results"

## 创建输出文件夹
os.makedirs(output_folders, exist_ok=True)

def process_jsonl(jsonlfile):
    """处理单个 JSONL 文件：上传 -> 提交 -> 轮询 -> 保存结果"""
    try:
        print(f"🚀 Processing file: {jsonlfile}", flush=True)

        # Step 1. 上传
        batch_input_file = client.files.create(
            file=open(os.path.join(json_folders, jsonlfile), "rb"),
            purpose="batch"
        )
        print(f"✅ Uploaded {jsonlfile}: {batch_input_file.id}", flush=True)

        # Step 2. 创建 batch
        job = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
                "description": "spatial eval",
                "version": "1.0.0",
                "source_file": jsonlfile
            }
        )
        print(f"📌 Job created for {jsonlfile}: {job.id}", flush=True)

        # Step 3. 轮询
        while True:
            batch = client.batches.retrieve(job.id)
            print(f"⏳ {jsonlfile} status: {batch.status}", flush=True)

            if batch.status in ["completed", "failed", "expired", "cancelled"]:
                break
            time.sleep(60)

        # Step 4. 保存结果
        if batch.status == "completed" and batch.output_file_id:
            result_content = client.files.content(batch.output_file_id).text
            output_path = os.path.join(
                output_folders, jsonlfile.replace(".jsonl", "_result.jsonl")
            )
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result_content)
            print(f"✅ Results saved: {output_path}", flush=True)
        else:
            print(f"❌ {jsonlfile} failed with status={batch.status}", flush=True)

    except Exception as e:
        print(f"🔥 Error processing {jsonlfile}: {e}", flush=True)


if __name__ == "__main__":
    files = [f for f in os.listdir(json_folders) if f.endswith(".jsonl")]

    # 限制线程数，避免过多并发请求
    max_workers = min(6, len(files))  # 可以改大或改小
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_jsonl, f) for f in files]
        for future in as_completed(futures):
            future.result()  # 抛出异常时立刻显示
1