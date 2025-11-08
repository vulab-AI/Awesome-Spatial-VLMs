import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from google import genai
from google.genai.types import CreateBatchJobConfig, JobState, HttpOptions
from google.cloud import storage

# init genai client
client = genai.Client(http_options=HttpOptions(api_version="v1"))

def list_gcs_dir(bucket_name, prefix="jsonls/"):
    """List all files in a GCS directory"""
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=prefix)

    files = []
    for blob in blobs:
        if not blob.name.endswith("/"):  # skip "folder" placeholders
            gcs_path = f"gs://{bucket_name}/{blob.name}"
            files.append(gcs_path)
    return files


def run_batch_job(src_file, output_uri="your-output-bucket/output/"):
    """submit a batch job to Gemini 2.5"""
    thread_name = threading.current_thread().name
    print(f"[{thread_name}] Submitting job for {src_file}")

    job = client.batches.create(
        model="gemini-2.5-pro",
        src=src_file,
        config=CreateBatchJobConfig(dest=output_uri),
    )

    print(f"[{thread_name}] Job created: {job.name}, state={job.state}")

    # wait for job completion
    completed_states = {
        JobState.JOB_STATE_SUCCEEDED,
        JobState.JOB_STATE_FAILED,
        JobState.JOB_STATE_CANCELLED,
        JobState.JOB_STATE_PAUSED,
    }

    while job.state not in completed_states:
        time.sleep(30)
        job = client.batches.get(name=job.name)
        print(f"[{thread_name}] Job {job.name} state: {job.state}")

    print(f"[{thread_name}] Job {job.name} finished with state: {job.state}")
    return job.name, job.state


if __name__ == "__main__":
    #
    bucket_name = "your-bucket-name"
    files = list_gcs_dir(bucket_name, prefix="jsonls/")

    print(f"Found {len(files)} files to process")

    # submit jobs in parallel
    with ThreadPoolExecutor(max_workers=len(files)) as executor:  # can be 10
        futures = {executor.submit(run_batch_job, f): f for f in files}

        for future in as_completed(futures):
            src_file = futures[future]
            try:
                job_name, job_state = future.result()
                print(f"[Main] File {src_file} → Job {job_name} finished with {job_state}")
            except Exception as e:
                print(f"[Main] File {src_file} → Error: {e}")
