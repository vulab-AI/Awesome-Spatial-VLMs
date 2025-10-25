import json
import os


from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
import os

# ========== init ==========
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CLIENT_SECRET_FILE = "client_secret.json"
TOKEN_FILE = "token.json"


# ========== login ==========
def get_drive_service():
    creds = None
    if os.path.exists(TOKEN_FILE):
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(CLIENT_SECRET_FILE, SCOPES)
        try:
            # local environment with browser
            creds = flow.run_local_server(port=8080)
        except Exception:
            # remote environment / no browser
            creds = flow.run_console()
        with open(TOKEN_FILE, "w") as token:
            token.write(creds.to_json())
    return build("drive", "v3", credentials=creds)

service = get_drive_service()

# ========== path to file_id ==========
def get_file_id_from_path(service, path: str) -> str:
    """
    translate (images/CV_Bench/0_image_0.png) to file_id
    """
    parts = path.strip("/").split("/")  # ["images", "CV_Bench", "0_image_0.png"]

    parent = "root"  #  root
    file_id = None

    for name in parts:
        query = f"name = '{name}' and '{parent}' in parents and trashed = false"
        results = service.files().list(q=query, fields="files(id, name)").execute()
        files = results.get("files", [])

        if not files:
            raise FileNotFoundError(f"'{name}' not found under parent {parent}")

        file_id = files[0]["id"]
        parent = file_id  # next parent

    return file_id


# ========= path to direct link ==========
def to_drive_url(path: str) -> str:
    file_id = get_file_id_from_path(service, path)
    direct_link = f"https://drive.google.com/uc?export=download&id={file_id}"
    return direct_link

# ========== main ==========

json_folder = "jsons"
output_folder = "jsonls"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for json_file in os.listdir(json_folder):
    if json_file.endswith(".json"):
        with open(os.path.join(json_folder, json_file), 'r') as f:
            data = json.load(f)

        jsonl_path = os.path.join(output_folder, json_file.replace(".json", ".jsonl"))
        with open(jsonl_path, "w", encoding="utf-8") as out:
            for idx, item in enumerate(data, start=1):
                # system message
                system_msg = {
                    "role": "system",
                    "content": "Answer the question with only the choice letter in the format <answer></answer>. No more explanations."
                }

                # user message: contains images and text
                user_content = []
                if "image_uris" in item:
                    for img_path in item["image_uris"]:
                        user_content.append({
                            "type": "image_url",
                            "image_url": {"url": to_drive_url(img_path)}
                        })

                if "prompt" in item:
                    user_content.append({
                        "type": "text",
                        "text": item["prompt"]
                    })

                user_msg = {"role": "user", "content": user_content}

                #modified for batch jsonl here and some parameters may differ for different models
                row = {
                    "custom_id": f"request-{item.get('id', idx)}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": "gpt-4o",
                        "messages": [system_msg, user_msg],
                        "max_tokens": 1000
                    }
                }

                out.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"✅ Converted {json_file} -> {jsonl_path}")
