import requests

url = "https://rulga-doc-chat.hf.space/rebuild-kb"
response = requests.post(url, params={"force": True})
print(response.json())

# Проверка статуса
status = requests.get("https://rulga-doc-chat.hf.space/kb-status")
print(status.json())