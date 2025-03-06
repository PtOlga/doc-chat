import requests
import time

base_url = "https://rulga-doc-chat.hf.space"
max_retries = 10  # Максимальное количество попыток
retry_delay = 30  # Задержка между попытками в секундах

def wait_for_service():
    print("Waiting for the service to start...")
    for attempt in range(max_retries):
        try:
            response = requests.get(base_url)
            if response.status_code == 200 and "Could not parse JSON" not in response.text:
                print(f"Service is ready after {attempt + 1} attempts!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        print(f"Attempt {attempt + 1}/{max_retries}. Service is still starting. Waiting {retry_delay} seconds...")
        time.sleep(retry_delay)
    
    return False

if wait_for_service():
    # Запуск создания базы знаний
    print("\nSending rebuild request...")
    rebuild_url = f"{base_url}/rebuild-kb"
    response = requests.post(rebuild_url, params={"force": True})
    print(f"Status code: {response.status_code}")
    print(f"Response: {response.text}")

    # Проверка статуса
    print("\nChecking status...")
    status_url = f"{base_url}/kb-status"
    status = requests.get(status_url)
    print(f"Status code: {status.status_code}")
    print(f"Status: {status.text}")
else:
    print("Service failed to start after maximum retries")
