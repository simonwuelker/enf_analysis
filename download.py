import requests
import zipfile
import os

url = "https://data.nationalgrideso.com/system/system-frequency-data/datapackage.json"
download_to = "tmp/"
extract_to = "data/"

resources = requests.get(url).json()["resources"]

print(f"Found {len(resources)} datasets")

for ix, resource in enumerate(resources):
    path = os.path.join(download_to, f"{resource['id']}.zip");
    print(f"[{ix:02}/{len(resources)}] Saving '{resource['title']:<40}' to {path}")

    response = requests.get(resource["path"])
    with open(path, "wb") as f:
        f.write(response.content)

for ix, filename in enumerate(os.listdir(download_to)):
    path = os.path.join(download_to, filename)
    print(f"[{ix:02}/{len(resources)}] Inflating '{path}'")

    try:
        with zipfile.ZipFile(path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
    except:
        print(f"Failed to unpack {path}")


