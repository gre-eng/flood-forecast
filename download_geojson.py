import requests

url = "https://simplemaps.com/static/svg/country/ng/admin1/ng.json"
response = requests.get(url, timeout=30)
response.raise_for_status()

with open("data/nigeria_states.geojson", "wb") as f:
    f.write(response.content)

print(f"Downloaded {len(response.content)} bytes")
print("Saved to data/nigeria_states.geojson
