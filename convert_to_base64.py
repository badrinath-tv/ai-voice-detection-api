import base64

file_path = "test.mp3"  # change this

with open(file_path, "rb") as f:
    encoded = base64.b64encode(f.read()).decode("utf-8")

print(encoded)
