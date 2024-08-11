import gdown

# Google Drive file ID from the provided link
file_id = '1DMpGvBIwL3ND9A8GXxb1GovM01435oYQ'
# Construct the direct download URL
url = f'https://drive.google.com/uc?id={file_id}'
# Output file name
output = 'chexnet_weights.pth'

# Download the file
gdown.download(url, output, quiet=False)

# Check if the file is downloaded successfully
import os
if os.path.exists(output):
    print(f"CheXNet weights downloaded successfully: {output}")
else:
    print("Download failed.")
