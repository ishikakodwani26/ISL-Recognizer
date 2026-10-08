import os
import glob
import pickle
import requests
import json

def test_prediction():
    # Find a sample .pkl file in the INCLUDE dataset
    sample_pattern = os.path.join("data", "raw", "INCLUDE", "**", "*.pkl")
    pkl_files = glob.glob(sample_pattern, recursive=True)
    
    if not pkl_files:
        print("Error: No .pkl sample found in data/raw/INCLUDE")
        return
        
    sample_path = pkl_files[0]
    print(f"Loading sample file: {sample_path}")
    
    with open(sample_path, "rb") as f:
        data = pickle.load(f)
        
    if isinstance(data, dict) and "keypoints" in data:
        keypoints = data["keypoints"]
    else:
        keypoints = data

    # Ensure keypoints are in list format for JSON payload
    if hasattr(keypoints, "tolist"):
        keypoints_list = keypoints.tolist()
    else:
        keypoints_list = keypoints

    print(f"Keypoints shape/length: {len(keypoints_list)} frames")
    
    url = "http://localhost:5000/api/predict"
    headers = {"Content-Type": "application/json"}
    payload = {"keypoints": keypoints_list}
    
    print(f"Sending POST request to {url}...")
    response = requests.post(url, json=payload, headers=headers)
    
    print("\n--- Response from Server ---")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    test_prediction()
