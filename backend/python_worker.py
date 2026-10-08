import sys
import json
from pathlib import Path
import numpy as np

# Make the project's src folder importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ml.predict import predict_sign


def main():
    # One JSON request per line; one JSON response per line
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if not isinstance(request, dict) or "keypoints" not in request:
                raise ValueError("Request must be a JSON object containing a 'keypoints' field.")

            keypoints = np.array(request["keypoints"])
            result = predict_sign(keypoints)
            print(json.dumps({"success": True, "prediction": result}), flush=True)
        except Exception as exc:
            print(json.dumps({"success": False, "error": str(exc)}), flush=True)


if __name__ == "__main__":
    main()