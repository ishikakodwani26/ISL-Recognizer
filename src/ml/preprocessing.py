import numpy as np

MAX_FRAMES = 92

SELECTED_KEYPOINTS = list(range(11, 23)) + list(range(33, 75))


def preprocess_keypoints(keypoints):
    """
    Convert raw keypoints of shape (frames, 75, 3)
    into the (92, 162) format expected by the BiLSTM.
    """

    keypoints = keypoints.astype(np.float32)

    # 1. Shoulder normalization
    left_shoulder = keypoints[:, 11, :]
    right_shoulder = keypoints[:, 12, :]

    shoulder_center = (
        left_shoulder + right_shoulder
    ) / 2

    shoulder_distance = np.linalg.norm(
        left_shoulder - right_shoulder,
        axis=1,
        keepdims=True
    )

    # Avoid division by zero
    shoulder_distance = np.where(
        shoulder_distance < 1e-6,
        1.0,
        shoulder_distance
    )

    keypoints = (
        keypoints - shoulder_center[:, None, :]
    ) / shoulder_distance[:, None, :]

    # 2. Select upper body + hands
    keypoints = keypoints[:, SELECTED_KEYPOINTS, :]

    # (frames, 54, 3) -> (frames, 162)
    keypoints = keypoints.reshape(
        keypoints.shape[0], -1
    )

    # 3. Pad / truncate to 92 frames
    if keypoints.shape[0] < MAX_FRAMES:

        padding = np.zeros(
            (
                MAX_FRAMES - keypoints.shape[0],
                162
            ),
            dtype=np.float32
        )

        keypoints = np.vstack([
            keypoints,
            padding
        ])

    else:
        keypoints = keypoints[:MAX_FRAMES]

    return keypoints