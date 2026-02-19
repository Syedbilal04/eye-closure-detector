from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core import base_options


def main():
    print("FaceLandmarker", vision.FaceLandmarker)
    print("FaceLandmarkerOptions", vision.FaceLandmarkerOptions)
    print("RunningMode", vision.RunningMode)
    print("BaseOptions", base_options.BaseOptions)
    print("FaceLandmarkerOptions_fields", getattr(vision.FaceLandmarkerOptions, "__annotations__", None))


if __name__ == "__main__":
    main()

