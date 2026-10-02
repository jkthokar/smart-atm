"""Download the OpenCV face-detector files into ./models (run once)."""

import hashlib
import urllib.request

from smart_atm.config import MODELS_DIR

FILES = {
    "deploy.prototxt": (
        "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
        "dcd661dc48fc9de0a341db1f666a2164ea63a67265c7f779bc12d6b3f2fa67e9",
    ),
    "res10_300x300_ssd_iter_140000.caffemodel": (
        "https://raw.githubusercontent.com/opencv/opencv_3rdparty/"
        "dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
        "2a56a11a57a4a295956b0660b4a3d76bbdca2206c4961cea8efe7d95c7cb2f2d",
    ),
}


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, (url, digest) in FILES.items():
        dest = MODELS_DIR / name
        if dest.exists() and sha256(dest) == digest:
            print(f"[ok]   {name} already downloaded")
            continue
        print(f"[get]  {name}")
        urllib.request.urlretrieve(url, dest)
        if sha256(dest) != digest:
            dest.unlink()
            raise SystemExit(f"Checksum mismatch for {name}; please try again.")
    print("Face detector ready in", MODELS_DIR)


if __name__ == "__main__":
    main()
