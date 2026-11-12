"""Create an isolated remote environment and fetch one checksum-pinned model."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
import venv

HERE = Path(__file__).resolve().parent
MODEL_NAME = "Cnn14_mAP=0.431.pth"
MODEL_MD5 = "541141fa2ee191a88f24a3219fff024e"
MODEL_URL = "https://zenodo.org/records/3987831/files/Cnn14_mAP%3D0.431.pth?download=1"


def md5(path):
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_model(destination):
    destination = Path(destination)
    if destination.is_file():
        if md5(destination) != MODEL_MD5:
            raise RuntimeError(f"Existing model checksum differs: {destination}; preserve it and provide the correct model.")
        print("VERIFIED_PANNS_CHECKPOINT",destination,flush=True)
        return
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary = destination.with_name(destination.name+".download")
    for attempt in range(1,4):
        try:
            request = urllib.request.Request(MODEL_URL,headers={"User-Agent":"VWB-AudioEvent/20261024"})
            downloaded = 0
            with urllib.request.urlopen(request,timeout=60) as response, temporary.open("wb") as handle:
                while True:
                    block = response.read(8*1024*1024)
                    if not block:
                        break
                    handle.write(block);downloaded += len(block)
                    print("PANNS_DOWNLOAD_MB",round(downloaded/1e6,1),flush=True)
            if md5(temporary) != MODEL_MD5:
                raise RuntimeError("Downloaded checkpoint checksum mismatch.")
            os.replace(temporary,destination)
            return
        except Exception as exc:
            print("PANNS_DOWNLOAD_ATTEMPT_FAILED",attempt,str(exc),flush=True)
            if attempt == 3:
                raise RuntimeError(f"Download failed. Download {MODEL_URL} on your computer, then Xftp it to {destination}. Required MD5: {MODEL_MD5}") from exc
            time.sleep(2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-only",action="store_true")
    args = parser.parse_args()
    if not args.model_only:
        import torch
        import transformers
        print("BASE_ENVIRONMENT",sys.executable,torch.__version__,transformers.__version__,flush=True)
        environment = HERE/".venv-events"
        interpreter = environment/("Scripts/python.exe" if os.name=="nt" else "bin/python")
        if not interpreter.is_file():
            venv.EnvBuilder(with_pip=True,system_site_packages=True).create(environment)
        # Constrain inherited torch: pip must never replace the user's CUDA build.
        constraint = HERE/"logs/setup_constraints.txt"
        constraint.parent.mkdir(exist_ok=True)
        constraint.write_text("torch=="+torch.__version__+"\n",encoding="utf8")
        subprocess.run([str(interpreter),"-m","pip","install","--disable-pip-version-check",
                        "-c",str(constraint),"-r",str(HERE/"requirements-events.txt")],check=True)
        subprocess.run([str(interpreter),"-c",
                        "import torch,transformers,numpy,librosa,torchlibrosa,imageio_ffmpeg;"
                        "print('EVENT_ENV_READY',torch.__version__,transformers.__version__,numpy.__version__,"
                        " 'cuda',torch.cuda.is_available(),'ffmpeg',imageio_ffmpeg.get_ffmpeg_exe())"],check=True)
    fetch_model(HERE/"assets"/MODEL_NAME)
    print("EVENT_SETUP_COMPLETE",flush=True)


if __name__ == "__main__":
    main()
