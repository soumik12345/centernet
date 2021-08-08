virtualenv venv --python=python3
source venv/bin/activate
pip install -r requirements.txt
pip install torch==1.9.0+cpu torchvision==0.10.0+cpu torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
