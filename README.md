# CenterNet

## Instructions for Running the CenterNet App

1. Clone the repository.

2. Install dependencies using:
    - `virtualenv venv --python=python3`
    - `source venv/bin/activate`
    - `pip install -r requirements.txt`
   
3. Download the dataset using:
    - `mkdir data && cd data`
    - `gdown https://drive.google.com/uc?d=1jx-5Pp9rVjPHzv9hmbECUvkWDwlk1PlJ`
    - `unzip pku-autonomous-driving.zip`
    - `rm pku-autonomous-driving.zip && cd ..`
    
4. Launch the app using `streamlit run app.py`.
