pip install gdown
mkdir data && cd data
gdown https://drive.google.com/uc?id=1jx-5Pp9rVjPHzv9hmbECUvkWDwlk1PlJ
echo "Unzipping Dataset..."
unzip -q pku-autonomous-driving.zip
echo "Done Unzipping"
rm pku-autonomous-driving.zip && cd ..
