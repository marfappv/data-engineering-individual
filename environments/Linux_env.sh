sudo apt-get update
python -m pip3 install requests
python -m pip3 install fastparquet
python -m pip3 install -q findspark
python -m pip3 install pyspark
pyspark --num-executors 2
python -m pip3 install pyarrow

sudo apt-get update install --temurin8
sudo apt-get update install --cask android-sdk

curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg AWSCLIV2.pkg -target /
curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
sudo installer -pkg ./AWSCLIV2.pkg -target /