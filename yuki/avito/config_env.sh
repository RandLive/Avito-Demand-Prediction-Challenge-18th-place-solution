sudo apt-get install -y git
sudo apt-get install -y unzip
sudo apt-get install -y libsm6 libxext6
sudo apt-get install -y libxrender-dev
mkdir projects
cd ~/projects/
mkdir kaggle
cd kaggle/
git clone https://yukimilk@bitbucket.org/yukimilk/avito.git
cd avito/
cd model/
mkdir gensim
cd ../src/
mkdir logs
cd ~/projects/kaggle/avito/
sudo apt-get install -y python3-pip
pip3 install -r requirements.txt
cd ~/projects/kaggle/
git clone https://github.com/facebookresearch/fastText.git
cd fastText
pip3 install .
cd ~/projects/kaggle/avito/
cd ~
wget https://github.com/facebookresearch/fastText/archive/v0.1.0.zip
unzip v0.1.0.zip
cd fastText-0.1.0
make
cd ~
echo 'export PATH="$HOME/fastText-0.1.0:$PATH"' >> ~/.bashrc
source ~/.bashrc

sudo apt-get -y install tesseract-ocr
pip3 install pytesseract
sudo apt-get -y install python-opencv
sudo apt-get -y install libopencv-dev
pip3 install opencv-contrib-python
pip3 install xgboost
