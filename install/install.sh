#  sudo apt install --no-install-recommends nvidia-driver-510 nvidia-dkms-510

sudo apt install --no-install-recommends nvidia-driver-525 nvidia-dkms-525
sudo apt install --no-install-recommends jq htop libc-dev g++ graphviz

curl micro.mamba.pm/install.sh | bash
micromamba create -f ./env.yaml

# micromamba install -f ./env.yaml
mamba env update --file env.yaml


#python3-pip htop python3-dev

# pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
sudo apt-get install gnupg
curl -fsSL https://pgp.mongodb.com/server-6.0.asc | \
   sudo gpg -o /usr/share/keyrings/mongodb-server-6.0.gpg \
   --dearmor
echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-6.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/6.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-6.0.list
sudo apt-get update
sudo apt-get install -y mongodb-org
sudo systemctl enable mongod
sudo systemctl start mongod
mongosh


# pip install tabnine-jupyterlab
pip install "jedi<0.18" jupyterlab-lsp 'python-lsp-server[all]'


optuna-dashboard --host 0.0.0.0 --port 8069 sqlite:///audio-ml/lib/tune/opt.db

find ./ -name "cuda.h"
sudo mkdir  /usr/local/cuda
sudo mkdir  /usr/local/cuda/include
 sudo cp ./envs/pytorch-env/envs/pytorch-env/lib/python3.10/site-packages/triton/third_party/cuda/include/cuda.h /usr/local/cuda/include
# sudo apt-get install nvidia-cuda-dev

sudo apt-get install --no-install-recommends nvidia-cuda-toolkit g++

#  pip3 install torchdata torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116


mamba install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia


dask-scheduler
docker run -d --restart always --network host cr.yandex/crp2897qtopgu6vmmabs/worker:latest


 pip3 install librosa optuna disklist dask dask-kubernetes
 pip3 install onnxruntime onnx nerus mosestokenizer shap corus navec razdel  slovnet jupyterlab numpy scipy pandas python-dotenv pydot tqdm ipywidgets==7.7.2 matplotlib

 pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformersrc
 pip3 install markupsafe==2.0.1

 pip install optuna-dashboard
pip install optuna-fast-fanova gunicorn
!opencorpora download
 sudo sudo parted /dev/vdb
 #  mklabel gpt
 # sudo parted -a optimal /dev/vdb mkpart primary 0% 100%
#  mkpart primary 0% 99%
# sudo mkfs.ext4 /dev/vdb1
# sudo mount /dev/vdb1 ~/proj/cache



# wget https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2
# navec


git clone https://github.com/iliadmitriev/DAWG
git checkout 'refs/remotes/origin/fix_py_3_10'
pip3 install cython
pip3 install .

git clone https://github.com/pymorphy2/pymorphy2-dicts
cd pymorphy2-dicts
 pip3 install -r ./requirements-build.txt
./update.py ru download
./update.py ru compile
./update.py ru package
./update.py ru cleanup
cd pymorphy2-dicts-ru/
pip3 install .
