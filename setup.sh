#!/bin/bash
pip3 install --upgrade Pillow
pip3 install flask tqdm torch tiktoken transformers peft accelerate torchvision torchaudio vllm auto-gptq optimum
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
sed -i "/^PATH='\/opt\/conda\/bin:\/usr\/local\/nvidia\/bin:\/usr\/local\/cuda\/bin:\/usr\/local\/sbin:\/usr\/local\/bin:\/usr\/sbin:\/usr\/bin:\/sbin:\/bin'$/s/^/#/" ~/.bashrc
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.38.0/install.sh | bash
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
nvm install 14.21.3
nvm alias default 14.21.3
source ~/.bashrc
npm install -g pm2