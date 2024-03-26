## Environment Settings
- 해당 작업은 /home/{user_name}에서 진행해주세요

### Install Pyenv
```bash
sudo su - # 아래 명령어 해보고 안되면 실행
sudo apt-get install -y make build-essential libssl-dev liblzma-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev
exit # sudo 빠져나오기

git clone https://github.com/pyenv/pyenv.git ~/.pyenv

echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
source ~/.bashrc

pyenv install 3.10.0
```

### Install Poetry
```bash
chmod -R 777 level2-3-cv-finalproject-cv-13/

sudo su -
apt install python3-pip
exit
pip install testresources poetry
echo 'export PATH=$PATH:~/.local/bin' >> ~/.bashrc
poetry
```

## To deploy...

### Install Docker and Docker-compose

```
# Add Docker's official GPG key:
sudo apt-get update
sudo apt-get install ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add the repository to Apt sources:
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
```

```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

```
sudo curl -L "https://github.com/docker/compose/releases/download/v2.19.1/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose

sudo chmod +x /usr/local/bin/docker-compose

docker-compose --version
```

### Install NVIDIA-Container-Toolkit

```
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

```
sudo apt-get update

sudo apt-get install -y nvidia-container-toolkit
```

```
sudo cat /etc/nvidia-container-runtime/config.toml
```

** 여기서 [nvidia-container-runtime]에 default-runtime이 없으면
** sudo vim /etc/nvidia-container-runtime/config.toml
   => [nvidia-container-runtime] 밑에 default-runtime = "nvidia" 추가해주세요
** 변경했다면 sudo systemctl restart docker