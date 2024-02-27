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

pip install poetry
poetry