# Metasploit VM automation

## Probing the network

### Prerequisites

* apt-get install

```bash
sudo apt-get install ruby libopenssl-ruby libyaml-ruby libdl-ruby libiconv-ruby libreadlineruby irb ri rubygems
sudo apt-get install subversion
sudo apt-get install build-essential ruby-dev libpcap-dev nmap python-pip
```

* Metasploit

```bash
wget https://downloads.metasploit.com/data/releases/metasploit-latest-linux-x64-installer.run && wget https://downloads.metasploit.com/data/releases/metasploit-latest-linux-x64-installer.run.sha1 && echo $(cat metasploit-latest-linux-x64-installer.run.sha1)'  'metasploit-latest-linux-x64-installer.run > metasploit-latest-linux-x64-installer.run.sha1 && shasum -c metasploit-latest-linux-x64-installer.run.sha1 && chmod +x ./metasploit-latest-linux-x64-installer.run && sudo ./metasploit-latest-linux-x64-installer.run
```

* Create RPC server for metasploit

`msfrpc msfrpc -U msf -P msfpass -a 127.0.0.1:55552`

* APT2

```bash
git clone https://github.com/MooseDojo/apt2.git
sudo pip install cython
sudo pip install unqlite
sudo pip install python-nmap
sudo pip install msgpack-python
```

* Metasploit RPC server on localhost

`msfrpcd -U msf -P msfpass -a 127.0.0.1 -p 55552`
