# Metasploit VM automation

## Probing the network

### Prerequisites

* Virtual wall IPv4 internet access (NAT)

```bash
sudo route del default gw 10.2.15.254 && sudo route add default gw 10.2.15.253
sudo route add -net 10.11.0.0 netmask 255.255.0.0 gw 10.2.15.254
sudo route add -net 10.2.32.0 netmask 255.255.240.0 gw 10.2.15.254
```

* apt-get install

```bash
sudo apt-get update
sudo apt-get install -y ruby git nmap python-pip
sudo apt-get install -y subversion
sudo apt-get install -y build-essential ruby-dev libpcap-dev
```

* Metasploit (no GUI installer !)

```bash
curl https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb > msfinstall && chmod 755 msfinstall &&  ./msfinstall
```

* Start Metasploit RPC server on localhost

`msfrpcd -U msf -P msfpass -a 127.0.0.1 -p 55552`

Currently has issues when trying to connect with apt2, workaround

```
msfconsole
load msgrpc User=msf Password=msfpass NetworkPort=55552
```

* APT2

```bash
git clone https://github.com/MooseDojo/apt2.git
sudo pip install cython
sudo pip install unqlite
sudo pip install python-nmap
sudo pip install msgpack-python
sudo pip install scapy
sudo pip install whois
sudo pip install ftputil
sudo pip install yattag
sudo pip install netaddr
sudo pip install shodan
sudo pip install pysmb

```

## Virtual wall shared location

in elke VM __/groups/wall2-ilabt-iminds-be/cybersecurity/__