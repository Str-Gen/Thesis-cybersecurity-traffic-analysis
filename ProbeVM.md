# Metasploit VM automation

## Prerequisites

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

* Metasploit (no graphical installer ! [nightly](https://github.com/rapid7/metasploit-framework/wiki/Nightly-Installers))

```bash
curl https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb > msfinstall && chmod 755 msfinstall &&  ./msfinstall
```

## Starting APT2

* Start Metasploit RPC server on localhost

`sudo msfrpcd -U msf -P msfpass -p 55552 -f "/api/" -a 127.0.0.1 -S &`

In case of issues when trying to connect with apt2, workaround:

```bash
msfconsole
load msgrpc User=msf Password=msfpass NetworkPort=55552
```

* APT2 basic

pip dependencies can't be listed in one line if one depends on the other

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
sudo pip install whois
sudo pip install ipwhois
sudo pip install pyasn1
sudo pip install impacket

```

* APT advanced (optional apt-get installs)

`sudo apt-get install john sslscan sqlite3 snmp smbclient hyrda python-netaddr phantomjs ldap-utils`

### actually runnning APT2 (requires msfrpcd)

Interactive: `sudo python apt2.py -v -v`

Non-interactive `sudo python apt2.py -v -v -b --target 192.168.0.0/24`

## Virtual wall shared location

in elke VM __/groups/wall2-ilabt-iminds-be/cybersecurity/__