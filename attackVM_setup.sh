ipv4_nat_access () { 
    sudo route del default gw 10.2.15.254 && sudo route add default gw 10.2.15.253
    sudo route add -net 10.11.0.0 netmask 255.255.0.0 gw 10.2.15.254
    sudo route add -net 10.2.32.0 netmask 255.255.240.0 gw 10.2.15.254
}

basic_prerequisites() {
    sudo apt-get update
    sudo apt-get install -y ruby git nmap python-pip
    sudo apt-get install -y subversion
    sudo apt-get install -y build-essential ruby-dev libpcap-dev
}

install_metasploit(){ 
    curl https://raw.githubusercontent.com/rapid7/metasploit-omnibus/master/config/templates/metasploit-framework-wrappers/msfupdate.erb > msfinstall && chmod 755 msfinstall &&  ./msfinstall
}

acquire_apt2_in_shared_space(){ 
    cd /groups/wall2-ilabt-iminds-be/cybersecurity/
    git clone https://github.com/Str-Gen/apt2.git
}

install_apt2_extra(){
    sudo apt-get install -y john sslscan sqlite3 snmp smbclient hydra python-netaddr phantomjs ldap-utils
}

pip_install_python_modules(){ 
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
}

start_msfrpcd(){
    sudo msfrpcd -U msf -P msfpass -p 55552 -f "/api/" -a 127.0.0.1 -S
}

