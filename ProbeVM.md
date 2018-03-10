# Metasploit VM automation

This markdown document contains information about the VM that has been set up to simulate a malicious node.
The automation code itself (shell), has been left out and only a description of the available functions in apt2/misc/attackVM_setup.sh is kept here.
The purpose of this file is to provide a detailed description to rebuild the image from the stock Ubuntu 16.04LTS image that is available on the wall.
Some caveats about using the automation framework are also listed here.

## Installation

* `ipv4_nat_access()`

Virtual wall IPv4 internet access (NAT)

[The virtual wall](https://www.ugent.be/ea/idlab/en/research/research-infrastructure/virtual-wall.htm) where the experiments are run, have a limited supply op IPv4 addresses.
In order to access the outside world, you have to configure NAT-ing as described on [this](http://doc.ilabt.iminds.be/ilabt-documentation/tipsandtricks.html) page.

* `basic_prerequisites()`

Some software is required before anything else. The required packages are bundled in this shell function. (Git, ruby, subversion, python-pip,...)
Uses apt-get, so for other distributions that could serve as base image this function would need to be adapted.

* `install_metasploit()`

An automated way to install the metasploit framework. The default installers are graphical and require user-interaction to complete the process.

Metasploit (no graphical installer ! [nightly](https://github.com/rapid7/metasploit-framework/wiki/Nightly-Installers))

* `acquire_apt2_in_shared_space()`

This will clone [my fork of the apt2 github repo](https://github.com/Str-Gen/apt2.git) to the shared location for the cybersecurity project.

* `pip_install_python_modules()`

APT2 relies on certain python modules to run the various payloads. This function installs the necessary ones (at the time of writing March 2018).

Dependencies that rely on each other can't be installed with a single invocation of pip install followed by a list. That's why in the shellscript every module install command is on a new line.

* `install_apt2_extra()`

APT2 also relies on some software distributed as packages.

__These functions used togheter should be able to mirror the current setup of the VM.__

## Starting APT2

* `start_msfrpcd()`

Starts Metasploit RPC server on localhost. 
__Note that this method does work, but you can't read output from the metasploit console when using this method, nor will you be able to tap into the sessions that can be automatically created on the compromised targets if an exploit succeeded.__

* The interactive approach

This method is currently required, certainly if you wish to run the ipidseq.py module. This is something that I've tried to automate, but so far no methods work. It's annoying if all you have is a single terminal ssh connection to the machine running metasploit. Msfconsole can't be backgrounded without stopping it and launching it in a subshell with a script that contains the necessary setup steps doesn't work either.
Therefore if you want to have interaction this is how to connect:

```bash
sudo msfconsole
load msgrpc User=msf Pass=msfpass NetworkPort=55552
```

### actually runnning APT2 (requires msfrpcd first or the interactive method)

__Caveat:__ running APT2 requires superuser rights, mostly because of the tools they may invoke. There are probably ways around this restriction, but this is the easiest way to get going.

Interactive: `sudo python apt2.py -v -v`

Non-interactive `sudo python apt2.py -v -v -b --target 192.168.0.0/24`

For more information on the capabilities of APT2 I'll refer to the official documentation over at [https://github.com/MooseDojo/apt2].

## Virtual wall shared location

In each VM this folder __/groups/wall2-ilabt-iminds-be/cybersecurity/__ is currently a preserved, shared space to store work.