#! /usr/bin/python2
from scapy.all import *
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import pprint
import time


def main(argv):
    myreader = PcapReader("2018-03-11 03:44:41enp0s9.pcap")
    i = 0
    for p in myreader:
        pkt = p.payload
        print "p time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(p.time))
        print "pkt time", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(pkt.time))
        for part in expand(p):
            print part, " ",
        print
        print "p length",len(p)
        print p.summary()

        if i == 50:
            break
        else:
            i += 1

def expand(pkt):
    yield pkt.name
    while pkt.payload:
        pkt = pkt.payload
        yield pkt.name


if __name__ == "__main__":
    main(sys.argv)
