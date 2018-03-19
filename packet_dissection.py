#! /usr/bin/python2
from scapy.all import *
import pandas as pd
import sys
import matplotlib.pyplot as plt
import numpy as np
import pprint


def main(argv):
    try:
        # Currently this reads the entire file at once, for really big files it would be better to stream the data, see scapy PacketReader()
        pcap_pkts = rdpcap(sys.argv[1])
    except OSError as err:
        print("OS error {0}".format(err))

    pp = pprint.PrettyPrinter(indent=2)
    layerstore_by_proto = {}
    attributes_per_layer = {}
    fully_mapped_data = []
    for pkt in pcap_pkts:
        layerlist = list(expand(pkt))
        one_full_packet = {}
        # always add the link layer protocol, will be ethernet most of the time
        # scapy doesn't recognize a layer called ethernet, but every packet still carries that info at its first level
        layerstore_by_proto.setdefault(layerlist[0], []).append(pkt.fields.values())
        one_full_packet["Ethernet"] = pkt.fields
        attributes_per_layer[layerlist[0]] = pkt.fields.keys()
        # now go do all the other layers
        for layer in layerlist[1::]:
            try:
                # the important line, take all the details at the desired level of the network stack and add them to the dictionary
                layerstore_by_proto.setdefault(
                    layer, []).append(pkt[layer].fields.values())
                # also add this information to the full packet representation
                one_full_packet[layer] = pkt[layer].fields
                if layer not in attributes_per_layer:
                    attributes_per_layer[layer] = pkt[layer].fields.keys()
            except IndexError as err:
                # some layers have odd names and don't like TCP in ICMP, assuming that the real protocol ends the string, we try if looking into the packet
                # specifically on that layer, if that succeeds, we extract the information, otherwise we skip
                detaillayer = layer.split(' ')[-1]
                if pkt.haslayer(detaillayer):
                    layerstore_by_proto.setdefault(detaillayer, []).append(pkt[detaillayer].fields.values())
                    one_full_packet[detaillayer] = pkt[detaillayer].fields
                    if detaillayer not in attributes_per_layer:
                        attributes_per_layer[detaillayer] = pkt[detaillayer].fields.keys()
                else:
                    print "Detail layer " + layer.split(' ')[-1]+"still not recognized"
                continue
        # Store all info of the packet in a single object
        fully_mapped_data.append(one_full_packet)
        one_full_packet = {}


    #pp.pprint(layerstore_by_proto)
    #pp.pprint(fully_mapped_data)
    #pp.pprint(attributes_per_layer)

    df_by_proto = {}
    for p in attributes_per_layer:        
         df_by_proto[p] = pd.DataFrame(columns=attributes_per_layer[p])
    pp.pprint(df_by_proto)
    for pk in fully_mapped_data:
         for pr in pk:
             tmpdf = pd.Series(data=pk[pr])
             df_by_proto[pr] = df_by_proto[pr].append(tmpdf,ignore_index=True)
    pp.pprint(df_by_proto)
         

def firstplot():
    np.random.seed(19680801)
    data = np.random.randn(2, 100)

    fig, axs = plt.subplots(2, 2, figsize=(5, 5))
    axs[0, 0].hist(data[0])
    axs[1, 0].scatter(data[0], data[1])
    axs[0, 1].plot(data[0], data[1])
    axs[1, 1].hist2d(data[0], data[1])

    plt.show()

def expand(pkt):
    yield pkt.name
    while pkt.payload:
        pkt = pkt.payload
        yield pkt.name


if __name__ == "__main__":
    main(sys.argv)