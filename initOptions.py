#!/usr/bin/env python


def selections():
    # Path to class folders with MIT-67 FC6 descriptors
    options = {'MIT-10-DescriptorsPath':'./MIT-10-Classes/Feats/'}
    options['MIT-67-DescriptorsPath']='./MIT-67-Classes/Feats/'
    
    #Dimensionality of feature space, i.e., FC6 descriptors
    options['DescriptorSize'] = 4096

    return options