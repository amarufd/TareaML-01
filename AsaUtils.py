#!/usr/bin/env python
import binascii
import numpy as np
import sys, getopt
import re
import os
#import cv2
import itertools
#import caffe
import scipy.io as sio 
import matplotlib.pyplot as plt



def getTrueMin():
    return (0,0)

def getDerivative(x):
    return 2*x 

def getDescriptors(inDir, fileType, descripSize):
    
    nFiles = 0
    for root, dirs, files in os.walk(inDir):
        for filename in files:    
            if filename.endswith(fileType):
                nFiles += 1
                
    #print('Total files:' + str(nFiles))
    
    #Get memory to store descriptors
    inFeats=np.empty([nFiles, descripSize], dtype=np.float32);
    inLabels=np.empty([nFiles], dtype=np.int32);
    
    # 1. Get filenames
    for root, dirs, files in os.walk(inDir):
        fileCounter=0
        #for name in dirs:
        for name,nClasses in zip(dirs,range(0,len(dirs))):
            print 'Reading descriptors of class' + str(nClasses+1) + ' of ' + str(len(dirs))
            classDir=inDir+name
            for root2, dirs2, files2 in os.walk(classDir):
                for name2,loopFilesCnt in zip(files2,range(0,len(files2))):
                    #print 'class' + str(nClasses+1) + ': file ' + str(loopFilesCnt+1) + ' of ' + str(len(files2))
                    if name2.endswith((fileType)):
                        descFile=os.path.join(root2, name2) 
                        inFeats[fileCounter]=np.loadtxt(descFile)
                        inLabels[fileCounter]=nClasses+1
                        fileCounter+=1                                   
    return inFeats,inLabels
                              

def getMatlabDescriptors(inDir, fileType, descripSize):
    
    nFiles = 0
    classNames = list()
    inDir=os.path.normpath(inDir)
    for _, dirs, files in os.walk(inDir):
        for nameDir in dirs:
            classNames.append(nameDir)
        for filename in files:    
            if filename.endswith(fileType):
                nFiles += 1
                            
    #print('Total files:' + str(nFiles))
    
    #Get memory to store descriptors
    inFeats=np.empty([nFiles, descripSize], dtype=np.float32);
    inLabels=np.empty([nFiles], dtype=np.int32);
    
    # 1. Get filenames
    for root, dirs, files in os.walk(inDir):
        fileCounter=0
        #for name in dirs:
        for name,nClasses in zip(dirs,range(0,len(dirs))):
            print 'Reading descriptors of class: ' + classNames[nClasses] + ', class:' + str(nClasses+1) + ' of ' + str(len(dirs))
            classDir=inDir+'/'+name
            classDir=os.path.normpath(classDir)
            for root2, dirs2, files2 in os.walk(classDir):
                for name2,loopFilesCnt in zip(files2,range(0,len(files2))):
                    #print 'class' + str(nClasses+1) + ': file ' + str(loopFilesCnt+1) + ' of ' + str(len(files2))
                    if name2.endswith((fileType)):
                        descFile=os.path.join(root2, name2) 
                        #inFeats[fileCounter]=np.loadtxt(descFile)
                        aux=sio.loadmat(descFile)
                        inFeats[fileCounter]=aux['stored']
                        #inFeats[fileCounter] = sio.loadmat(descFile)
                        
                        inLabels[fileCounter]=nClasses+1
                        fileCounter+=1                                   
    return classNames,inFeats,inLabels



def getDescriptors2(options, inDir, namesFile):
    # 1. Get filenames
    descripSize=options['DescriptorSize']
    inFilenames=getNames(namesFile)
    nFiles=len(inFilenames)
    print nFiles
    # 2. Get mem to store descriptors
    inFeats=np.empty([nFiles, descripSize], dtype=np.float32);
    inLabels=np.empty([nFiles, 1], dtype=np.int32);
    
    # 3. Read descriptor files
    for i in range(0,1):
        print 'file ' + str(i+1) + ' of ' + str(nFiles)

        descFile=inFilenames[i][2:]
        descFile=inDir + descFile
        
        inFeats[i]=np.loadtxt(descFile)
        inLabels[i]=i
    return inFeats,inLabels





def getNames(fileinfo):
    f = open(fileinfo, 'r')
    inFileNames= f.read().splitlines()
    return inFileNames


def getImgNames(fileinfo):
    f = open(fileinfo, 'r')
    inFileImNames= f.read().splitlines()
    return inFileImNames


def getImgPatches(img, patchRowSize, patchColSize, stride):
    
    #cv2.imshow('image',img)
    #cv2.waitKey(0)

    maxRow,maxCol,colorDepth=np.shape(img)

    initRow,endRow, initCol, endCol=[0,patchRowSize,0,patchColSize]

    #total number of images patches
    totalPatches=(1+int((maxRow-patchRowSize)/stride))*(1+int((maxCol-patchColSize)/stride));
    patchesArray=np.empty([totalPatches, patchRowSize, patchRowSize, colorDepth], dtype=np.float32);

    nPatch=0
    while True:
        if(initCol >= maxCol or endCol > maxCol) :
            initRow,endRow, initCol, endCol=[initRow+stride,endRow+stride,0,patchColSize]

        if(initRow >= maxRow or endRow > maxRow) :
            break

        patchesArray[nPatch]=img[initRow:endRow, initCol:endCol,0:colorDepth]
        #cv2.imshow('image',patchesArray[nPatch].astype(np.uint8))
        #cv2.waitKey(0)
        nPatch+=1
        initCol, endCol=[initCol+stride, endCol+stride]
    return patchesArray

#def saveDescriptor(descriptor, inFilename, descriptorsPath):
#    head, tail = os.path.split(inFilename)
#    dummy = re.split(r'.jpg{1,}',tail)
#    filename=descriptorsPath + dummy[0] +'.fc6'
#    np.savetxt(filename, descriptor, delimiter=',') 
#    return

def saveDescriptor(descriptor, inFilename, fileType='fc6', descriptorsPath='', keepClassFolders=0):
    
    imgFolder, filename = os.path.split(inFilename)

    if descriptorsPath=="":
        descriptorsPath='data/Descriptors/'

    if keepClassFolders==1:
        classFolder = re.split('/',imgFolder)
        outFolder = descriptorsPath + classFolder[-1]
    else:
        outFolder = descriptorsPath
    
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    
    outFilename = filename[:-4]

    #filename= outFolder + '/' + outFilename + '.' + fileType
    #np.savetxt(filename, descriptor, delimiter=',')
    filename2= outFolder + '/' + outFilename + '.' + fileType
    sio.savemat(filename2,  {'stored' : descriptor})
    
    return

def saveDescriptor2(descriptor, inFilename, fileType='fc6', descriptorsPath='', keepClassFolders=0, charID='A'):
    
    imgFolder, filename = os.path.split(inFilename)

    if descriptorsPath=="":
        descriptorsPath='data/Descriptors/'

    if keepClassFolders==1:
        classFolder = re.split('/',imgFolder)
        outFolder = descriptorsPath + classFolder[-1]
    else:
        outFolder = descriptorsPath
    
    if not os.path.exists(outFolder):
        os.makedirs(outFolder)
    
    outFilename = filename[:-4]

    #filename= outFolder + '/' + outFilename + '.' + fileType
    #np.savetxt(filename, descriptor, delimiter=',')
    filename2= outFolder + '/' + outFilename + charID + '.' + fileType
    sio.savemat(filename2,  {'stored' : descriptor})
    
    return

def savePatchFeats(descriptors, inFilename, descriptorsPath='', keepClassFolders=0):
    saveDescriptor(descriptors, inFilename, 'Pfc6', descriptorsPath, keepClassFolders)
    return

def checkValidCells(row,col,maxRow, maxCol):
    if(row >= maxRow or row < 0):
        return 0
    if(col >= maxCol or col < 0):
        return 0
    return 1

#This function scale an input image keeping the aspect ratio
#and setting the size of the smallest side to minSide
# def ScaleImg(img, minSide):
#     inRow,inCol,dummy=np.shape(img)
# 
#     if inRow > inCol :
#         dim = (int((float(minSide)*inRow)/float(inCol)),minSide)
#     else:
#         dim = (minSide,int((float(minSide)*inCol)/float(inRow)))
#  
#     #return(caffe.io.resize_image(img, dim))
#     return(cv2.resize(img, (dim[1],dim[0])))


#This function scale an input image keeping the aspect ratio
#and setting the size of the smallest side to minSide
# def ScaleImg3(img, minSide, patchSize):
# 
#     if(img.ndim>2):
#         inRow,inCol,dummy=img.shape
#     else:
#         inRow,inCol=img.shape
# 
#     if inRow > inCol :
#         dim = (int((float(minSide)*inRow)/float(inCol)),minSide)
#         lostImgPart=dim[0] % patchSize
#         if(lostImgPart > 0.5 * patchSize):
#             dim=(dim[0]+patchSize-lostImgPart,dim[1])        
#     else:
#         dim = (minSide,int((float(minSide)*inCol)/float(inRow)))
#         lostImgPart=dim[1] % patchSize
#         if(lostImgPart > 0.5 * patchSize):
#             dim=(dim[0],dim[1]+patchSize-lostImgPart)
#     
#     return(cv2.resize(img, dim))



def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    #else:
        #print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, round(cm[i, j],2),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def parseCmdLine(argv):
   names = ['','','','']
   try:
      opts, args = getopt.getopt(argv,"hi:o:p:t")
   except getopt.GetoptError:
      return None
 
   for opt, arg in opts:
      if opt == '-h':
         print 'getPatchLevelFC6.py -i <file with imagenames> -o <name output folder> -p <0 output patch descriptors, 1 apply maxPooling> -t <1 keep class folders, 0 dont>'
         sys.exit()
      elif opt == '-i':
         names[0]=arg
      elif opt == '-o':
         names[1]=arg
      elif opt == '-p':
         names[2]=arg
      elif opt == '-t':
         names[2]=arg


   return(names)
