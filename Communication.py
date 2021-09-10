
# Using baseband processing, obtain the BER (bit error rate) vs. SNR plots for
# * BPSK
# * QPSK
# * 8-PSK
# * 16-QAM
# * 64-QAM.
# by averaging over a 10,000 transmitted symbols each.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd


####################################################################################################################################

## mapping bits to constellation points


## BPSK
def map_bpsk(bitstream,BPSK):
    
    mapped=np.zeros(len(bitstream));
    for i in range(len(bitstream)):
        if (bitstream[i]==0):
            mapped[i]=BPSK[0];
        else:
            mapped[i]=BPSK[1];
    return np.real(mapped+0.707*np.random.randn(1,len(mapped))+1j*0.707*np.random.randn(1,len(mapped)));

## QPSK
def map_qpsk(bitstream,QPSK):
    if ((len(bitstream)%3)==0):
        pad =0;
    else:
        pad = 3- (len(bitstream)%3);
    bit=np.ravel(np.concatenate([bitstream,np.random.randint(0,2,pad)]));
    bits= bit.reshape(int(len(bit)/3),3);
    mapped=np.array([np.zeros(len(bits)),np.zeros(len(bits))]).T;
    for i in range(len(bits)):
        #print(i)
        if (all(bits[i]==[0,0,0])):
            mapped[i][0]=QPSK[0][0];mapped[i][1]=QPSK[1][0];
        elif (all(bits[i]==[0,0,1])):
            mapped[i][0]=QPSK[0][1];mapped[i][1]=QPSK[1][1];
        elif (all(bits[i]==[0,1,0])):
            mapped[i][0]=QPSK[0][2];mapped[i][1]=QPSK[1][2];
        elif (all(bits[i]==[0,1,1])):
            mapped[i][0]=QPSK[0][3];mapped[i][1]=QPSK[1][3];
        elif (all(bits[i]==[1,0,0])):
            mapped[i][0]=QPSK[0][4];mapped[i][1]=QPSK[1][4];
        elif (all(bits[i]==[1,0,1])):
            mapped[i][0]=QPSK[0][5];mapped[i][1]=QPSK[1][5];
        elif (all(bits[i]==[1,1,0])):
            mapped[i][0]=QPSK[0][6];mapped[i][1]=QPSK[1][6];
        else:
            mapped[i][0]=QPSK[0][7];mapped[i][1]=QPSK[1][7];
    
    mpp = mapped.T;  
    return mpp;


## PSK
def map_psk(bitstream,PSK):
    if ((len(bitstream)%3)==0):
        pad =0;
    else:
        pad = 3- (len(bitstream)%3);
    bit=np.ravel(np.concatenate([bitstream,np.random.randint(0,2,pad)]));
    bits= bit.reshape(int(len(bit)/3),3);
    mapped=np.array([np.zeros(len(bits)),np.zeros(len(bits))]).T;
    for i in range(len(bits)):
        if (all(bits[i]==[0,0,0])):
            mapped[i][0]=PSK[0][5];mapped[i][1]=PSK[1][5];
        elif (all(bits[i]==[0,0,1])):
            mapped[i][0]=PSK[0][4];mapped[i][1]=PSK[1][4];
        elif (all(bits[i]==[0,1,0])):
            mapped[i][0]=PSK[0][2];mapped[i][1]=PSK[1][2];
        elif (all(bits[i]==[0,1,1])):
            mapped[i][0]=PSK[0][3];mapped[i][1]=PSK[1][3];
        elif (all(bits[i]==[1,0,0])):
            mapped[i][0]=PSK[0][6];mapped[i][1]=PSK[1][6];
        elif (all(bits[i]==[1,0,1])):
            mapped[i][0]=PSK[0][7];mapped[i][1]=PSK[1][7];
        elif (all(bits[i]==[1,1,0])):
            mapped[i][0]=PSK[0][1];mapped[i][1]=PSK[1][1];
        elif (all(bits[i]==[1,1,1])):
            mapped[i][0]=PSK[0][0];mapped[i][1]=PSK[1][0];

    mapp= mapped.T;         
    return mapp;
            
    

## QAM16
def map_qam16(bits,QAM16):
    if ((len(bits)%4)==0):
        pad =0;
    else:
        pad = 4- (len(bits)%4);
    bits=np.ravel(np.concatenate([bits,np.random.randint(0,2,pad)]));
    bitstream= bits.reshape(int(len(bits)/4),4);
    mapped=np.array([np.zeros(len(bitstream)),np.zeros(len(bitstream))]).T;
    
    for i in range(len(bitstream)):
        if (all(bitstream[i]==[0,0,0,0])):
            mapped[i][0]=QAM16[0,0];mapped[i][1]=QAM16[1,0];
        elif (all(bitstream[i]==[0,0,0,1])):
            mapped[i][0]=QAM16[0,1];mapped[i][1]=QAM16[1,1];
        elif (all(bitstream[i]==[0,0,1,0])):
            mapped[i][0]=QAM16[0,2];mapped[i][1]=QAM16[1,2];
        elif (all(bitstream[i]==[0,0,1,1])):
            mapped[i][0]=QAM16[0,3];mapped[i][1]=QAM16[1,3];
        elif (all(bitstream[i]==[0,1,0,0])):
            mapped[i][0]=QAM16[0,4];mapped[i][1]=QAM16[1,4];
        elif (all(bitstream[i]==[0,1,0,1])):
            mapped[i][0]=QAM16[0,5];mapped[i][1]=QAM16[1,5];
        elif (all(bitstream[i]==[0,1,1,0])):
            mapped[i][0]=QAM16[0,6];mapped[i][1]=QAM16[1,6];
        elif (all(bitstream[i]==[0,1,1,1])):
            mapped[i][0]=QAM16[0,7];mapped[i][1]=QAM16[1,7];
        elif (all(bitstream[i]==[1,0,0,0])):
            mapped[i][0]=QAM16[0,8];mapped[i][1]=QAM16[1,8];
        elif (all(bitstream[i]==[1,0,0,1])):
            mapped[i][0]=QAM16[0,9];mapped[i][1]=QAM16[1,9];
        elif (all(bitstream[i]==[1,0,1,0])):
            mapped[i][0]=QAM16[0,10];mapped[i][1]=QAM16[1,10];
        elif (all(bitstream[i]==[1,0,1,1])):
            mapped[i][0]=QAM16[0,11];mapped[i][1]=QAM16[1,11];
        elif (all(bitstream[i]==[1,1,0,0])):
            mapped[i][0]=QAM16[0,12];mapped[i][1]=QAM16[1,12];
        elif (all(bitstream[i]==[1,1,0,1])):
            mapped[i][0]=QAM16[0,13];mapped[i][1]=QAM16[1,13];
        elif (all(bitstream[i]==[1,1,1,0])):
            mapped[i][0]=QAM16[0,14];mapped[i][1]=QAM16[1,14];
        else:
            mapped[i][0]=QAM16[0,15];mapped[i][1]=QAM16[1,15];

    mapp= mapped.T;        
    return mapp;




## QAM64

def map_qam64(bits,QAM64):
    if ((len(bits)%6)==0):
        pad =0;
    else:
        pad = 6- (len(bits)%6);

    bit=np.ravel(np.concatenate([bits,np.random.randint(0,2,pad)]));
    bitstream= bit.reshape(int(len(bit)/6),6);
    mapped=np.array([np.zeros(len(bitstream)),np.zeros(len(bitstream))]).T;
    
    for i in range(len(bitstream)):
        
        if (all(bitstream[i]==[0,0,0,0,0,0])):
            mapped[i][0]=QAM64[0,0];mapped[i][1]=QAM64[1,0];
        elif (all(bitstream[i]==[0,0,0,0,0,1])):
            mapped[i][0]=QAM64[0,1];mapped[i][1]=QAM64[1,1];
        elif (all(bitstream[i]==[0,0,0,0,1,0])):
            mapped[i][0]=QAM64[0,2];mapped[i][1]=QAM64[1,2];
        elif (all(bitstream[i]==[0,0,0,0,1,1])):
            mapped[i][0]=QAM64[0,3];mapped[i][1]=QAM64[1,3];
        elif (all(bitstream[i]==[0,0,0,1,0,0])):
            mapped[i][0]=QAM64[0,4];mapped[i][1]=QAM64[1,4];
        elif (all(bitstream[i]==[0,0,0,1,0,1])):
            mapped[i][0]=QAM64[0,5];mapped[i][1]=QAM64[1,5];
        elif (all(bitstream[i]==[0,0,0,1,1,0])):
            mapped[i][0]=QAM64[0,6];mapped[i][1]=QAM64[1,6];
        elif (all(bitstream[i]==[0,0,0,1,1,1])):
            mapped[i][0]=QAM64[0,7];mapped[i][1]=QAM64[1,7];
        elif (all(bitstream[i]==[0,0,1,0,0,0])):
            mapped[i][0]=QAM64[0,8];mapped[i][1]=QAM64[1,8];
        elif (all(bitstream[i]==[0,0,1,0,0,1])):
            mapped[i][0]=QAM64[0,9];mapped[i][1]=QAM64[1,9];
        elif (all(bitstream[i]==[0,0,1,0,1,0])):
            mapped[i][0]=QAM64[0,10];mapped[i][1]=QAM64[1,10];
        elif (all(bitstream[i]==[0,0,1,0,1,1])):
            mapped[i][0]=QAM64[0,11];mapped[i][1]=QAM64[1,11];
        elif (all(bitstream[i]==[0,0,1,1,0,0])):
            mapped[i][0]=QAM64[0,12];mapped[i][1]=QAM64[1,12];
        elif (all(bitstream[i]==[0,0,1,1,0,1])):
            mapped[i][0]=QAM64[0,13];mapped[i][1]=QAM64[1,13];
        elif (all(bitstream[i]==[0,0,1,1,1,0])):
            mapped[i][0]=QAM64[0,14];mapped[i][1]=QAM64[1,14];
        elif (all(bitstream[i]==[0,0,1,1,1,1])):
            mapped[i][0]=QAM64[0,15];mapped[i][1]=QAM64[1,15];
            
        elif (all(bitstream[i]==[0,1,0,0,0,0])):
            mapped[i][0]=QAM64[0,16];mapped[i][1]=QAM64[1,16];            
        elif (all(bitstream[i]==[0,1,0,0,0,1])):
            mapped[i][0]=QAM64[0,17];mapped[i][1]=QAM64[1,17];
        elif (all(bitstream[i]==[0,1,0,0,1,0])):
            mapped[i][0]=QAM64[0,18];mapped[i][1]=QAM64[1,18];
        elif (all(bitstream[i]==[0,1,0,0,1,1])):
            mapped[i][0]=QAM64[0,19];mapped[i][1]=QAM64[1,19];
        elif (all(bitstream[i]==[0,1,0,1,0,0])):
            mapped[i][0]=QAM64[0,20];mapped[i][1]=QAM64[1,20];
        elif (all(bitstream[i]==[0,1,0,1,0,1])):
            mapped[i][0]=QAM64[0,21];mapped[i][1]=QAM64[1,21];
        elif (all(bitstream[i]==[0,1,0,1,1,0])):
            mapped[i][0]=QAM64[0,22];mapped[i][1]=QAM64[1,22];
        elif (all(bitstream[i]==[0,1,0,1,1,1])):
            mapped[i][0]=QAM64[0,23];mapped[i][1]=QAM64[1,23];
        elif (all(bitstream[i]==[0,1,1,0,0,0])):
            mapped[i][0]=QAM64[0,24];mapped[i][1]=QAM64[1,24];
        elif (all(bitstream[i]==[0,1,1,0,0,1])):
            mapped[i][0]=QAM64[0,25];mapped[i][1]=QAM64[1,25];
        elif (all(bitstream[i]==[0,1,1,0,1,0])):
            mapped[i][0]=QAM64[0,26];mapped[i][1]=QAM64[1,26];
        elif (all(bitstream[i]==[0,1,1,0,1,1])):
            mapped[i][0]=QAM64[0,27];mapped[i][1]=QAM64[1,27];
        elif (all(bitstream[i]==[0,1,1,1,0,0])):
            mapped[i][0]=QAM64[0,28];mapped[i][1]=QAM64[1,28];
        elif (all(bitstream[i]==[0,1,1,1,0,1])):
            mapped[i][0]=QAM64[0,29];mapped[i][1]=QAM64[1,29];
        elif (all(bitstream[i]==[0,1,1,1,1,0])):
            mapped[i][0]=QAM64[0,30];mapped[i][1]=QAM64[1,30];
        elif (all(bitstream[i]==[0,1,1,1,1,1])):
            mapped[i][0]=QAM64[0,31];mapped[i][1]=QAM64[1,31];
            
        elif (all(bitstream[i]==[1,0,0,0,0,0])):
            mapped[i][0]=QAM64[0,32];mapped[i][1]=QAM64[1,32];            
        elif (all(bitstream[i]==[1,0,0,0,0,1])):
            mapped[i][0]=QAM64[0,33];mapped[i][1]=QAM64[1,33];
        elif (all(bitstream[i]==[1,0,0,0,1,0])):
            mapped[i][0]=QAM64[0,34];mapped[i][1]=QAM64[1,34];
        elif (all(bitstream[i]==[1,0,0,0,1,1])):
            mapped[i][0]=QAM64[0,35];mapped[i][1]=QAM64[1,35];
        elif (all(bitstream[i]==[1,0,0,1,0,0])):
            mapped[i][0]=QAM64[0,36];mapped[i][1]=QAM64[1,36];
        elif (all(bitstream[i]==[1,0,0,1,0,1])):
            mapped[i][0]=QAM64[0,37];mapped[i][1]=QAM64[1,37];
        elif (all(bitstream[i]==[1,0,0,1,1,0])):
            mapped[i][0]=QAM64[0,38];mapped[i][1]=QAM64[1,38];
        elif (all(bitstream[i]==[1,0,0,1,1,1])):
            mapped[i][0]=QAM64[0,39];mapped[i][1]=QAM64[1,39];
        elif (all(bitstream[i]==[1,0,1,0,0,0])):
            mapped[i][0]=QAM64[0,40];mapped[i][1]=QAM64[1,40];
        elif (all(bitstream[i]==[1,0,1,0,0,1])):
            mapped[i][0]=QAM64[0,41];mapped[i][1]=QAM64[1,41];
        elif (all(bitstream[i]==[1,0,1,0,1,0])):
            mapped[i][0]=QAM64[0,42];mapped[i][1]=QAM64[1,42];
        elif (all(bitstream[i]==[1,0,1,0,1,1])):
            mapped[i][0]=QAM64[0,43];mapped[i][1]=QAM64[1,43];
        elif (all(bitstream[i]==[1,0,1,1,0,0])):
            mapped[i][0]=QAM64[0,44];mapped[i][1]=QAM64[1,44];
        elif (all(bitstream[i]==[1,0,1,1,0,1])):
            mapped[i][0]=QAM64[0,45];mapped[i][1]=QAM64[1,45];
        elif (all(bitstream[i]==[1,0,1,0,1,0])):
            mapped[i][0]=QAM64[0,46];mapped[i][1]=QAM64[1,46];
        elif (all(bitstream[i]==[1,0,1,1,1,1])):
            mapped[i][0]=QAM64[0,47];mapped[i][1]=QAM64[1,47];
            
            
        elif (all(bitstream[i]==[1,1,0,0,0,0])):
            mapped[i][0]=QAM64[0,32];mapped[i][1]=QAM64[1,48];            
        elif (all(bitstream[i]==[1,1,0,0,0,1])):
            mapped[i][0]=QAM64[0,33];mapped[i][1]=QAM64[1,49];
        elif (all(bitstream[i]==[1,1,0,0,1,0])):
            mapped[i][0]=QAM64[0,34];mapped[i][1]=QAM64[1,50];
        elif (all(bitstream[i]==[1,1,0,0,1,1])):
            mapped[i][0]=QAM64[0,35];mapped[i][1]=QAM64[1,51];
        elif (all(bitstream[i]==[1,1,0,1,0,0])):
            mapped[i][0]=QAM64[0,36];mapped[i][1]=QAM64[1,52];
        elif (all(bitstream[i]==[1,1,0,1,0,1])):
            mapped[i][0]=QAM64[0,37];mapped[i][1]=QAM64[1,53];
        elif (all(bitstream[i]==[1,1,0,1,1,0])):
            mapped[i][0]=QAM64[0,38];mapped[i][1]=QAM64[1,54];
        elif (all(bitstream[i]==[1,1,0,1,1,1])):
            mapped[i][0]=QAM64[0,39];mapped[i][1]=QAM64[1,55];
        elif (all(bitstream[i]==[1,1,1,0,0,0])):
            mapped[i][0]=QAM64[0,40];mapped[i][1]=QAM64[1,56];
        elif (all(bitstream[i]==[1,1,1,0,0,1])):
            mapped[i][0]=QAM64[0,41];mapped[i][1]=QAM64[1,57];
        elif (all(bitstream[i]==[1,1,1,0,1,0])):
            mapped[i][0]=QAM64[0,42];mapped[i][1]=QAM64[1,58];
        elif (all(bitstream[i]==[1,1,1,0,1,1])):
            mapped[i][0]=QAM64[0,43];mapped[i][1]=QAM64[1,59];
        elif (all(bitstream[i]==[1,1,1,1,0,0])):
            mapped[i][0]=QAM64[0,44];mapped[i][1]=QAM64[1,60];
        elif (all(bitstream[i]==[1,1,1,1,0,1])):
            mapped[i][0]=QAM64[0,45];mapped[i][1]=QAM64[1,61];
        elif (all(bitstream[i]==[1,1,1,1,1,0])):
            mapped[i][0]=QAM64[0,46];mapped[i][1]=QAM64[1,62];
        else:
            mapped[i][0]=QAM64[0,47];mapped[i][1]=QAM64[1,63];
    mapp= mapped.T;
    return mapp;

    
    

####################################################################################################################################
#Defining decision boundaries.

 
#BPSK
def bpsk_decision(inpu,BPSK):
    inp=(np.ravel(inpu))
    for i in range(len(inp)):
        if (inp[i]>0):
            inp[i]=BPSK[1];
        else:
            inp[i]=BPSK[0];
    return inp;


#QPSK boundaries.

def qpsk_dist(inp,quad,QPSK):
    dist =1000;
    index=10;
    for i in range(8):
        if dist>((inp-QPSK[0][i])**2 +(quad-QPSK[1][i])**2)**0.5:
            dist = ((inp-QPSK[0][i])**2 +(quad-QPSK[1][i])**2)**0.5;
            index= i;
    return dist,index;


def qpsk_decision(inp,b,QPSK):
    inphase=np.ravel(inp[0]);
    quadphase=np.ravel(inp[1]);
    a= 2**0.5;
    
    m= (a+1)/2;
    # print(inp.shape)
    #dec_qpsk = np.array([[-m,-0.5,0.5,m],[-m,-0.5,0.5,m]]);
    for i in range(len(inphase)):
        dist,Cons=qpsk_dist(inphase[i],quadphase[i],QPSK)
        inphase[i]= QPSK[0][Cons];
        quadphase[i]= QPSK[1][Cons];
        
    return np.array([inphase,quadphase]);


# defining psk bundaries.
def psk_decision(inp,Amp,PSK):
    inphase=np.zeros(len(inp[0]));
    quadphase=np.zeros(len(inp[1]));
    psk_sym = np.angle(inp[0]+1j*inp[1],deg=True);
    dec = np.angle(np.cos(np.arange(1,17,2)*(np.pi/8))+1j*np.sin(np.arange(1,17,2)*(np.pi/8)),deg=True)
    print(psk_sym[0:30]);
    for j in range(len(psk_sym)):
        if (psk_sym[j]<(-22.5)):
            psk_sym[j]=psk_sym[j]+360;
    print(psk_sym[0:30]);
    for i in range(len(psk_sym)):
        

        if ((psk_sym[i]<dec[0]) & (psk_sym[i]>=dec[7])):
            inphase[i]= PSK[0][0];
            quadphase[i]=PSK[0][2];

        elif ((psk_sym[i]<dec[1]) & (psk_sym[i]>=dec[0])):
            inphase[i]= PSK[0][1];
            quadphase[i]=PSK[0][1];

        elif ((psk_sym[i]<dec[2]) & (psk_sym[i]>=dec[1])):
            inphase[i]= PSK[0][2];
            quadphase[i]=PSK[0][0];

        elif ((psk_sym[i]<dec[3]) & (psk_sym[i]>=dec[2])):
            inphase[i]= PSK[0][3];
            quadphase[i]=PSK[0][1];

        elif ((psk_sym[i]<dec[4]+360) & (psk_sym[i]>=dec[3])):
            inphase[i]= PSK[0][4];
            quadphase[i]=PSK[0][2];

        elif ((psk_sym[i]<dec[5]+360) & (psk_sym[i]>=dec[4]+360)):
            inphase[i]= PSK[0][3];
            quadphase[i]=PSK[0][3];

        elif ((psk_sym[i]<dec[6]+360) & (psk_sym[i]>=dec[5]+360)):
            inphase[i]= PSK[0][2];
            quadphase[i]=PSK[0][4];

        elif ((psk_sym[i]<dec[7]+360) & (psk_sym[i]>=dec[6]+360)):
        # else:
            inphase[i]= PSK[0][1];
            quadphase[i]=PSK[0][3];

    return np.array([inphase,quadphase]);


# QAM 16 boundaries

def QAM16_decision(inp,Amp,QAM16):
    inphase=np.ravel(inp[0]);
    quadphase=np.ravel(inp[1]);
    dec= (Amp)*np.array([-2,0,2]);
    for i in range(len(inphase)):
        if (inphase[i]<dec[0]):
            inphase[i]= QAM16[0][0];
        elif (inphase[i]>=dec[0])&(inphase[i]<dec[1]):
            inphase[i]= QAM16[0][1];
        elif (inphase[i]>=dec[1])&(inphase[i]<dec[2]):
            inphase[i]= QAM16[0][2];
        else:
            inphase[i]=QAM16[0,3];
            
        if (quadphase[i]<dec[0]):
            quadphase[i]= QAM16[1][0];
        elif (quadphase[i]>=dec[0])&(quadphase[i]<dec[1]):
            quadphase[i]= QAM16[1][4];
        elif (quadphase[i]>=dec[1])&(quadphase[i]<dec[2]):
            quadphase[i]= QAM16[1][8];
        else:
            quadphase[i]=QAM16[1,12];
    return np.array([inphase,quadphase]);


# QAM 64 boundaries           
            
def QAM64_decision(inp,Amp,QAM64):
    inphase=np.ravel(inp[0]);
    quadphase=np.ravel(inp[1]);
    dec= (Amp)*np.array([-6,-4,-2,0,2,4,6]);
    for i in range(len(inphase)):
        if (inphase[i]<dec[0]):
            inphase[i]= QAM64[0][0];
        elif (inphase[i]>=dec[0])&(inphase[i]<dec[1]):
            inphase[i]= QAM64[0][1];
        elif (inphase[i]>=dec[1])&(inphase[i]<dec[2]):
            inphase[i]= QAM64[0][2];
        elif (inphase[i]>=dec[2])&(inphase[i]<dec[3]):
            inphase[i]= QAM64[0][3];
        elif (inphase[i]>=dec[3])&(inphase[i]<dec[4]):
            inphase[i]= QAM64[0][4];   
        elif (inphase[i]>=dec[4])&(inphase[i]<dec[5]):
            inphase[i]= QAM64[0,5]; 
        elif (inphase[i]>=dec[5])&(inphase[i]<dec[6]):
            inphase[i]= QAM64[0][6]; 
        else:
            inphase[i]=QAM64[0,7];
            
        if (quadphase[i]<dec[0]):
            quadphase[i]= QAM64[1][0];
        elif (quadphase[i]>=dec[0])&(quadphase[i]<dec[1]):
            quadphase[i]= QAM64[1][8];
        elif (quadphase[i]>=dec[1])&(quadphase[i]<dec[2]):
            quadphase[i]= QAM64[1][16];
        elif (quadphase[i]>=dec[2])&(quadphase[i]<dec[3]):
            quadphase[i]= QAM64[1][24];
        elif (quadphase[i]>=dec[3])&(quadphase[i]<dec[4]):
            quadphase[i]= QAM64[1][32];   
        elif (quadphase[i]>=dec[4])&(quadphase[i]<dec[5]):
            quadphase[i]= QAM64[1][40]; 
        elif (quadphase[i]>=dec[5])&(quadphase[i]<dec[6]):
            quadphase[i]= QAM64[1][48]; 
        else:
            quadphase[i]=QAM64[1,56];

    return np.array([inphase,quadphase]);         
    

#############################################################################################################################


## decoding bits
def bpsk_decode(inp1):
    inp= np.ravel(inp1);
    bpsk=np.zeros(len(inp));
    for i  in range(len(inp)):
        if (inp[i]>0):
            bpsk[i]=1;
        else:
            bpsk[i]=0;
    return bpsk.astype(int);


def qpsk_decode(inp,QPSK):
    a=2**0.5;
    inphase = np.ravel(inp[0]);
    quadphase = np.ravel(inp[1]);
    qpsk=[];
    for i in range(len(inphase)):
        if (inphase[i]==QPSK[0][0]):
            qpsk.append(np.array([0,0,0]));
        elif (inphase[i]==QPSK[0][1]):
            if (quadphase[i]==QPSK[0][0]):
                qpsk.append(np.array([0,0,1]));
            else:
                qpsk.append(np.array([0,1,1]));
        elif (inphase[i]==QPSK[0][2]):
            qpsk.append(np.array([0,1,0]));
        elif (inphase[i]==QPSK[0][4]):
            if (quadphase[i]==QPSK[0][4]):
                qpsk.append(np.array([1,0,0]));
            else:
                qpsk.append(np.array([1,1,1]));
        elif (inphase[i]==QPSK[0][5]):
        # else:
            if (quadphase[i]==QPSK[0][4]):
                qpsk.append(np.array([1,0,1]));
            else:
                qpsk.append(np.array([1,1,0]));
                
    return np.ravel(np.array(qpsk)).astype(int);

def psk_decode(inp,amp,PSK):
    inphase = np.ravel(inp[0]);
    quadphase = np.ravel(inp[1]);
    # a=1/np.cos(np.pi/4);
    psk=[];
    for i in range(len(inphase)):
        
        if (inphase[i]==PSK[0,0]):
            psk.append(np.array([1,1,1]))
        elif (inphase[i]==PSK[0,1]):
            if (quadphase[i]==PSK[0,1]):
                psk.append(np.array([1,1,0]));
            else:
                psk.append(np.array([1,0,1]));
        elif (inphase[i]==PSK[0,4]):
            psk.append(np.array([0,0,1]));
        elif (inphase[i]==PSK[0,2]):
            if (quadphase[i]==PSK[0,0]):
                psk.append(np.array([0,1,0]));
            else:
                psk.append(np.array([1,0,0]));
        elif (inphase[i]==PSK[0,3]):
        # else:
            if (quadphase[i]==PSK[0,1]):
                psk.append(np.array([0,1,1]));
            else:
                psk.append(np.array([0,0,0]));
    return np.ravel(np.array(psk).astype('int'));

def qam16_decode(inp,amp):
    inphase = np.ravel(inp[0]);
    quadphase = np.ravel(inp[1]);
    a=2**0.5;
    qam16=[];
    for i in range(len(inphase)):
        
        if (inphase[i]==-3*amp):
            if (quadphase[i]==-3*amp):
                qam16.append([0,0,0,0]);
            elif (quadphase[i]==-amp):
                qam16.append([0,1,0,0]);
            elif (quadphase[i]==amp):
                qam16.append([1,0,0,0]);
            else:
                qam16.append([1,1,0,0]);
                
        elif (inphase[i]==-amp):
            if (quadphase[i]==-3*amp):
                qam16.append([0,0,0,1]);
            elif (quadphase[i]==-amp):
                qam16.append([0,1,0,1]);
            elif (quadphase[i]==amp):
                qam16.append([1,0,0,1]);
            else:
                qam16.append([1,1,0,1]);
                
        elif (inphase[i]==amp):
            if (quadphase[i]==-3*amp):
                qam16.append([0,0,1,0]);
            elif (quadphase[i]==-amp):
                qam16.append([0,1,1,0]);
            elif (quadphase[i]==amp):
                qam16.append([1,0,1,0]);
            else:
                qam16.append([1,1,1,0]);
                
        else:
            if(quadphase[i]==-3*amp):
                qam16.append([0,0,1,1]);
            elif(quadphase[i]==-amp):
                qam16.append([0,1,1,1]);
            elif(quadphase[i]==amp):
                qam16.append([1,0,1,1]);
            else:
                qam16.append([1,1,1,1]);
        
    return np.ravel(np.array(qam16)).astype(int);


def qam64_decode(inp,amp):
    inphase = np.ravel(inp[0]);
    quadphase = np.ravel(inp[1]);
    a=2**0.5;
    qam64=[];
    for i in range(len(inphase)):
        
        if (inphase[i]==-7*amp):
            if (quadphase[i]==-7*amp):
                qam64.append([0,0,0,0,0,0]);
            elif (quadphase[i]==-5*amp):
                qam64.append([0,0,1,0,0,0]);
            elif (quadphase[i]==-3*amp):
                qam64.append([0,1,0,0,0,0]);
            elif (quadphase[i]==-amp):
                qam64.append([0,1,1,0,0,0]);
            elif (quadphase[i]==amp):
                qam64.append([1,0,0,0,0,0]);
            elif (quadphase[i]==3*amp):
                qam64.append([1,0,1,0,0,0]);
            elif (quadphase[i]==5*amp):
                qam64.append([1,1,0,0,0,0]);
            else:
                qam64.append([1,1,1,0,0,0]);            
           
                
        elif (inphase[i]==-5*amp):
            if (quadphase[i]==-7*amp):
                qam64.append([0,0,0,0,0,1]);
            elif (quadphase[i]==-5*amp):
                qam64.append([0,0,1,0,0,1]);
            elif (quadphase[i]==-3*amp):
                qam64.append([0,1,0,0,0,1]);
            elif (quadphase[i]==-amp):
                qam64.append([0,1,1,0,0,1]);
            elif (quadphase[i]==amp):
                qam64.append([1,0,0,0,0,1]);
            elif (quadphase[i]==3*amp):
                qam64.append([1,0,1,0,0,1]);
            elif (quadphase[i]==5*amp):
                qam64.append([1,1,0,0,0,1]);
            else:
                qam64.append([1,1,1,0,0,1]);  
                
        elif (inphase[i]==-3*amp):
            if (quadphase[i]==-7*amp):
                qam64.append([0,0,0,0,1,0]);
            elif (quadphase[i]==-5*amp):
                qam64.append([0,0,1,0,1,0]);
            elif (quadphase[i]==-3*amp):
                qam64.append([0,1,0,0,1,0]);
            elif (quadphase[i]==-amp):
                qam64.append([0,1,1,0,1,0]);
            elif (quadphase[i]==amp):
                qam64.append([1,0,0,0,1,0]);
            elif (quadphase[i]==3*amp):
                qam64.append([1,0,1,0,1,0]);
            elif (quadphase[i]==5*amp):
                qam64.append([1,1,0,0,1,0]);
            else:
                qam64.append([1,1,1,0,1,0]);
                
        elif (inphase[i]==-amp):
            if (quadphase[i]==-7*amp):
                qam64.append([0,0,0,0,1,1]);
            elif (quadphase[i]==-5*amp):
                qam64.append([0,0,1,0,1,1]);
            elif (quadphase[i]==-3*amp):
                qam64.append([0,1,0,0,1,1]);
            elif (quadphase[i]==-amp):
                qam64.append([0,1,1,0,1,1]);
            elif (quadphase[i]==amp):
                qam64.append([1,0,0,0,1,1]);
            elif (quadphase[i]==3*amp):
                qam64.append([1,0,1,0,1,1]);
            elif (quadphase[i]==5*amp):
                qam64.append([1,1,0,0,1,1]);
            else:
                qam64.append([1,1,1,0,1,1]);
                
        elif (inphase[i]==amp):
            if (quadphase[i]==-7*amp):
                qam64.append([0,0,0,1,0,0]);
            elif (quadphase[i]==-5*amp):
                qam64.append([0,0,1,1,0,0]);
            elif (quadphase[i]==-3*amp):
                qam64.append([0,1,0,1,0,0]);
            elif (quadphase[i]==-amp):
                qam64.append([0,1,1,1,0,0]);
            elif (quadphase[i]==amp):
                qam64.append([1,0,0,1,0,0]);
            elif (quadphase[i]==3*amp):
                qam64.append([1,0,1,1,0,0]);
            elif (quadphase[i]==5*amp):
                qam64.append([1,1,0,1,0,0]);
            else:
                qam64.append([1,1,1,1,0,0]);
                
        elif (inphase[i]==3*amp):
            if (quadphase[i]==-7*amp):
                qam64.append([0,0,0,1,0,1]);
            elif (quadphase[i]==-5*amp):
                qam64.append([0,0,1,1,0,1]);
            elif (quadphase[i]==-3*amp):
                qam64.append([0,1,0,1,0,1]);
            elif (quadphase[i]==-amp):
                qam64.append([0,1,1,1,0,1]);
            elif (quadphase[i]==amp):
                qam64.append([1,0,0,1,0,1]);
            elif (quadphase[i]==3*amp):
                qam64.append([1,0,1,1,0,1]);
            elif (quadphase[i]==5*amp):
                qam64.append([1,1,0,1,0,1]);
            else:
                qam64.append([1,1,1,1,0,1]);
                
        elif (inphase[i]==5*amp):
            if (quadphase[i]==-7*amp):
                qam64.append([0,0,0,1,1,0]);
            elif (quadphase[i]==-5*amp):
                qam64.append([0,0,1,1,1,0]);
            elif (quadphase[i]==-3*amp):
                qam64.append([0,1,0,1,1,0]);
            elif (quadphase[i]==-amp):
                qam64.append([0,1,1,1,1,0]);
            elif (quadphase[i]==amp):
                qam64.append([1,0,0,1,1,0]);
            elif (quadphase[i]==3*amp):
                qam64.append([1,0,1,1,1,0]);
            elif (quadphase[i]==5*amp):
                qam64.append([1,1,0,1,1,0]);
            else:
                qam64.append([1,1,1,1,1,0]);
                
        elif (inphase[i]==7*amp):
            if (quadphase[i]==-7*amp):
                qam64.append([0,0,0,1,1,1]);
            elif (quadphase[i]==-5*amp):
                qam64.append([0,0,1,1,1,1]);
            elif (quadphase[i]==-3*amp):
                qam64.append([0,1,0,1,1,1]);
            elif (quadphase[i]==-amp):
                qam64.append([0,1,1,1,1,1]);
            elif (quadphase[i]==amp):
                qam64.append([1,0,0,1,1,1]);
            elif (quadphase[i]==3*amp):
                qam64.append([1,0,1,1,1,1]);
            elif (quadphase[i]==5*amp):
                qam64.append([1,1,0,1,1,1]);
            else:
                qam64.append([1,1,1,1,1,1]);

        
    return np.ravel(np.array(qam64)).astype(int);

#############################################################################################################################

### insertion on an image and image manipulation
import matplotlib.image as mpimg

def img2bin(image):
    q = np.ravel(image);
    bits=[];
    for i in range(len(q)):
        b = np.binary_repr(int(q[i]), width=8)
        for i in range(8):
            bits.append(int(b[i]));
    return bits;


def image_bitstream(image):
    
    N_iter = len(image)
    SNR_db = np.arange(12,13,1);
    pe_bpsk=[];
    pe_qpsk=[];
    pe_psk=[];
    pe_qam16=[];
    pe_qam64=[];
    for i in range(len(SNR_db)):
        ########----------PArameters defined--------------------------##########################################
        Energy = 10**(SNR_db[i]/10);# energy per symbol(least energy per sym.)
        
        a= 2**0.5;
        b = np.cos(np.pi/4);
        
        #######--------------------constellation---------------------##########################################
        ## bitstearm
        bitstream= image;
        
        
        ###################------------------------------------BPSK---------------------##########################
        BPSK_cons= np.array([-1,1]);
        
        BPSK_Amp = (Energy**0.5);
        BPSK = BPSK_Amp* BPSK_cons;

        map_err_bpsk=map_bpsk(bitstream,BPSK);

        bpsk_received=bpsk_decision(map_err_bpsk,BPSK);

        bpsk_rec_symbols=bpsk_decode(bpsk_received);

        pe_bpsk.append((len(image)-np.sum(np.array(bpsk_rec_symbols[0:len(image)]==bitstream).astype(int)))/len(image));

        del map_err_bpsk;
        del bpsk_received;
        

        # ###################------------------------------------QPSK---------------------##########################

        qpsk_amp=(8*Energy/20)**0.5;
        QPSK= (qpsk_amp)*np.array([[1,0,-1,0,1/b,-1/b,-1/b,1/b],[0,1,0,-1,1/b,1/b,-1/b,-1/b]])# let it be a 8-pt QPSK;

        map_err_qpsk=map_qpsk(bitstream,QPSK);

        inp_error_qpsk= map_err_qpsk[0]+ 0.707*np.random.randn(1,len(map_err_qpsk[0]));
        quad_error_qpsk= map_err_qpsk[1]+ 0.707*np.random.randn(1,len(map_err_qpsk[0]));

        error = np.array([inp_error_qpsk,quad_error_qpsk]);

        qpsk_received=qpsk_decision(error,qpsk_amp,QPSK);
        
        qpsk_rec_symbols=qpsk_decode(qpsk_received,QPSK);

        pe_qpsk.append((len(image)-np.sum(np.array(qpsk_rec_symbols[0:len(image)]==bitstream).astype(int)))/len(image));


        del map_err_qpsk;
        del qpsk_received;
        

        ###################------------------------------------PSK---------------------##########################
        
        psk_amp=(Energy)**0.5;
        
        
        PSK = (psk_amp)*np.array([[1,b,0,-b,-1,-b,0,b],[0,b,1,b,0,-b,-1,-b]]);

        map_err_psk=map_psk(bitstream,PSK);
        
        map_err_psk[0]= map_err_psk[0]+ 0.707*np.random.randn(1,len(map_err_psk[0]));
        map_err_psk[1]= map_err_psk[1]+ 0.707*np.random.randn(1,len(map_err_psk[0]));

        psk_received=psk_decision(map_err_psk,psk_amp,PSK);

        psk_rec_symbols=psk_decode(psk_received,psk_amp,PSK);

        pe_psk.append((len(image)-np.sum(np.array(psk_rec_symbols[0:len(image)]==bitstream).astype(int)))/len(image));

        
        del map_err_psk;
        del psk_received;
        

        ##################------------------------------------QAM16---------------------##########################
        X =np.array(np.linspace(-3,3,4),dtype=int);
        inphase16=[];
        quadphase16=[];
        for i in range(len(X)):
            inphase16.append(X);
            quadphase16.append(X[i]*np.ones(len(X)));
        
        qam16_amp=(Energy/10)**0.5;
        QAM16= (qam16_amp)*np.array([np.ravel(np.array(inphase16)),np.ravel(np.array(quadphase16))]);

        map_err_qam16=map_qam16(bitstream,QAM16);

        map_err_qam16[0]= map_err_qam16[0]+ 0.707*np.random.randn(1,len(map_err_qam16[0]));
        map_err_qam16[1]= map_err_qam16[1]+ 0.707*np.random.randn(1,len(map_err_qam16[0]));

        qam16_received=QAM16_decision(map_err_qam16,qam16_amp,QAM16);

        qam16_rec_symbols=qam16_decode(qam16_received,qam16_amp);
        
        pe_qam16.append((len(image)-np.sum(np.array(qam16_rec_symbols[0:len(image)]==bitstream).astype(int)))/len(image));


        del map_err_qam16;
        del qam16_received;
        


        # ###################------------------------------------QAM64---------------------##########################   
        X64 =np.array(np.linspace(-7,7,8),dtype=int);
        inphase64=[];
        quadphase64=[];
        for i in range(len(X64)):
            inphase64.append(X64);
            quadphase64.append(X64[i]*np.ones(len(X64)));
        
        qam64_amp=(Energy/42)**0.5;
        QAM64= (qam64_amp)*np.array([np.ravel(np.array(inphase64)),np.ravel(np.array(quadphase64))]);

        map_err_qam64=map_qam64(bitstream,QAM64);

        map_err_qam64[0]= map_err_qam64[0]+ 0.707*np.random.randn(1,len(map_err_qam64[0]));
        map_err_qam64[1]= map_err_qam64[1]+ 0.707*np.random.randn(1,len(map_err_qam64[0]));

        qam64_received=QAM64_decision(map_err_qam64,qam64_amp,QAM64);

        qam64_rec_symbols=qam64_decode(qam64_received,qam64_amp);

        pe_qam64.append((len(image)-np.sum(np.array(qam64_rec_symbols[0:len(image)]==bitstream).astype(int)))/len(image));

        del map_err_qam64;
        del qam64_received;
        
        
        
    bpsk_err= np.array(pe_bpsk);
    qpsk_err= np.array(pe_qpsk);
    psk_err= np.array(pe_psk);
    qam16_err= np.array(pe_qam16);
    qam64_err= np.array(pe_qam64);

    received_image_bits = np.concatenate([[bpsk_rec_symbols],[qpsk_rec_symbols],[psk_rec_symbols],[qam16_rec_symbols],[qam64_rec_symbols]]);
    error = np.concatenate([[bpsk_err],[qpsk_err],[psk_err],[qam16_err],[qam64_err]]);

    del bpsk_rec_symbols;
    del qpsk_rec_symbols;
    del psk_rec_symbols;
    del qam16_rec_symbols;
    del qam64_rec_symbols;

    

    return received_image_bits,error;


#####################################################################################################################################
# Receiver side image conversion
def coverting_back_img(received_image,shape):
    ###############-------------BPSK--------------------############

    BPSK_img = received_image[0];
    net_len = shape[0]*shape[1]*shape[2]*8;

    net_bit_stream = BPSK_img[0:net_len];


    bpsk_reshape = net_bit_stream.reshape(int(len(net_bit_stream)/8),8);

    l,m = bpsk_reshape.shape;

    print(l,'  ',m);

    bpsk_vector=np.zeros(l);

    for i in range(l):
        for j in range(m):
            bpsk_vector[i]= bpsk_vector[i] + bpsk_reshape[i][7-j]*(2**j) 
    
    bpsk_final_image = bpsk_vector.reshape(shape[0],shape[1],shape[2]);

    del BPSK_img;
    del net_bit_stream;
    del bpsk_vector;


        ##########################-------------QPSK--------------------#########################

    QPSK_img = received_image[1];
    # net_len = shape[0]*shape[1]*shape[2]*8;

    net_bit_stream = QPSK_img[0:net_len];

    qpsk_reshape = net_bit_stream.reshape(int(len(net_bit_stream)/8),8);

    l,m = qpsk_reshape.shape;

    qpsk_vector=np.zeros(l);

    for i in range(l):
        for j in range(m):
            qpsk_vector[i]= qpsk_vector[i] + qpsk_reshape[i][7-j]*(2**j) 
    
    qpsk_final_image = qpsk_vector.reshape(shape[0],shape[1],shape[2]);

    del QPSK_img;
    del net_bit_stream;
    del qpsk_vector;


    ##################-------------PSK--------------------##############################

    PSK_img = received_image[2];
    # net_len = shape[0]*shape[1]*shape[2]*8;

    net_bit_stream = PSK_img[0:net_len];

    psk_reshape = net_bit_stream.reshape(int(len(net_bit_stream)/8),8);

    l,m = qpsk_reshape.shape;

    psk_vector=np.zeros(l);

    for i in range(l):
        for j in range(m):
            psk_vector[i]= psk_vector[i] + psk_reshape[i][7-j]*(2**j) 
    
    psk_final_image = psk_vector.reshape(shape[0],shape[1],shape[2]);

    del PSK_img;
    del net_bit_stream;
    del psk_vector;


    ###########################-------------QAM16--------------------################################

    QAM16_img = received_image[3];
    # net_len = shape[0]*shape[1]*shape[2]*8;

    net_bit_stream = QAM16_img[0:net_len];

    qam16_reshape = net_bit_stream.reshape(int(len(net_bit_stream)/8),8);

    l,m = qam16_reshape.shape;

    qam16_vector=np.zeros(l);

    for i in range(l):
        for j in range(m):
            qam16_vector[i]= qam16_vector[i] + qam16_reshape[i][7-j]*(2**j) 
    
    qam16_final_image = qam16_vector.reshape(shape[0],shape[1],shape[2]);

    del QAM16_img;
    del net_bit_stream;
    del qam16_vector;


    #####################-------------QAM64--------------------############################

    QAM64_img = received_image[4];
    # net_len = shape[0]*shape[1]*shape[2]*8;

    net_bit_stream = QAM64_img[0:net_len];

    qam64_reshape = net_bit_stream.reshape(int(len(net_bit_stream)/8),8);

    l,m = qam64_reshape.shape;

    qam64_vector=np.zeros(l);

    for i in range(l):
        for j in range(m):
            qam64_vector[i]= qam64_vector[i] + qam64_reshape[i][7-j]*(2**j) 
    
    qam64_final_image = qam64_vector.reshape(shape[0],shape[1],shape[2]);

    del QAM64_img;
    del net_bit_stream;
    del qam64_vector;

    #################-------------RESULTS--------------------##########################

    Received_Image=np.concatenate([[bpsk_final_image],[qpsk_final_image],[psk_final_image],[qam16_final_image],[qam64_final_image]]);
    return Received_Image




################################################################################################################################

# Transmission and reception of image1

# The result is present as image 1 for transmission SNR =5db;(results for 15 db and 2db are also show in the readme section.)

panda=mpimg.imread('website.jpg')
bitstream=img2bin(panda);
Image_intermediate,error=image_bitstream(bitstream);
size =np.array(panda.shape);
received_img=coverting_back_img(Image_intermediate,size);
# panda.shape  (510, 1020, 3)
plt.figure(figsize=(20,20))
plt.subplot(3,2,1)
plt.imshow((received_img[0]).astype(int))
plt.xlabel('BPSK Transmission at SNR = 5db, BER = 0.00593786')
plt.subplot(3,2,2)
plt.imshow((received_img[1]).astype(int))
plt.xlabel('8 Two level QAM Transmission at SNR = 5db, BER = 0.1603932')
plt.subplot(3,2,3)
plt.imshow((received_img[2]).astype(int))
plt.xlabel('PSK Transmission at SNR = 5db, BER = 0.11761462')
plt.subplot(3,2,4)
plt.imshow((received_img[3]).astype(int))
plt.xlabel('QAM16 Transmission at SNR = 5db, BER = 0.19476235')
plt.subplot(3,2,5)
plt.imshow((received_img[4]).astype(int))
plt.xlabel('QAM64 Transmission at SNR = 5db, BER = 0.29068627')
plt.subplot(3,2,6)
plt.imshow(panda)
plt.xlabel('Original panda Image')

################################################################################################################################






