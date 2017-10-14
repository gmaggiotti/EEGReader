#!/usr/bin/python
# -*- coding: utf-8 -*-
import pygame, sys
from numpy import *
from pygame import *
import scipy
from mindwave.pyeeg import bin_power
from mindwave.parser import ThinkGearParser, TimeSeriesRecorder
from mindwave.bluetooth_headset import connect_magic, connect_bluetooth_addr
from mindwave.bluetooth_headset import BluetoothError
from example_startup import mindwave_startup
from Talk import Talk
from nnet.A4NN import A4NN

description = """Pygame Example
"""


socket, args = mindwave_startup(description=description)
recorder = TimeSeriesRecorder()
parser = ThinkGearParser(recorders= [recorder])

def main():
    pygame.init()
    t = Talk()
    net = A4NN()
    net.train()
    dataset = [30,30,30,30,30,30,30,30,30,30,30,30,30,14,14,14,14,14,14,14,14,14,14,14,14,11,11,11,11,11,11,11,11,11,11,11,10,10,10,10,10,12,14,10]
    cont=0
    result=0

    fpsClock= pygame.time.Clock()

    window = pygame.display.set_mode((1280,720))
    pygame.display.set_caption("Mindwave Viewer")


    blackColor = pygame.Color(0,0,0)
    redColor = pygame.Color(255,0,0)
    greenColor = pygame.Color(0,255,0)
    deltaColor = pygame.Color(0,0,128)
    thetaColor = pygame.Color(238,130,238)
    alphaColor = greenColor
    betaColor = pygame.Color(255,255,0)
    gammaColor = redColor


    background_img = pygame.image.load("pygame_background.png")


    font = pygame.font.Font("freesansbold.ttf", 20)
    raw_eeg = True
    spectra = []
    iteration = 0

    meditation_img = font.render("Meditation", False, redColor)
    attention_img = font.render("Attention", False, redColor)

    record_baseline = False
    quit = False
    while quit is False:
        try:
            data = socket.recv(10000)
            parser.feed(data)
        except BluetoothError:
            pass
        window.blit(background_img,(0,0))
        if len(recorder.attention)>0:
            iteration+=1
            flen = 50
            if len(recorder.raw)>=500:
                spectrum, relative_spectrum = bin_power(recorder.raw[-512*3:], range(flen),512)
                spectra.append(array(relative_spectrum))
                if len(spectra)>30:
                    spectra.pop(0)

                spectrum = mean(array(spectra),axis=0)
                for i in range (flen-1):
                    value = float(spectrum[i]*1000)
                    if i<3:
                        color = deltaColor  #blue
                    elif i<8:
                        color = thetaColor  #violet
                    elif i<13:
                        color = alphaColor  #green
                    elif i<30:
                        color = betaColor  #yellow
                    else:
                        color = gammaColor  #red
                    pygame.draw.rect(window, color, (25+i*10, 400-value, 5, value))
            else:
                pass
            pygame.draw.circle(window, redColor, (800,200), int(recorder.attention[-1]/2))
            pygame.draw.circle(window, greenColor, (800,200), 60/2,1)
            pygame.draw.circle(window, greenColor, (800,200), 100/2,1)
            window.blit(attention_img, (760,260))
            meditation = int(recorder.meditation[-1]/2)
            print meditation
            dataset.extend([meditation])
            del dataset[0]
            if(cont>=22):
                result = net.predict(dataset)
                cont=0
            cont += 1

            if(result >= 0.999 ):
                print "Prediction: "+str(result)
                t.sayYes()
                result=0
            pygame.draw.circle(window, redColor, (700,200), meditation)
            pygame.draw.circle(window, greenColor, (700,200), 60/2, 1)
            pygame.draw.circle(window, greenColor, (700,200), 100/2, 1)

            window.blit(meditation_img, (600,260))

            """if len(parser.current_vector)>7:
                m = max(p.current_vector)
                for i in range(7):
                    if m == 0:
                        value = 0
                    else:
                        value = p.current_vector[i] *100.0/m
                    pygame.draw.rect(window, redColor, (600+i*30,450-value, 6,value))"""
            if raw_eeg:
                lv = 0
                for i,value in enumerate(recorder.raw[-1000:]):
                    v = value/ 2.0
                    pygame.draw.line(window, redColor, (i+25, 500-lv), (i+25, 500-v))
                    lv = v
        else:
            img = font.render("Not receiving any data from mindwave...", False, redColor)
            window.blit(img,(100,100))
            pass

        for event in pygame.event.get():
            if event.type==QUIT:
                quit = True
            if event.type==KEYDOWN:
                if event.key==K_ESCAPE:
                    quit = True
        pygame.display.update()
        fpsClock.tick(12)

if __name__ == '__main__':
    try:
        main()
    finally:
        pygame.quit()
