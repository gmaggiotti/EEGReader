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
from nnet.AlphaA4NN import AlphaA4NN
from nnet.WinkA4NN import WinkA4NN

description = """Pygame Example
"""


socket, args = mindwave_startup(description=description)
recorder = TimeSeriesRecorder()
parser = ThinkGearParser(recorders= [recorder])

def main():
    pygame.init()
    t = Talk()
    med_net = A4NN()
    print ""
    print "meditation signal training...."
    print ""
    med_net.train()
    alpha_net = AlphaA4NN()
    print ""
    print "Alpha signal training...."
    print ""
    alpha_net.train()
    wink_net = WinkA4NN()
    print ""
    print "Wink signal training...."
    print ""
    wink_net.train()

    meditation_dataset = [30,30,30,30,30,30,30,30,30,30,30,30,30,14,14,14,14,14,14,14,14,14,14,14,14,11,11,11,11,11,11,11,11,11,11,11,10,10,10,10,10,12,14,10]
    alpha_dataset = [21,21,21,22,22,22,22,22,22,22,22,22,22,22,21,21,21,21,21,21,21,21,21,21,21,21,20,20,20,20,20,20,19,19,19,19,20,20,20,19,19,19,19,18]
    raw_dataset = [87, 87, 97, 97, 97, 97, 97, 97, 97, 97, 97, 97, 107, 107, 107, 107, 107, 107, 167, 247, 55, 29, 59,
                    59, 29, 29, 29, 129, 129, 129, 129, 307, 309, 92, 37, 60, 72, 75, 97]

    cont=0
    result_med= result_alpha = 0.0
    alpha=0

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
                        if i ==12:
                            alpha = int(value)
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
            #print "med: "+str(meditation)+" alpha: "+str(alpha)

            meditation_dataset.extend([meditation])
            del meditation_dataset[0]

            alpha_dataset.extend([alpha])
            del alpha_dataset[0]

            max = recorder.raw[-100:].max()
            raw_dataset.extend([max])
            del raw_dataset[0]

            #predict wink, apha and med
            if cont>=22:
                result_med = med_net.predict(meditation_dataset)
                result_alpha = alpha_net.predict(alpha_dataset)
                cont=0
            cont += 1

            if wink_net.predict(raw_dataset) > 0.8 :
                print "prediction Wink"
                print ""
                #t.say("wink")
                raw_dataset = [0]*39
                meditation_dataset =alpha_dataset = [0]*44
            elif result_med >= 0.99 :
                print "prediction med: "+str(result_med)
                print ""
                #t.say("med")
                result_med=0
                meditation_dataset =alpha_dataset = [0]*44
            elif result_alpha > 0.80 :
                print "prediction alpha: "+str(result_alpha)
                print ""
                #t.say("alpha")
                result_alpha = 0
                meditation_dataset =alpha_dataset = [0]*44


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
