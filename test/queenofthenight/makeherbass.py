import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from numericalmethods import *
import rustlets
import scipy.io.wavfile as wvf

def fwavelength(w_0):
    return 4 * np.pi / (w_0 + np.sqrt(2 + w_0**2))

def main():
    FILENAME = "vocals.wav"
    hz, data = wvf.read(FILENAME)

    log_steps = 10

    num_seconds = 0.5
    start = 7.5

    octave_shift = 3

    signal = data[:,0] + data[:,1]
    signal = signal[int(hz * start): int(hz*(num_seconds+start))]
    times = np.arange(0,len(signal)/hz,1/hz)

    print("Audio Sample Rate: ", hz)

    #plt.plot(times,signal,label="original")
    #plt.show()

    ## do the morlet transform
    print("Computing CWT...")
    cwt_list,scales_list = rustlets.cwt_morlet(signal,hz,log_steps,True)

    print("Done.")

    scales = np.array(scales_list)
    freqs = 1 / (scales * fwavelength(2 * np.pi))
    cwtmat = np.array(cwt_list,dtype=np.complex64).reshape((len(scales),-1))[:,:len(times)]

    newcwtmat = np.zeros(cwtmat.shape,dtype=np.complex64)
    for i in range(octave_shift * log_steps,cwtmat.shape[0]):
        newcwtmat[i] = cwtmat[i-(octave_shift * log_steps)]

    #plt.pcolormesh(times,freqs, np.abs(newcwtmat), cmap='viridis', shading='gouraud')
    #plt.show()
##

    newcwtflat = newcwtmat.flatten()

    print("Computing iCWT...")
    f_rec = np.array(rustlets.icwt_morlet(newcwtflat,scales,times)) * np.pi
    print("Done.")

    ratio = max(f_rec)/max(signal)
    f_rec = f_rec / ratio

    plt.plot(times,signal,label="original")
    plt.plot(times,f_rec,label="recovered")
    plt.legend()
    plt.show()

    wvf.write("bassbitch.wav", hz, f_rec.astype(np.int32))
    #wvf.write("bassbitch.wav", hz, signal)


if __name__ == "__main__":
    main()
    #testAnalytic()
