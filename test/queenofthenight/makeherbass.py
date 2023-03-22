import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
import rustlets
import scipy.io.wavfile as wvf
import cmath

def main():
    FILENAME = "queenaudio.wav"
    hz, data = wvf.read(FILENAME)

    log_steps = 16

    num_seconds = 10
    start = 7

    octave_shift = 3

    signal = data[:,0] + data[:,1]
    signal = signal[int(hz * start): int(hz*(num_seconds+start))]
    times = np.arange(0,len(signal)/hz,1/hz)

    print("Audio Sample Rate: ", hz)

    plt.plot(times,signal,label="original")
    plt.show()

    ## do the morlet transform
    print("Computing CWT...")

    cwt_list,scales_list = rustlets.cwt_morlet_ext(signal,hz,log_steps, False)
    scales = np.array(scales_list)
    freqs = 1 / (scales * rustlets.morlet_wavelength(2 * np.pi))
    cwtmat = np.array(cwt_list,dtype=np.complex64).reshape((len(scales),-1))[:,:len(times)]

    print("Done.")

    rescale = 0.5

    cwtabs = np.abs(cwtmat)
    cwtphase = np.unwrap(np.arctan2(cwtmat.imag, cwtmat.real))

    cscale = cwtabs * np.exp(cwtphase * 1j * rescale)
    scales = scales / rescale


    print("Computing iCWT on transformed CWT matrix...")
    f_rec = np.array(rustlets.icwt_morlet(cscale.flatten(),scales,times))
    print("Done. Output in bassbitch.wav")

    #plt.plot(times,signal,label="original")
    plt.plot(times,f_rec,label="recovered")
    plt.legend()
    plt.show()
    #print(np.sqrt(sum((f_rec - signal)**2)/len(signal)))

    wvf.write("bassbitch.wav", hz, f_rec.astype(np.int16))


if __name__ == "__main__":
    main()
    #testAnalytic()
