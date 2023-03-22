import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
import rustlets
import scipy.io.wavfile as wvf
import cmath
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import sys





def signal_from_wav(file, start, duration):
    hz, data = wvf.read(file)
    signal = data[:,0] + data[:,1]
    signal = signal[int(hz * start): int(hz*(duration+start))]
    times = np.arange(0,len(signal)/hz,1/hz)
    return hz, signal, times

def signal_from_wav_spleeter(file, start, duration, keepLR=False):
    separator = Separator('spleeter:2stems')

    audio_loader = AudioAdapter.default()
    hz = 44100
    waveform, _ = audio_loader.load(file, sample_rate=hz)

    # Perform the separation :
    prediction = separator.separate(waveform)

    signal = prediction["vocals"]
    accompaniment = prediction["accompaniment"]
    if not keepLR:
        signal = signal[:,0] + signal[:,1]
        accompaniment = accompaniment[:,0] + accompaniment[:,1]
    signal = signal[int(hz * start): int(hz*(duration+start))]
    print(signal)
    accompaniment = accompaniment[int(hz * start): int(hz*(duration+start))]
    times = np.arange(0,len(signal)/hz,1/hz)
    return hz, signal, times, accompaniment

def main():
    """
    Usage: freqshift.py FILENAME start, duration, steps_per_octave, cwtdisplay(y or n), frequency_rescale
    """
    print("Splitting file...")
    hz, signal, times, accompaniment = signal_from_wav_spleeter(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
    print("Done.")

    log_steps = int(sys.argv[4])

    print("Using Audio Sample Rate: ", hz)
    print("Number Samples: ", len(signal))
    print("Number Scales Per Octave: ", log_steps)

    #plt.plot(times,signal,label="original")
    #plt.show()

    ## do the morlet transform
    print("Computing CWT...")

    cwt_list,scales_list = rustlets.cwt_morlet_ext(signal,hz,log_steps, False)
    scales = np.array(scales_list)
    freqs = 1 / (scales * rustlets.morlet_wavelength(2 * np.pi))
    cwtmat = np.array(cwt_list,dtype=np.complex64)

    print("Done.")

    if sys.argv[5] == 'y':
        cwtmat2d = cwtmat.reshape((len(scales),-1))[:,:len(times)]
        print("Working on displaying CWT visualizaton...")
        plt.pcolormesh(times,freqs, np.abs(cwtmat2d), cmap='viridis', shading='gouraud')
        plt.show()

    rescale = float(sys.argv[6])

    cwtabs = np.abs(cwtmat)
    cwtphase = np.unwrap(np.arctan2(cwtmat.imag, cwtmat.real))

    cscale = cwtabs * np.exp(cwtphase * 1j * rescale)
    scales = scales / rescale


    print("Computing iCWT on transformed (" + str(rescale) + ") CWT matrix...")
    f_rec = np.array(rustlets.icwt_morlet(cscale,scales,times))
    print("Done. Recombined output in shifted.wav")

    #plt.plot(times,signal,label="original")
    #plt.plot(times,f_rec,label="/ecovered")
    
    #plt.legend()
    #plt.show()
    #print(np.sqrt(sum((f_rec - signal)**2)/len(signal)))

    wvf.write("shifted.wav", hz, (f_rec + accompaniment).astype(np.int16))


if __name__ == "__main__":
    main()
    #testAnalytic()
