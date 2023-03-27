import numpy as np
from scipy import signal, fft
import matplotlib.pyplot as plt
import rustlets
import scipy.io.wavfile as wvf
from skimage.restoration import unwrap_phase
import cmath
from spleeter.separator import Separator
from spleeter.audio.adapter import AudioAdapter
import sys





def signal_from_wav(file, start, duration):
    hz, data = wvf.read(file)
    signal = data[:,0] + data[:,1]
    signal = signal[int(hz * start): int(hz*(duration+start))]
    return hz, signal

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
    return hz, signal, accompaniment

def cwt(signal,hz,log_steps):
    cwt_list,scales_list = rustlets.cwt_morlet_ext(signal,hz,log_steps,False)
    scales = np.array(scales_list)
    cwtmat = np.array(cwt_list,dtype=np.complex64)
    return cwtmat, scales

def unwrap_rows(p1d):
    """unwrap phase of flattened 2d array, row by row where each row is of length n"""
    for i,row in enumerate(p1d):
        p1d[i] = np.unwrap(row)
    return p1d

def get_scaled_cwt(cwtmat,rescale):
    #cwtphase = unwrap_rows(np.arctan2(cwtmat.imag, cwtmat.real))
    cwtphase = unwrap_rows(np.arctan2(cwtmat.imag, cwtmat.real))
    plt.pcolormesh(cwtphase)
    plt.show()
    return (np.abs(cwtmat) * np.exp(cwtphase * 1j * rescale))



def main():
    """
    Usage: freqshift.py FILENAME start, duration, steps_per_octave, cwtdisplay(y or n), frequency_rescale
    """
    SPLITDIR = "split/"
    file = sys.argv[1]
    start = float(sys.argv[2])
    duration = float(sys.argv[3])
    log_steps = int(sys.argv[4])
    rescale = float(sys.argv[6])
    
    hz = 44100
    signal = np.array([])
    accompaniment = np.array([])

    path = SPLITDIR + file.split(".")[0] + "/"
    print("Looking for previously split files in \"" +path + "\"...")
    try :
        vname = path + "vocals.wav"
        aname = path + "accompaniment.wav"
        hz, signal = signal_from_wav(vname, start, duration)
        _, accompaniment = signal_from_wav(aname, start, duration)
    except :
        print("Previously split files not found")
        print("Splitting file...")
        hz, signal, accompaniment = signal_from_wav_spleeter(file, start, duration)
        print("Done.")

    times = np.arange(0,len(signal)/hz,1/hz)

    #plt.plot(times,signal)
    #plt.show()


    print("Using Audio Sample Rate: ", hz)
    print("Number Samples: ", len(signal))
    print("Number Scales Per Octave: ", log_steps)

    #plt.plot(times,signal,label="original")
    #plt.show()

    ## do the morlet transform
    
    print("Computing CWT...")

    cwtmat, scales = cwt(signal,hz,log_steps)
    cwtmat = cwtmat.reshape((len(scales),-1))[:,:len(times)]

    print("Done.")

    if sys.argv[5] == 'y':
        freqs = 1 / (scales * rustlets.morlet_wavelength(2 * np.pi))
        print("Working on displaying CWT visualizaton...")
        plt.pcolormesh(times,freqs, np.abs(cwtmat), cmap='viridis', shading='gouraud')
        plt.show()


    scales = scales / rescale
    

    
    print("Computing iCWT on transformed (" + str(rescale) + ") CWT matrix...")
    f_rec = np.array(rustlets.icwt_morlet(get_scaled_cwt(cwtmat,rescale).flatten(),scales,times))
    print("Done. Recombined output in shifted.wav")
    
    
    #f_rec = np.array(rustlets.freqshift(signal,times,hz,log_steps,rescale))

    f_rec = f_rec * np.max(signal) / np.max(f_rec)
     
    #plt.plot(times,signal,label="original")
    #plt.plot(times,f_rec,label="/ecovered")
    
    #plt.legend()
    #plt.show()
    #print(np.sqrt(sum((f_rec - signal)**2)/len(signal)))

    wvf.write("shifted.wav", hz, f_rec.astype(np.float32))


if __name__ == "__main__":
    main()
    #testAnalytic()
