import numpy as np
from scipy import fft, signal
import matplotlib.pyplot as plt
import matplotlib
import rustlets


def main():
    hz=10180
    length = 1.0
    times = np.arange(0,length,1/hz)
    samples = signal.chirp(times, f0=2200, f1=1500, t1=1, method='linear')
    samples = samples + signal.chirp(times, f0=600, f1=1100, t1=1, method='linear') + signal.gausspulse(times,fc=700)
    #samples = samples + signal.chirp(times, f0=20000, f1=8000, t1=1, method='linear')


    cwt_list,scales_list = rustlets.cwt_morlet(samples,hz,10)

    scales = np.array(scales_list)
    freqs = 1 / (scales * rustlets.morlet_wavelength(2 * np.pi))
    cwtmat = np.array(cwt_list,dtype=np.complex64).reshape((len(scales),-1))[:,:len(times)]

    plt.plot(times,samples,label="original")
    plt.show()

    #plt.pcolormesh(times,freqs, np.abs(cwtmat), cmap='viridis', shading='gouraud')
    #plt.show()


    f_rec = np.array(rustlets.icwt_morlet(cwt_list,scales,times))

    plt.plot(times,samples,label="original")
    plt.plot(times,f_rec,label="recovered")
    plt.legend()
    plt.show()

    plt.plot(times,f_rec/samples,label="ratio")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
    #testAnalytic()
