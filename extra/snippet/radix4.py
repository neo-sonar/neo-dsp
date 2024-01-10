'''
Radix-4 DIT,  Radix-4 DIF,  Radix-2 DIT,  Radix-2 DIF FFTs
John Bryan,  2017
Python 2.7.3
'''

# https://www.theradixpoint.com
# https://www.theradixpoint.com/radix4/r4.html
# https://www.theradixpoint.com/radix4/radix4_in_python.html

import numpy as np
import time
import matplotlib.pyplot as plt

np.set_printoptions(precision=4, suppress=1)


def swap(xarray, i, j):
    '''
    swap
    '''
    temp = xarray[i]
    xarray[i] = xarray[j]
    xarray[j] = temp
    return None


def digitreversal(x, radix, log2length, length):
    '''
    digitreversal
    '''
    assert x.shape[0] == length
    if log2length % 2 == 0:
        n1 = int(np.sqrt(length))  # seed table size
    else:
        n1 = int(np.sqrt(int(length/radix)))
    # algorithm 2,  compute seed table
    reverse = np.zeros(n1, dtype=int)
    reverse[1] = int(length/radix)
    for jvar in range(1, radix):
        reverse[jvar] = reverse[jvar-1]+reverse[1]
        for i in range(1, int(n1/radix)):
            reverse[radix*i] = int(reverse[i]/radix)
            for jvar in range(1, radix):
                reverse[int(radix*i)+jvar] = reverse[int(radix*i)] + \
                    reverse[jvar]
    # algorithm 1
    for i in range(0, n1-1):
        for j in range(i+1, n1):
            u = i+reverse[j]
            v = j+reverse[i]
            swap(x, u, v)
            if log2length % 2 == 1:
                for z in range(1, radix):
                    uu = i+reverse[j]+(z*n1)
                    vv = j+reverse[i]+(z*n1)
                    swap(x, uu, vv)
    return x


def dif_fft4(x, twiddle, log2length):
    '''
    radix-4 dif fft
    '''
    length = np.power(4, log2length)
    tss = 1
    krange = int(float(length)/4.)
    block = 1
    base = 0
    for w in range(0, log2length):
        for h in range(0, block):
            for k in range(0, krange):
                # butterfly
                offset = int(length/4)
                a = base+k
                b = base+k+offset
                c = base+k+(2*offset)
                d = base+k+(3*offset)
                apc = x[a]+x[c]
                bpd = x[b]+x[d]
                amc = x[a]-x[c]
                bmd = x[b]-x[d]
                x[a] = apc+bpd
                if k == 0:
                    x[b] = amc-(1j*bmd)
                    x[c] = apc-bpd
                    x[d] = amc+(1j*bmd)
                else:
                    r1 = twiddle[k*tss]
                    r2 = twiddle[2*k*tss]
                    r3 = twiddle[3*k*tss]
                    x[b] = (amc-(1j*bmd))*r1
                    x[c] = (apc-bpd)*r2
                    x[d] = (amc+(1j*bmd))*r3
            base = base+(4*krange)
        block = block*4
        length = float(length)/4.
        krange = int(float(krange)/4.)
        base = 0
        tss = int(tss*4)
    return x


def fft4(x, twiddles, log2length):
    '''
    radix-4 dit fft
    '''
    nvar = 4
    tss = np.power(4, log2length-1)
    krange = 1
    block = int(x.size/4)
    base = 0
    for w in range(0, log2length):
        for zvar in range(0, int(block)):
            for k in range(0, krange):
                # butterfly
                offset = nvar/4
                avar = base+k
                bvar = base+k+offset
                cvar = base+k+(2*offset)
                dvar = base+k+(3*offset)
                if k == 0:
                    xbr1 = x[int(bvar)]
                    xcr2 = x[int(cvar)]
                    xdr3 = x[int(dvar)]
                else:
                    r1var = twiddles[int(k*tss)]
                    r2var = twiddles[int(2*k*tss)]
                    r3var = twiddles[int(3*k*tss)]
                    xbr1 = (x[int(bvar)]*r1var)
                    xcr2 = (x[int(cvar)]*r2var)
                    xdr3 = (x[int(dvar)]*r3var)
                evar = x[int(avar)]+xcr2
                fvar = x[int(avar)]-xcr2
                gvar = xbr1+xdr3
                h = xbr1-xdr3
                j_h = 1j*h
                x[int(avar)] = evar+gvar
                x[int(bvar)] = fvar-j_h
                x[int(cvar)] = -gvar+evar
                x[int(dvar)] = j_h+fvar
            base = base+(4*krange)
        block = block/4
        nvar = 4*nvar
        krange = 4*krange
        base = 0
        tss = float(tss)/4.
    return x


def dif_fft0(xarray, twiddle, log2length):
    '''
    radix-2 dif
    '''
    b_p = 1
    nvar_p = xarray.size
    twiddle_step_size = 1
    for pvar in range(0,  log2length):           # pass loop
        nvar_pp = nvar_p//2
        base_e = 0
        for bvar in range(0,  b_p):       # block loop
            base_o = base_e+nvar_pp
            for nvar in range(0,  int(nvar_pp)):   # butterfly loop
                evar = xarray[int(base_e+nvar)]+xarray[int(base_o+nvar)]
                if nvar == 0:
                    ovar = xarray[int(base_e+nvar)]-xarray[int(base_o+nvar)]
                else:
                    twiddle_factor = nvar*twiddle_step_size
                    ovar = (xarray[int(base_e+nvar)]
                            - xarray[int(base_o+nvar)])*twiddle[twiddle_factor]
                xarray[int(base_e+nvar)] = evar
                xarray[int(base_o+nvar)] = ovar
            base_e = base_e+nvar_p
        b_p = b_p*2
        nvar_p = nvar_p/2
        twiddle_step_size = 2*twiddle_step_size
    return xarray


def fft2(x, twiddle, log2length):
    '''
    radix-2 dit
    '''
    nvar = x.size
    b_p = nvar/2
    nvar_p = 2
    twiddle_step_size = nvar/2
    for pvar in range(0,  log2length):
        nvar_pp = nvar_p/2
        base_t = 0
        for bvar in range(0,  int(b_p)):
            base_b = base_t+nvar_pp
            for nvar in range(0,  int(nvar_pp)):
                if nvar == 0:
                    bot = x[int(base_b+nvar)]
                else:
                    twiddle_factor = nvar*twiddle_step_size
                    bot = x[int(base_b+nvar)]*twiddle[int(twiddle_factor)]
                top = x[int(base_t+nvar)]
                x[int(base_t+nvar)] = top+bot
                x[int(base_b+nvar)] = top-bot
            base_t = base_t+nvar_p
        b_p = b_p/2
        nvar_p = nvar_p*2
        twiddle_step_size = twiddle_step_size/2
    return x


def testr4dif():
    '''
    Test and time dif radix4 w/ multiple length random sequences
    '''
    flag = 0
    i = 0
    radix = 4
    r4diftimes = np.zeros(8)
    for log2length in range(2, 10):
        x = np.random.rand(2*np.power(4, log2length)).view(np.complex128)
        x_ref = np.fft.fft(x)
        length = np.power(4, log2length)
        assert x.shape[0] == length
        kmax = 3*((float(length)/4.)-1)
        k_wavenumber = np.linspace(0, kmax, int(kmax+1))
        twiddlefactors = np.exp(-2j*np.pi*k_wavenumber/length)
        tvar = time.time()
        x = dif_fft4(x, twiddlefactors, log2length)
        r4diftimes[i] = time.time()-tvar
        x = digitreversal(x, radix, log2length, length)
        t_f = np.allclose(x, x_ref)
        if t_f == 0:
            flag = 1
        assert (t_f)
        i = i+1
    if flag == 0:
        print("All radix-4 dif results were correct.")
    return r4diftimes


def testr4():
    '''
    Test and time dit radix4 w/ multiple length random sequences
    '''
    flag = 0
    i = 0
    radix = 4
    r4times = np.zeros(8)
    for svar in range(2, 10):
        xarray = np.random.rand(2*np.power(4, svar)).view(np.complex128)
        xpy = np.fft.fft(xarray)
        nvar = np.power(4, svar)
        xarray = digitreversal(xarray, radix, svar, nvar)
        kmax = 3*((float(nvar)/4.)-1)
        k_wavenumber = np.linspace(0, kmax, int(kmax+1))
        twiddles = np.exp(-2j*np.pi*k_wavenumber/nvar)
        tvar = time.time()
        xarray = fft4(xarray, twiddles, svar)
        r4times[i] = time.time()-tvar
        t_f = np.allclose(xarray, xpy)
        if t_f == 0:
            flag = 1
        assert (t_f)
        i = i+1
    if flag == 0:
        print("All radix-4 dit results were correct.")
    return r4times


def testr2dif():
    '''
    Test and time radix2 dif w/ multiple length random sequences
    '''
    flag = 0
    i = 0
    radix = 2
    r2diftimes = np.zeros(8)
    for rvar in range(2, 10):
        svar = np.power(4, rvar)
        cpy = np.random.rand(2*svar).view(np.complex_)
        gpy = np.fft.fft(cpy)
        nvar = svar
        kmax = (float(nvar)/2.)-1
        k_wavenumber = np.linspace(0, kmax, int(kmax+1))
        twiddles = np.exp(-2j*np.pi*k_wavenumber/nvar)
        t1time = time.time()
        gvar = dif_fft0(cpy, twiddles, int(2*rvar))
        r2diftimes[i] = time.time()-t1time
        zvar = digitreversal(gvar, radix, int(2*rvar), svar)
        t_f = np.allclose(zvar, gpy)
        if t_f == 0:
            flag = 1
        assert (t_f)
        i = i+1
    if flag == 0:
        print("All radix-2 dif results were correct.")
    return r2diftimes


def testr2():
    '''
    Test and time radix2 dit w/ multiple length random sequences
    '''
    radix = 2
    flag = 0
    i = 0
    r2times = np.zeros(8)
    for rvar in range(2, 10):
        svar = np.power(4, rvar)
        cpy = np.random.rand(2*svar).view(np.complex_)
        gpy = np.fft.fft(cpy)
        nvar = svar
        kmax = (float(nvar)/2.)-1
        k_wavenumber = np.linspace(0, kmax, int(kmax+1))
        twiddles = np.exp(-2j*np.pi*k_wavenumber/nvar)
        zvar = digitreversal(cpy, radix, int(2*rvar), svar)
        t1time = time.time()
        garray = fft2(zvar, twiddles, int(2*rvar))
        r2times[i] = time.time()-t1time
        t_f = np.allclose(garray, gpy)
        if t_f == 0:
            flag = 1
        assert (t_f)
        i = i+1
    if flag == 0:
        print("All radix-2 dit results were correct.")
    return r2times


def plot_times(tr2, trdif2, tr4, trdif4):
    '''
    plot performance
    '''
    uvector = np.zeros(6, dtype=int)
    for i in range(2, 10):
        uvector[i-2] = np.power(4, i)
    plt.figure(figsize=(7, 5))
    plt.rc("font", size=9)
    plt.loglog(uvector, trdif2, 'o',  ms=5,  markerfacecolor="None",
               markeredgecolor='red',  markeredgewidth=1,
               basex=4,  basey=10,  label='radix-2 DIF')
    plt.loglog(uvector, tr2, '^',  ms=5,  markerfacecolor="None",
               markeredgecolor='green',  markeredgewidth=1,
               basex=4,  basey=10,  label='radix-2 DIT')
    plt.loglog(uvector, trdif4, 'D',  ms=5,  markerfacecolor="None",
               markeredgecolor='blue',  markeredgewidth=1,
               basex=4,  basey=10,  label='radix-4 DIF')
    plt.loglog(uvector, tr4, 's',  ms=5,  markerfacecolor="None",
               markeredgecolor='black',  markeredgewidth=1,
               basex=4,  basey=10,  label='radix-4 DIT')
    plt.legend(loc=2)
    plt.grid()
    plt.xlim([12, 18500])
    plt.ylim([.00004, 1])
    plt.ylabel("time (seconds)")
    plt.xlabel("sequence length")
    plt.title("Time vs Length")
    plt.savefig('tvl2.png',  bbox_inches='tight')
    plt.show()
    return None


def test():
    '''
    test performance
    '''
    trdif4 = testr4dif()
    tr4 = testr4()
    trdif2 = testr2dif()
    tr2 = testr2()
    print(f"DIF-4: {trdif4}")
    print(f"DIT-4: {tr4}")
    print(f"DIF-2: {trdif2}")
    print(f"DIT-2: {tr2}")
    # plot_times(tr2, trdif2, tr4, trdif4)
    return None


def foo():
    radix = 4
    log2length = 2
    length = np.power(4, log2length)
    x = np.zeros(length, dtype=np.complex128)
    x[0] = 1.0
    x[1] = 2.0
    x[2] = 3.0

    print(x)
    print(np.fft.fft(x))

    kmax = (length / 4 - 1) * 3
    k_wavenumber = np.linspace(0, kmax, int(kmax+1))
    wf = np.exp(-2j*np.pi*k_wavenumber/length)
    wb = np.exp(+2j*np.pi*k_wavenumber/length)
    X = dif_fft4(x, wf, log2length)
    X = digitreversal(X, radix, log2length, length)
    print(X)

    # Y = dif_fft4(X, wb, log2length)
    # Y = digitreversal(Y, radix, log2length, length)
    # print(Y/length)


foo()
# test()
