import numpy as np

a = np.zeros((5, 2)) # array di 0 con 5 righe e 2 colonne
print(a)

a[3,1] = 1 # modifico valore a riga 3, col 1 (contando da 0)
a[2,0] = 10 # ecc
print(a)
print(a[3, 1])
print(a[2, 0])

b = np.arange(10)
print(b) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(a.shape) # forma di a: tupla di righe e colonne

d = b+5 # somma 5 a ciascun elemento dell'array

e = b * d # moltiplicazione elemento per elemento (pointwise)

g = np.sqrt(e) # operazioni applicate pointwise
   ## attenzione: non è math.sqrt, ma np.sqrt

# ESEMPIO: oscillatore

import math
t = np.arange(44100) # array con indici di campioni
ph = t / 44100 * 100 * 2 * math.pi # array con le fasi di 100 oscillazioni
s = np.sin(ph) # 100 oscillazioni sinusoidali su 44100 punti

# oscillatore più facile:
ph = np.linspace(0, 100*math.pi*2, 44100) # la fase va 100 volte da 0 a 2π su 44100 punti
s = np.sin(ph)


### importare / esportare file audio

import aa_soundfile as sf
snd, meta = sf.read('chelsea.aif') # meta contiene frequenza di campionamento, durata ecc.
att = snd / 2 # abbasso di 6 dB
sf.write('chelseaAtt.aif')
