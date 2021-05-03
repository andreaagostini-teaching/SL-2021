


# definire una tupla

t = (10, 20, 30)

print(t)
print(type(t)) # il tipo è 'tuple'

## la tupla è immutabile, mentre la lista è mutabile

l = [10, 20, 30]
print(l[1]) # 20
l[1] = 100
print(l) # [10, 100, 30]
# tutto ok perché è una lista

print(t[1]) # 20 (vedi sopra)
"""
t[1] = 100 ##### non si può fare perché t è immutabile
"""

## e se voglio una tupla di 1 solo valore?
s = (1,) # devo usare parentesi e virgola aperta


#### CASI TIPICI
# 1. passare valori multipli come un solo argomento di una funzione
import numpy as np
d = (3, 4) # d è una tupla
a = np.zeros(d) # passo la tupla come un solo argomento

## equivalente a:
a2 = np.zeros((3, 4))



# 2. restituire valori multipli da una funzione

def doppiotriplo(x):
    return (x*2, x*3)

# 3. assegnazione multipla a una tupla di variabili
a, b = doppiotriplo(10)
# caso particolare: scambiare i valori di due variabili
x = 10
y = 20
y, x = x, y
print(f'ora x è {x} e y è {y}!')

### IN LINEA GENERALE, PER TUTTI GLI ALTRI CASI MEGLIO USARE LE LISTE
