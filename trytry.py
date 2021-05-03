def iterornot(a):
    try:
        for i in a:
            print('a', i)
    except TypeError:
        print('b', a)

iterornot(10)
iterornot([20, 30, 40])
iterornot((50, 60, 70))
