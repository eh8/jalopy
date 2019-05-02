from keyPressed import PressKey, ReleaseKey, W, A, S, D


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    PressKey(A)
    ReleaseKey(D)


def right():
    PressKey(D)
    ReleaseKey(A)


def slow():
    ReleaseKey(W)
    PressKey(S)
