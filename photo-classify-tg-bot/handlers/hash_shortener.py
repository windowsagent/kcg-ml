import math

class HashShortener():

    ALPHABET = "bcdfghjklmnpqrstvwxyz0123456789BCDFGHJKLMNPQRSTVWXYZ"
    BASE = len(ALPHABET)

    def __init__(self, n_character = 16) -> None:
        # Set number of character on constructor
        self.n_character = n_character

    def encode_id(self, n):

        pad = self.n_character - 1
        n = int(n + pow(self.BASE, pad))

        s = []
        t = int(math.log(n, self.BASE))
        while True:
            bcp = int(pow(self.BASE, t))
            a = int(n / bcp) % self.BASE
            s.append(self.ALPHABET[a:a+1])
            n = n - (a * bcp)
            t -= 1
            if t < 0: break

        if len(s) > self.n_character:
            print (f'[WARNING] Length of output hash is {len(s)}, truncated to 16')

        return "".join(reversed(s[-self.n_character:]))

    def decode_id(self, n):

        n = "".join(reversed(n))
        s = 0
        l = len(n) - 1
        t = 0
        while True:
            bcpow = int(pow(self.BASE, l - t))
            s = s + self.ALPHABET.index(n[t:t+1]) * bcpow
            t += 1
            if t > l: break

        pad = self.n_character - 1
        s = int(s - pow(self.BASE, pad))

hash_shortener = HashShortener(16)
