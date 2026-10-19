"""Seekable HTTP range reader so zipfile can list/extract single members of a remote zip."""
import io, urllib.request, sys, time

URL = "https://zenodo.org/api/records/10946413/files/zenodo.zip/content"

class HTTPRangeFile(io.RawIOBase):
    def __init__(self, url, size, block=1 << 20):
        self.url, self.size, self.pos, self.block = url, size, 0, block
        self.cache = {}
        self.nreq = 0
        self.nbytes = 0
    def seekable(self): return True
    def readable(self): return True
    def tell(self): return self.pos
    def seek(self, off, whence=0):
        if whence == 0: self.pos = off
        elif whence == 1: self.pos += off
        else: self.pos = self.size + off
        return self.pos
    def _get(self, a, b):  # inclusive
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers={"Range": f"bytes={a}-{b}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                self.nreq += 1; self.nbytes += len(data)
                return data
            except Exception as e:
                print("retry", attempt, e, file=sys.stderr); time.sleep(5 * (attempt + 1))
        raise IOError("range fetch failed")
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0: return b""
        if n > 4 * self.block:  # big read: fetch directly
            data = self._get(self.pos, self.pos + n - 1)
            self.pos += len(data); return data
        out = bytearray()
        while n > 0:
            bi = self.pos // self.block
            if bi not in self.cache:
                if len(self.cache) > 64: self.cache.clear()
                a = bi * self.block; b = min(self.size, a + self.block) - 1
                self.cache[bi] = self._get(a, b)
            blk = self.cache[bi]; off = self.pos - bi * self.block
            chunk = blk[off: off + n]
            out += chunk; self.pos += len(chunk); n -= len(chunk)
        return bytes(out)
    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d; return len(d)

def open_remote(size=21950506374):
    import zipfile
    f = HTTPRangeFile(URL, size)
    return zipfile.ZipFile(io.BufferedReader(f, buffer_size=1 << 20)), f
