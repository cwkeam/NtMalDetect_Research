import sys
import os

def main():
    count = 0
    for arg in sys.argv:
        if count >= 1:
            getemptyfiles(arg)
        count += 1


def getemptyfiles(rootdir):
    for root, dirs, files in os.walk(rootdir):
        for d in ['RECYCLER', 'RECYCLED']:
            if d in dirs:
                dirs.remove(d)

        for f in files:
            fullname = os.path.join(root, f)
            try:
                if os.path.getsize(fullname) == 0:
                    print(fullname)
                    os.remove(fullname)
            except WindowsError:
                continue
main()
