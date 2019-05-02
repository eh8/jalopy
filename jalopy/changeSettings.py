import os

# readFile and WriteFile from
# http://www.cs.cmu.edu/~112/notes/notes-strings.html


def readFile(path):
    with open(path, "rt") as f:
        return f.read()


def writeFile(path, contents):
    with open(path, "wt") as f:
        f.write(contents)


def findConfig():
    # Find the Euro Truck Simulator 2 directory in Documents
    home = os.path.expanduser('~')
    documents = os.path.join(home, 'Documents')
    euro = os.path.join(documents, 'Euro Truck Simulator 2')
    path = euro + '\config.cfg'
    # Open config.cfg
    return path


def checkConfig():
    # See if the user has ran Jalopy before
    path = findConfig()
    config = readFile(path)
    result = ""
    for line in config.splitlines():
        # If anomaly detected in game parameters, then break immediately
        # and rewrite all settings

        # Game height
        if line.startswith('user r_mode_height') or\
           line.startswith('uset r_mode_height'):
            if '600' not in line:
                return False
        # Game width
        elif line.startswith('user r_mode_width') or\
                line.startswith('uset r_mode_width'):
            if '800' not in line:
                return False
        # Game fullscreen
        elif line.startswith('user r_fullscreen') or\
                line.startswith('uset r_fullscreen'):
            if '0' not in line:
                return False
    return True


def modifyConfig():
    # Change game parameters if not
    path = findConfig()
    config = readFile(path)
    result = ""
    for line in config.splitlines():
        # Game height
        if line.startswith('user r_mode_height') or\
                line.startswith('uset r_mode_height'):
            lineToAdd = 'uset r_mode_height "600"\n'
            result += lineToAdd
        # Game width
        elif line.startswith('user r_mode_width') or\
                line.startswith('uset r_mode_width'):
            lineToAdd = 'uset r_mode_width "800"\n'
            result += lineToAdd
        # Game fullscreen
        elif line.startswith('user r_fullscreen') or\
                line.startswith('uset r_fullscreen'):
            lineToAdd = 'uset r_fullscreen "0"\n'
            result += lineToAdd
        # Other parameters
        else:
            line = line + '\n'
            result += line
    writeFile(path, result)


def main():
    if not checkConfig():
        modifyConfig()


if __name__ == '__main__':
    main()
