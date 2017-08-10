from subprocess import call

def convert(filename):
    call(["convert", filename, "-blur", "0x2", "-paint", "3", filename])

