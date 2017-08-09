from subprocess import call

def convert(filename):
    call(["convert", filename, "-paint", "2", "-blur", "0x4", "-paint", "4", filename])

