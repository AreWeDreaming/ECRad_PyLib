#!/bin/env python

def writef(file, format, n, array):
    ''' Formatted output of an array.                            
                                                             
    ** Python example **
    import sys
    sys.path.append("/home/rbb/python/modules/")
    import numpy as np
    import fmtout as fmt


    file = open("writef.dat","w")
    N = 22
    X = ' '
    a = np.linspace(1.E-6,1.E12,N)
    print a
    fmt= "%3d"+X
    st = wf.writef(file,fmt,1,[N])
    fmt="%12.4e"+X
    st =wf.writef(file,fmt,6, a)
    file.close()

    ** Fortran example **

    program writef
    implicit none
    real :: a(30)
    integer :: n, i
    
    open(12,file='writef.dat')
    read(12,'(I3)') n
    print *, ' n = ', n
    read(12,'(6(E12.4,1X))') (a(i), i=1,n)
    close(12)
    
    print *, ' first = ', a(1)
    print *, ' last = ', a(n)
    end program writef
    
    
    R. Bilato 11.11.2009'''

    import numpy as np
    a = np.shape(array)
    nel = a[0]
    fformat = str(''.join([format for i in xrange(0, n)]))
    j1 = -1
    for j in xrange(0, nel - n, n):
        j1 = j + (n - 1)
        astrg = ''.join(["array[" + str(i) + "]," for i in xrange(j, j1)])
        astrg = "(" + astrg + "array[" + str(j1) + "])"
        exec(str("a = " + astrg))
        file.write(fformat % a + "\n")
    if j1 < (nel - 1):
        fformat = str(''.join([format for i in xrange(j1 + 1, nel)]))
        astrg = ''.join(["array[" + str(i) + "]," for i in xrange(j1 + 1, nel - 1)])
        astrg = "(" + astrg + "array[" + str(nel - 1) + "])"
        exec(str("a = " + astrg))
        file.write(fformat % a + "\n")


def readu(filename):

    file = open(filename, 'r')

    line = file.readline()
    line = line.split()

    nel = len(line)
    cmd = str('res=[' + ','.join(["[float(line[" + str(i) + "])]" for i in xrange(0, nel)]) + ']')
    exec(cmd)


    while True:
        line = file.readline()
        a = len(line.split())
        if len(line) == 0:
            file.close()
            return res
        line = line.split()
        for i in xrange(0, nel):
            res[i].append(float(line[i]))

    file.close()
    return res
# ============================================================= #




def readarray(lines, ilst, n, dtype='none'):
    '''Extract values on a line.

    Input:
        lines: string of array as given by readlines() routine
        ilst  = line from where to start to read lines
        n = number of values inline
    Output: (y,il)
        y: array of n read values
        il: next line to be read'''

    import numpy as np
    if dtype == 'none':
        dtype = np.float

    if dtype == np.complex:
        jjump = 2
        aim = 0. + 1.j
    else:
        jjump = 1

    y = np.zeros(n / jjump, dtype=dtype)
    leng = len(lines[ilst].split())

    il = ilst
    j1 = 0
    for i in range(0, n, leng):
        line = lines[il].split()
        for j in range(0, len(line), jjump):
            if dtype == np.complex:
                y[j1] = float(line[j]) + aim * float(line[j + 1])
                j1 = j1 + 1
            else:
                y[j1] = line[j]
                j1 = j1 + 1
        il = il + 1


    return (y, il)
# ============================================================= #

def readvarname(lines):

    ivar = []
    nlines = len(lines)
    for j in range(0, nlines):
        l = lines[j]
        if l.find("_VN_") != -1:
            ivar.append(j)
    return ivar

# ============================================================= #

def readvar(lines, varname, jpos=0):
    ' --------------------------------------------------------- '
    ' Input:                                                    '
    '    lines: string of array as given by readlines() routine '
    '    varn: name of the variable                             '
    '                                                           '
    ' Output:                                                   '
    '    y: array of variable values                            '
    ' --------------------------------------------------------- '
    import numpy as np

    varn = varname.strip()
    varn = '_VN_' + varn

# < Looking for the variable >#
    if jpos == 0:
        j = 0
        found = -1
        for l in lines:
            if l.find(varn) != -1:
                line = lines[j].split()
                for m in line:
                    v = m.strip()
                    if v == varn:
                        found = 1
                        break
            if found == 1: break
            if j == len(lines) - 1:
                print varn, ' not found'
                return -1
            else:
                j = j + 1
    else:
        j = jpos

# < Looking for the rank and type of the variable >#
    j = j + 3
    line = lines[j].split()
    rank = int(line[0])
    if len(line) > 1:
        if line[1] == 'i':
            vartype = np.int
        elif line[1] == 'f':
            vartype = np.float
        elif line[1] == 'c':
            vartype = np.complex
    else:
        vartype = np.float

    dim = np.zeros(rank, dtype=np.int)
    for i in range(0, rank):
        dim[i] = line[2 + i]

    var = np.zeros(dim, dtype=vartype)

    if vartype == np.complex:
        njump = 2
    else:
        njump = 1

    j = j + 1
    if rank == 0:
        n = njump
        (var, j) = readarray(lines, j, n, dtype=vartype)
    elif rank == 1:
        n = njump * dim[rank - 1]
        (var, j) = readarray(lines, j, n, dtype=vartype)
    elif rank == 2:
        n = njump * dim[rank - 1]
        aa = np.zeros(n, dtype=vartype)
        for i in range(0, dim[0]):
            (aa, j) = readarray(lines, j, n, dtype=vartype)
            var[i, :] = aa
    elif rank == 3:
        n = njump * dim[rank - 1]
        aa = np.zeros(n, dtype=vartype)
        for i in range(0, dim[0]):
            for l in range(0, dim[1]):
                (aa, j) = readarray(lines, j, n, dtype=vartype)
                var[i, l, :] = aa
    elif rank == 4:
        n = njump * dim[rank - 1]
        aa = np.zeros(n, dtype=vartype)
        for i in range(0, dim[0]):
            for l in range(0, dim[1]):
                for m in range(0, dim[2]):
                    (aa, j) = readarray(lines, j, n, dtype=vartype)
                    var[i, l, m, :] = aa
    else:
        print 'None'

    return var
# ============================================================= #

class efmtout:

    def __init__(self, name, descr, units, val):
        self.name = name
        self.descr = descr
        self.units = units
        self.val = val


class fmtout:
    def __init__(self, file):
        import numpy as np
        self.file = file
        f = open(self.file, 'r')
        self.lines = f.readlines()
        self.ivar = readvarname(self.lines)
        list = []

        for i in self.ivar:
            name = self.lines[i].strip()
            name = name[4:len(name)]
            val = readvar(self.lines, name, jpos=i)
            varname = 'self.' + name
            idouble = 0
            for lname in list:
                if name == lname:
                    cmd = 'dim = ' + varname + '.val.shape'
                    exec cmd
# Extend the rank of the val matrix
                    if len(dim) == len(val.shape):
                        cmd = 'dim = (2'
                        for i in range(0, len(dim)):
                            cmd = cmd + ',' + str(dim[i])
                        cmd = cmd + ')'
                        exec cmd
                        cmd = 'a = np.resize(' + varname + '.val,dim)'
                        exec cmd
                        a[1] = val
                        cmd = varname + '.val=a'
                        exec cmd
                    else:
                        cmd = 'dim = (' + str(dim[0] + 1)
                        for i in range(1, len(dim)):
                            cmd = cmd + ',' + str(dim[i])
                        cmd = cmd + ')'
                        exec cmd
                        cmd = 'a = np.resize(' + varname + '.val,dim)'
                        exec cmd
                        a[dim[0] - 1] = val
                        cmd = varname + '.val=a'
                        exec cmd
                    idouble = 1
                    break

            if idouble != 0:
                continue

            list.append(name)

            descr = self.lines[i + 1].strip()
            descr = descr[4:len(descr)]

            units = self.lines[i + 2].strip()
            units = units[4:len(units)]

            cmd = 'self.' + name + '= efmtout(name,descr,units,val)'
            exec cmd

        self.list = list

    def table(self):
        print ' '
        print '{0:30} {1:20}'.format(" Table of variables in ", self.file)
        print '{0:20} {1:40} {2:10} {3:4} {4:4}'.format("Name", "Description", "units", "rnk ", " dim")
        print '{0:20} {1:40} {2:10} {3:4} {4:4}'.format("====", "===========", "=====", "=== ", " ===")
        for var in self.list:
            cmd = 'vd = self.' + var + '.descr'
            exec cmd
            cmd = 'vu = self.' + var + '.units'
            exec cmd
            cmd = 'dim = self.' + var + '.val.shape'
            exec cmd
            cmd = 'rnk = len(dim)'
            exec cmd
            cm1 = str("'{0:20} {1:40} {2:10} {3:4}")
            cm2 = str(".format(var, vd, vu, rnk")
            for k in range(0, rnk):
                cm1 = cm1 + " {" + str(4 + k) + ":4}"
                cm2 = cm2 + ", dim[" + str(k) + "]"
            cm1 = cm1 + "'"
            cm2 = cm2 + ")"
            cmd = "print " + cm1 + cm2
            exec cmd
        print ' '

