import numpy as np
from cython import boundscheck, wraparound
from random import randint

# TODO: Refresh understanding of boundscheck, wraparound (determine if they need to be in front of every function)
# TODO: ?? Convert pile tuple return to class ?? (Probably only if continue to add more params to return)
# TODO: Validate array and variable typing (unsigned long, uint32, etc.)
# TODO: Check checkerboard seed stays inside image sink (without the need for post seed clearing if possible)
# TODO: Update all function description comments.
# TODO: Assess array size effect on performance
# TODO: Assess impact to performance of if-case structure (flattened vs. hierarchical vs. separate functions)

'''
Calculates the stable configuration for sandpiles of different lattice structures,
quantities and locations of grains dropped, different sizes and shapes of sinks, and
different seed quantities and configurations.
'''

#######################
# Calculate Sandpiles #
#######################

@boundscheck(False)
@wraparound(False)
def calculate_sandpile_grid_seeded(int xMax, int yMax, int grains, neighbors, list dropSpots=None, int seed=0, seedType='uniform', seedAttr=None):
    sandbox = np.full((xMax, yMax), seed, dtype = np.int64)
    if seedType == 'checker':
        sandbox = checkerboard_seed(xMax, yMax, sandbox, seedAttr)
    sandbox = drop_all_grains(xMax, yMax, grains, sandbox, dropSpots)
    if neighbors == "von_neumann" or neighbors == "von_neumann2" or neighbors == "xcross" or neighbors == "knightleft" or neighbors == "knightright":
        return topple_grid_algorithm2_neighborhoods_bounded(xMax, yMax, sandbox, neighbors, 4)
    elif neighbors == "moore" or neighbors == "moore2" or neighbors == "knight" or neighbors == "knightsweepleft" or neighbors == "knightsweepright":
        return topple_grid_algorithm2_neighborhoods_bounded(xMax, yMax, sandbox, neighbors, 8)
    elif neighbors == "pinwheelleft" or neighbors == "pinwheelright" or neighbors == "8cross" or neighbors == "8pointstar":
        return topple_grid_algorithm2_neighborhoods_bounded(xMax, yMax, sandbox, neighbors, 8)

@boundscheck(False)
@wraparound(False)
def calculate_sandpile_wraparound_seeded(int xMax, int yMax, int grains, topple, list dropSpots=None, int seed=0, seedType='uniform', seedAttr=None, shape='cylinder'):
    sandbox = np.full((xMax, yMax), seed, dtype = np.int64)
    if seedType == 'checker':
        sandbox = checkerboard_seed(xMax, yMax, sandbox, seedAttr)
    sandbox = drop_all_grains(xMax, yMax, grains, sandbox, dropSpots)
    if topple == "von_neumann":
        return topple_grid_algorithm2_von_neumann_wraparound(xMax, yMax, sandbox, shape)

@boundscheck(False)
@wraparound(False)
def calculate_sandpile_cubesurface_seeded(int xMax, int yMax, int zMax, int grains, topple, list dropSpots=None, int seed=0, seedType='uniform', seedAttr=None):
    sandbox_frontback = np.full((2, xMax, yMax), seed, dtype = np.int64)
    sandbox_leftright = np.full((2, zMax, yMax), seed, dtype = np.int64)
    sandbox_bottomtop = np.full((2, xMax, zMax), seed, dtype = np.int64)

    if seedType == 'checker':
        sandbox_frontback = checkerboard_seed_faces(xMax, yMax, sandbox_frontback, seedAttr)
        sandbox_leftright = checkerboard_seed_faces(zMax, yMax, sandbox_leftright, seedAttr)
        sandbox_bottomtop = checkerboard_seed_faces(xMax, zMax, sandbox_bottomtop, seedAttr)

    sandbox_frontback = drop_all_grains_faces(xMax, yMax, grains, sandbox_frontback, "frontback", dropSpots)
    sandbox_leftright = drop_all_grains_faces(zMax, yMax, grains, sandbox_leftright, "leftright", dropSpots)
    sandbox_bottomtop = drop_all_grains_faces(xMax, zMax, grains, sandbox_bottomtop, "bottomtop", dropSpots)

    if topple == "von_neumann":
        return topple_grid_algorithm2_von_neumann_cubesurface(xMax, yMax, zMax, sandbox_frontback, sandbox_leftright, sandbox_bottomtop)

@boundscheck(False)
@wraparound(False)
def calculate_sandpile_icosahedronsurface_seeded(int xMax, int yMax, int numRows, int grains, topple, list dropSpots=None, int seed=0, seedType='uniform', seedAttr=None):
    arrX = 2 * (xMax + 1)
    arrY = 5 * yMax
    sandbox = np.full((arrX, arrY), seed, dtype = np.int64)

    if seedType == 'checker':
        sandbox = checkerboard_seed_icosahedron(arrX, arrY, sandbox, seedAttr)
    sandbox = drop_all_grains_icosahedron(xMax, yMax, arrX, arrY, grains, sandbox, dropSpots)

    # Toppling goes here...
    sandboxTuple = topple_sandpile_icosahedron_surface(xMax, yMax, arrX, arrY, sandbox)
    sandbox = sandboxTuple[0]
    toppleCtr = sandboxTuple[1]

    pileArray = []
    boundArray = []
    for side in range(20):
        tempArr = np.zeros((xMax, yMax), dtype = np.int64)
        boundTempArr = np.zeros((xMax, yMax), dtype = np.int64)
        y_sidx_offset = int(side / 4) * yMax
        sy_range = y_sidx_offset
        ey_range = y_sidx_offset + yMax
        
        col = side % 4
        if side % 2 == 1: # odd
            x_sidx_offset = 1 + (int(col/3) * (xMax+1))    
            for y_sidx in range(sy_range, ey_range):
                y_eidx = y_sidx - y_sidx_offset
                
                sx_range = (y_eidx*2) + x_sidx_offset
                ex_range = xMax + x_sidx_offset
                for x_sidx in range(sx_range, ex_range):
                    x_eidx = (x_sidx - x_sidx_offset) - y_eidx
                    tempArr[x_eidx][y_eidx] = sandbox[x_sidx][y_sidx]
                    boundTempArr[x_eidx][y_eidx] = 1
        else:
            x_sidx_offset = (int(col/2) * (xMax+1))
            for y_sidx in range(sy_range, ey_range):
                y_eidx = y_sidx - y_sidx_offset
                
                sx_range = x_sidx_offset
                ex_range = (y_eidx*2) + x_sidx_offset + 1
                for x_sidx in range(sx_range, ex_range):
                    x_eidx = (x_sidx - x_sidx_offset) + (int(xMax/2) - y_eidx)
                    tempArr[x_eidx][y_eidx] = sandbox[x_sidx][y_sidx]
                    boundTempArr[x_eidx][y_eidx] = 1
        pileArray.append(tempArr)
        boundArray.append(boundTempArr)
    return (pileArray, toppleCtr, boundArray, sandbox)

#########################
# Seeds and grain drops #
#########################

@boundscheck(False)
@wraparound(False)
def drop_all_grains(int xMax, int yMax, int grains, sandbox, list dropSpots=None):
    cdef long long[:, :] sandbox_view = sandbox
    cdef int dx
    cdef int dy
    cdef float mult

    if dropSpots == None:
        dropSpots = [(int(xMax/2), int(yMax/2), 1.0)]
    for drop in dropSpots:
        dx = drop[0]
        dy = drop[1]
        mult = drop[2]
        sandbox_view[dx, dy] = int(grains * mult)
    return sandbox

@boundscheck(False)
@wraparound(False)
def drop_all_grains_faces(int xMax, int yMax, int grains, sandbox, facetype, list dropSpots=None):
    cdef long long[:, :, :] sandbox_view = sandbox
    cdef int dx
    cdef int dy
    cdef int faceIdx
    cdef float mult

    if dropSpots == None:
        dropSpots = [(int(xMax/2), int(yMax/2), 1.0, "top")]
    for drop in dropSpots:
        dx = drop[0]
        dy = drop[1]
        mult = drop[2]
        face = drop[3]

        skip = False
        if facetype == "frontback":
            if face == "front":
                faceIdx = 0
            elif face == "back":
                faceIdx = 1
            else:
                skip = True
        elif facetype == "leftright":
            if face == "left":
                faceIdx = 0
            elif face == "right":
                faceIdx = 1
            else:
                skip = True
        elif facetype == "bottomtop":
            if face == "top":
                faceIdx = 0
            elif face == "bottom":
                faceIdx = 1
            else:
                skip = True
        
        if skip == False:
            sandbox_view[faceIdx, dx, dy] = int(grains * mult)
    return sandbox

@boundscheck(False)
@wraparound(False)
def drop_all_grains_icosahedron(int xMax, int yMax, int arrX, int arrY, int grains, sandbox, list dropSpots=None):
    cdef long long[:, :] sandbox_view = sandbox
    cdef int dx
    cdef int dy
    cdef int arrCol
    cdef int arrRow
    cdef int midMaxX
    cdef int faceIdx
    cdef float mult

    if dropSpots == None:
        midMaxX = 1 + (2 * int(yMax/2))
        dropSpots = [(int(midMaxX/2), int(yMax/2), 1, 1.0)]

    for drop in dropSpots:
        dx = drop[0]
        dy = drop[1]
        mult = drop[2]
        faceIdx = int(drop[3]) - 1 # Face 1-20, index 0-19
        arrRow = dy + (int(faceIdx / 4) * yMax)
        
        faceX = faceIdx % 4
        arrCol = 0
        if faceX == 0:
            arrCol = dx
        elif faceX == 1:
            arrCol = 1 + (2 * dy) + dx
        elif faceX == 2:
            arrCol = int(arrX/2) + dx
        elif faceX == 3:
            arrCol = arrX - dx - 1

        sandbox_view[arrCol, arrRow] = int(grains * mult)
    return sandbox

@boundscheck(False)
@wraparound(False)
def checkerboard_seed(int xMax, int yMax, sandbox, seeds):
    cdef long long[:, :] sandbox_view = sandbox
    cdef int x
    cdef int y
    cdef int val
    cdef int idx

    for x in range(xMax):
        for y in range(yMax):
            idx = (x+y) % 2
            val = seeds[idx]
            sandbox_view[x, y] = val
    return sandbox

@boundscheck(False)
@wraparound(False)
def checkerboard_seed_faces(int xMax, int yMax, sandbox, seeds):
    cdef long long [:, :, :] sandbox_view = sandbox
    cdef int x
    cdef int y
    cdef int face
    cdef int val
    cdef int idx

    for face in range(2):
        for x in range(xMax):
            for y in range(yMax):
                idx = (x+y) % 2
                val = seeds[idx]
                sandbox_view[face, x, y] = val
    return sandbox

@boundscheck(False)
@wraparound(False)
def checkerboard_seed_icosahedron(int xMax, int yMax, sandbox, seeds):
    cdef long long[:, :] sandbox_view = sandbox
    cdef int x
    cdef int y
    cdef int val
    cdef int idx

    for x in range(xMax):
        for y in range(yMax):
            idx = x % 2
            val = seeds[idx]
            sandbox_view[x, y] = val
    return sandbox

#######################
#  Topple Algorithms  #
#######################

'''
Grid Lattice Topple Algorithm 2 (Calculate Bound):

Same as Algorithm 2, but adds a second array which tracks the
spread of the grains of sand. This enables easy setting of a
different background color from the 0 grain color.

Returns a tuple (sandpile, toppleCtr, pileBorder).
'''

@boundscheck(False)
@wraparound(False)
def topple_grid_algorithm2_neighborhoods_bounded(int xMax, int yMax, sandbox, neighbors, int threshold):
    cdef int x
    cdef int y
    cdef int idx
    cdef tuple arr
    cdef long long numToMove
    cdef long long toppleCtr = 0
    cdef long long [:, :] sandbox_view = sandbox
    direction = "odd"

    pileBorders = np.full((xMax, yMax), False, dtype=np.uint8)
    cdef char[:, :] pileBorders_view = pileBorders

    arr = np.where(sandbox >= threshold) # Topples piles of sand with a number of grains greater than or equal to a threshold
    while len(arr[0]) > 0:
        toppleCtr += len(arr[0])
        for idx in range(len(arr[0])):
            x = int(arr[0][idx])
            y = int(arr[1][idx])
            numToMove = int(sandbox_view[x, y] / threshold) # Does this work for int64 numbers?
            pileBorders_view[x, y] = 1

            if neighbors == 'von_neumann' or neighbors == 'moore' or neighbors == 'pinwheelright' or neighbors == 'pinwheelleft' or neighbors == '8cross':
                if (x > 0):
                    sandbox_view[x-1, y] += numToMove
                    pileBorders_view[x-1, y] = 1
                if ((x+1) < xMax):
                    sandbox_view[x+1, y] += numToMove
                    pileBorders_view[x+1, y] = 1
                if (y > 0):
                    sandbox_view[x, y-1] += numToMove
                    pileBorders_view[x, y-1] = 1
                if ((y+1) < yMax):
                    sandbox_view[x, y+1] += numToMove
                    pileBorders_view[x, y+1] = 1

            if neighbors == 'xcross' or neighbors == 'moore' or neighbors == '8pointstar':
                if (x > 0 and y > 0):
                    sandbox_view[x-1, y-1] += numToMove
                    pileBorders_view[x-1, y-1] = 1
                if (x > 0 and (y+1) < yMax):
                    sandbox_view[x-1, y+1] += numToMove
                    pileBorders_view[x-1, y+1] = 1
                if ((x+1) < xMax and y > 0):
                    sandbox_view[x+1, y-1] += numToMove
                    pileBorders_view[x+1, y-1] = 1
                if ((x+1) < xMax and (y+1) < yMax):
                    sandbox_view[x+1, y+1] += numToMove
                    pileBorders_view[x+1, y+1] = 1

            if neighbors == 'knight' or neighbors == 'knightright' or neighbors == 'knightsweepright' or neighbors == 'pinwheelright':
                if (y > 1 and x > 0):
                    sandbox_view[x-1, y-2] += numToMove
                    pileBorders_view[x-1, y-2] = 1
                if ((y+2) < yMax and (x+1) < xMax):
                    sandbox_view[x+1, y+2] += numToMove
                    pileBorders_view[x+1, y+2] = 1
                if (x > 1 and (y+1) < yMax):
                    sandbox_view[x-2, y+1] += numToMove
                    pileBorders_view[x-2, y+1] = 1
                if ((x+2) < xMax and y > 0):
                    sandbox_view[x+2, y-1] += numToMove
                    pileBorders_view[x+2, y-1] = 1

            if neighbors == 'knight' or neighbors == 'knightleft' or neighbors == 'knightsweepleft' or neighbors == 'pinwheelleft':
                if (x > 1 and y > 0):
                    sandbox_view[x-2, y-1] += numToMove
                    pileBorders_view[x-2, y-1] = 1
                if ((x+2) < xMax and (y+1) < yMax):
                    sandbox_view[x+2, y+1] += numToMove
                    pileBorders_view[x+2, y+1] = 1
                if (y > 1 and (x+1) < xMax):
                    sandbox_view[x+1, y-2] += numToMove
                    pileBorders_view[x+1, y-2] = 1
                if ((y+2) < yMax and x > 0):
                    sandbox_view[x-1, y+2] += numToMove
                    pileBorders_view[x-1, y+2] = 1

            if neighbors == 'von_neumann2' or neighbors == 'moore2' or neighbors == 'knightsweepleft' or neighbors == 'knightsweepright' or neighbors == '8cross' or neighbors == '8pointstar':
                if (x > 1):
                    sandbox_view[x-2, y] += numToMove
                    pileBorders_view[x-2, y] = 1
                if ((x+2) < xMax):
                    sandbox_view[x+2, y] += numToMove
                    pileBorders_view[x+2, y] = 1
                if (y > 1):
                    sandbox_view[x, y-2] += numToMove
                    pileBorders_view[x, y-2] = 1
                if ((y+2) < yMax):
                    sandbox_view[x, y+2] += numToMove
                    pileBorders_view[x, y+2] = 1

            if neighbors == 'moore2':
                if (x > 1 and y > 1):
                    sandbox_view[x-2, y-2] += numToMove
                    pileBorders_view[x-2, y-2] = 1
                if (y > 1 and (x+2) < xMax):
                    sandbox_view[x+2, y-2] += numToMove
                    pileBorders_view[x+2, y-2] = 1
                if ((y+2) < yMax and x > 1):
                    sandbox_view[x-2, y+2] += numToMove
                    pileBorders_view[x-2, y+2] = 1
                if ((x+2) < xMax and (y+2) < yMax):
                    sandbox_view[x+2, y+2] += numToMove
                    pileBorders_view[x+2, y+2] = 1
            
            sandbox_view[x, y] -= (threshold * numToMove)
        arr = np.where(sandbox >= threshold)
    return (sandbox, toppleCtr, pileBorders)

'''
Grid Lattice Topple Algorithm 2 (Wrap-around)

Same as Algorithm 2, but wraps around the x, y, or diagonal depending on the specified shape.
'cylinder' == wrap-around x-axis

Returns a tuple (sandpile, toppleCtr)
'''
@boundscheck(False)
@wraparound(False)
def topple_grid_algorithm2_von_neumann_wraparound(int xMax, int yMax, sandbox, shape):
    cdef int x
    cdef int y
    cdef int idx
    cdef tuple arr
    cdef long long numToMove
    cdef long long toppleCtr = 0
    cdef long long [:, :] sandbox_view = sandbox

    arr = np.where(sandbox > 3) # Topples at 4 grains tall or greater
    while len(arr[0]) > 0:
        toppleCtr += len(arr[0])
        for idx in range(len(arr[0])):
            x = int(arr[0][idx])
            y = int(arr[1][idx])
            numToMove = int(sandbox_view[x, y] / 4)
            if x > 0:
                sandbox_view[x-1, y] += numToMove
            if (x+1) < xMax:
                sandbox_view[x+1, y] += numToMove
            if y > 0:
                sandbox_view[x, y-1] += numToMove
            if (y+1) < yMax:
                sandbox_view[x, y+1] += numToMove

            # Wrap-around x-axis
            if (x == 0) and (shape == 'cylinder' or shape == 'squarewrap'):
                sandbox_view[xMax-1, y] += numToMove
            if ((x+1) == xMax) and (shape == 'cylinder' or shape == 'squarewrap'):
                sandbox_view[0, y] += numToMove

            # Wrap-around y-axis
            if (y == 0) and (shape == 'squarewrap'):
                sandbox_view[x, yMax-1] += numToMove
            if ((y+1) == yMax) and (shape == 'squarewrap'):
                sandbox_view[x, 0] += numToMove
            sandbox_view[x, y] -= (4 * numToMove)
        arr = np.where(sandbox > 3)
    return (sandbox, toppleCtr)

@boundscheck(False)
@wraparound(False)
def topple_grid_algorithm2_von_neumann_cubesurface(int xMax, int yMax, int zMax, sandbox_frontback, sandbox_leftright, sandbox_bottomtop):
    cdef int x
    cdef int y
    cdef int z
    cdef int idx
    cdef tuple fbArr
    cdef tuple lrArr
    cdef tuple btArr
    cdef int faceIdx
    cdef long long numToMove
    cdef long long toppleCtr = 0
    cdef long long [:, :, :] sandbox_frontback_view = sandbox_frontback
    cdef long long [:, :, :] sandbox_leftright_view = sandbox_leftright
    cdef long long [:, :, :] sandbox_bottomtop_view = sandbox_bottomtop

    fbArr = np.where(sandbox_frontback > 3) # Topples at 4 grains tall or greater
    lrArr = np.where(sandbox_leftright > 3)
    btArr = np.where(sandbox_bottomtop > 3)
    while len(fbArr[0]) > 0 or len(lrArr[0]) > 0 or len(btArr[0]) > 0:
        toppleCtr += len(fbArr[0])
        toppleCtr += len(lrArr[0])
        toppleCtr += len(btArr[0])

        for idx in range(len(fbArr[0])):
            faceIdx = int(fbArr[0][idx])
            x = int(fbArr[1][idx])
            y = int(fbArr[2][idx])
            numToMove = int(sandbox_frontback_view[faceIdx, x, y] / 4)

            if x > 0:
                sandbox_frontback_view[faceIdx, x-1, y] += numToMove
            if (x+1) < xMax:
                sandbox_frontback_view[faceIdx, x+1, y] += numToMove
            if y > 0:
                sandbox_frontback_view[faceIdx, x, y-1] += numToMove
            if (y+1) < yMax:
                sandbox_frontback_view[faceIdx, x, y+1] += numToMove

            if faceIdx == 0: # Front (xy)
                if x == 0:
                    sandbox_leftright_view[0, zMax-1, y] += numToMove # Left (zy)
                if ((x+1) == xMax):
                    sandbox_leftright_view[1, 0, y] += numToMove # Right (zy)
                if y == 0:
                    sandbox_bottomtop_view[0, x, zMax-1] += numToMove # Bottom (xz)
                if ((y+1) == yMax):
                    sandbox_bottomtop_view[1, x, 0] += numToMove # Top (xz)
            elif faceIdx == 1: # Back (xy)
                if x == 0:
                    sandbox_leftright_view[1, zMax-1, y] += numToMove # Right (zy)
                if ((x+1) == xMax):
                    sandbox_leftright_view[0, 0, y] += numToMove # Left (zy)
                if y == 0:
                    sandbox_bottomtop_view[0, (xMax-1)-x, 0] += numToMove # Bottom (xz)
                if ((y+1) == yMax):
                    sandbox_bottomtop_view[1, (xMax-1)-x, zMax-1] += numToMove # Top (xz)
            sandbox_frontback_view[faceIdx, x, y] -= (4 * numToMove)

        for idx in range(len(lrArr[0])):
            faceIdx = int(lrArr[0][idx])
            z = int(lrArr[1][idx])
            y = int(lrArr[2][idx])
            numToMove = int(sandbox_leftright_view[faceIdx, z, y] / 4)

            if z > 0:
                sandbox_leftright_view[faceIdx, z-1, y] += numToMove
            if (z+1) < zMax:
                sandbox_leftright_view[faceIdx, z+1, y] += numToMove
            if y > 0:
                sandbox_leftright_view[faceIdx, z, y-1] += numToMove
            if (y+1) < yMax:
                sandbox_leftright_view[faceIdx, z, y+1] += numToMove

            if faceIdx == 0: # Left (zy)
                if z == 0:
                    sandbox_frontback_view[1, xMax-1, y] += numToMove # Back (xy)
                if ((z+1) == zMax):
                    sandbox_frontback_view[0, 0, y] += numToMove # Front (xy)
                if y == 0:
                    sandbox_bottomtop_view[0, 0, z] += numToMove # Bottom (xz)
                if ((y+1) == yMax):
                    sandbox_bottomtop_view[1, 0, (zMax-1)-z] += numToMove # Top (xz)
            elif faceIdx == 1: # Right (zy)
                if z == 0:
                    sandbox_frontback_view[0, xMax-1, y] += numToMove # Front (xy)
                if ((z+1) == zMax):
                    sandbox_frontback_view[1, 0, y] += numToMove # Back (xy)
                if y == 0:
                    sandbox_bottomtop_view[0, xMax-1, (zMax-1)-z] += numToMove # Bottom (xz)
                if ((y+1) == yMax):
                    sandbox_bottomtop_view[1, xMax-1, z] += numToMove # Top (xz)
            sandbox_leftright_view[faceIdx, z, y] -= (4 * numToMove)

        for idx in range(len(btArr[0])):
            faceIdx = int(btArr[0][idx])
            x = int(btArr[1][idx])
            z = int(btArr[2][idx])
            numToMove = int(sandbox_bottomtop_view[faceIdx, x, z] / 4)

            if x > 0:
                sandbox_bottomtop_view[faceIdx, x-1, z] += numToMove
            if (x+1) < xMax:
                sandbox_bottomtop_view[faceIdx, x+1, z] += numToMove
            if z > 0:
                sandbox_bottomtop_view[faceIdx, x, z-1] += numToMove
            if (z+1) < zMax:
                sandbox_bottomtop_view[faceIdx, x, z+1] += numToMove

            if faceIdx == 0: # Bottom (xz)
                if x == 0:
                    sandbox_leftright[0, z, 0] += numToMove # Left (zy)
                if ((x+1) == xMax):
                    sandbox_leftright[1, (zMax-1)-z, 0] += numToMove # Right (zy)
                if z == 0:
                    sandbox_frontback_view[1, (xMax-1)-x, 0] += numToMove # Back (xy)
                if ((z+1) == zMax):
                    sandbox_frontback_view[0, x, 0] += numToMove # Front (xy)
            elif faceIdx == 1: # Top (xz)
                if x == 0:
                    sandbox_leftright[0, (zMax-1)-z, yMax-1] += numToMove # Left (zy)
                if ((x+1) == xMax):
                    sandbox_leftright[1, z, yMax-1] += numToMove # Right (zy)
                if z == 0:
                    sandbox_frontback_view[0, x, yMax-1] += numToMove # Front (xy)
                if ((z+1) == zMax):
                    sandbox_frontback_view[1, (xMax-1)-x, yMax-1] += numToMove # Back (xy)
            sandbox_bottomtop_view[faceIdx, x, z] -= (4 * numToMove)

        fbArr = np.where(sandbox_frontback > 3)
        lrArr = np.where(sandbox_leftright > 3)
        btArr = np.where(sandbox_bottomtop > 3)
    return (sandbox_frontback, sandbox_leftright, sandbox_bottomtop, toppleCtr)

@boundscheck(False)
@wraparound(False)
def topple_sandpile_icosahedron_surface(int xMax, int yMax, int arrX, int arrY, sandbox):
    cdef int x
    cdef int y
    cdef int idx
    cdef int strip
    cdef int rowStart
    cdef int rowEnd
    cdef int targetX
    cdef int targetY
    cdef int yOffset
    cdef int arrXmid
    cdef tuple arr
    cdef long long numToMove
    cdef long long toppleCtr = 0
    cdef long long[:, :] sandbox_view = sandbox

    arrXmid = int(arrX / 2)

    arr = np.where(sandbox > 2)
    while len(arr[0]) > 0:
        toppleCtr += len(arr[0])
        for idx in range(len(arr[0])):
            x = int(arr[0][idx])
            y = int(arr[1][idx])
            numToMove = int(sandbox_view[x, y] / 3)

            strip = int(y/yMax)
            rowStart = yMax * strip
            rowEnd = rowStart + (yMax - 1)

            # Internal to strip x toppling
            if x > 0:
                sandbox_view[x-1, y] += numToMove
            if x < (arrX - 1):
                sandbox_view[x+1, y] += numToMove

            # Internal to strip y toppling
            if x % 2 == 0 and y < rowEnd: # Even xIdx
                sandbox_view[x+1, y+1] += numToMove
            if x % 2 == 1 and y > rowStart: # Odd xIdx
                sandbox_view[x-1, y-1] += numToMove
            
            # faceIdx % 4 == 0 toppling edge cases
            if x == 0: # left edge goes up to bottom edge
                yOffset = y % yMax
                targetX = ((yMax-1) - yOffset) * 2
                if strip == 0:
                    targetY = arrY - 1
                else:
                    targetY = rowStart - 1
                sandbox_view[targetX, targetY] += numToMove
            
            if x % 2 == 0 and x < xMax and y == rowEnd: # bottom edge goes down to left edge
                targetY = int(((xMax-1) - x) / 2)
                if strip < 4:
                    targetY = targetY + rowEnd + 1
                sandbox_view[0, targetY] += numToMove

            # faceIdx % 4 == 1 toppling edge cases
            if x % 2 == 1 and x < arrXmid and y == rowStart: # top edge goes up to bottom edge
                targetX = x + xMax
                if strip == 0:
                    targetY = arrY - 1
                else:
                    targetY = rowStart - 1
                sandbox_view[targetX, targetY] += numToMove

            # faceIdx % 4 == 2 toppling edge cases
            if x % 2 == 0 and x >= arrXmid and x < (arrX-1) and y == rowEnd: # bottom edge goes down to top edge
                targetX = x - xMax
                if strip == 4:
                    targetY = 0
                else:
                    targetY = rowEnd + 1
                sandbox_view[targetX, targetY] += numToMove

            # faceIdx % 3 == 3 toppling edge cases
            if x == (arrX-1): # right edge goes down to top edge
                yOffset = y % yMax
                targetX = (((yMax-1) - yOffset) * 2) + (arrX - xMax)
                if strip == 4:
                    targetY = 0
                else:
                    targetY = rowEnd + 1
                sandbox_view[targetX, targetY] += numToMove

            if x % 2 == 1 and x >= (arrX - xMax) and y == rowStart: # top edge goes up to right edge
                targetX = arrX - 1
                targetY = int(((arrX-1) - x) / 2)
                if strip == 0:
                    targetY = targetY + (arrY - yMax)
                else:
                    targetY = targetY + (rowStart - yMax)
                sandbox_view[targetX, targetY] += numToMove

            sandbox_view[x, y] -= (3 * numToMove)
        arr = np.where(sandbox > 2)
    return (sandbox, toppleCtr)