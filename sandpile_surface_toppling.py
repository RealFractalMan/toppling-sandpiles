import sandpile_calculations as sp
import draw_sandpile as draw_sp

import argparse
import json
from time import time

import numpy as np

def load_JSON(jsonToLoad):
    with open(jsonToLoad) as jsonFile:
        jsonData = json.load(jsonFile)
    return jsonData

def main():
    # Set command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Path to Abelian Sandpile JSON generator')
    parser.add_argument('-d', '--drawoutput', action='store_true', help='If flag present, open tkinter window and draw image')
    args = parser.parse_args()

    # Read in argument values
    config = args.config
    drawOutput = args.drawoutput

    # Load sandpile generation details from JSON file
    spData = load_JSON(config)
    filenom = str(spData.get('filenom') or 'AbelianSandpile')
    title = str(spData.get('title') or 'Abelian Sandpile')
    type = str(spData.get('pileType') or 'square')
    drawBounded = spData.get('drawBounded') # Returns python True, False if json boolean. If not present returns None.
    xMax = int(spData.get('xMax') or 101)
    yMax = int(spData.get('yMax') or 101)
    zMax = int(spData.get('zMax') or xMax)
    grains = int(spData.get('grains') or 0)
    topple = spData.get('topple') or 'von_neumann'
    seed = int(spData.get('initialSeed') or 0)
    seedType = str(spData.get('seedType') or 'uniform')
    seedAttrRaw = spData.get('seedAttributes') or None
    imgPxWidth = int(spData.get('imagePixelWidth') or 1)
    tkPxWidth = int(spData.get('tkinterPixelWidth') or 1)
    colors = spData.get('colors') or None
    bgColor = spData.get('backgroundColor') or None

    # Icosahedron Surface Params (for now only 1 topple called 'default' since not technically von_neumann)
    # Triangle size is numRows tall, 1 + 2 * (numRows - 1) wide
    numRows = int(spData.get('numRows') or 3)
    if type == 'icosahedronsurface':
        xMax = 1 + (2 * (numRows - 1))
        yMax = numRows

    # Parse border parameters
    borderData = spData.get('border') or {}
    bWidth = borderData.get('width') or 0
    bColor = borderData.get('color') or '#000000'
    bStyle = borderData.get('style') or 'all'

    # Find additional drop-spots (if specified)
    dropSpots = None
    dropSpotsRaw = spData.get('dropSpots') or []
    if(len(dropSpotsRaw) > 0):
        dropSpots = []
    for rawDrop in dropSpotsRaw:
        x = int(rawDrop['x']) # For icosahedron this corresponds to 'col'
        y = int(rawDrop['y']) # For icosahedron this corresponds to 'row'
        mult = rawDrop.get('mult') or (1.0 / len(dropSpotsRaw))
        face = rawDrop.get('face') or "top" # For icosahedron this will be a number 1-20
        dropSpots.append((x, y, mult, face))
    drawDropSpots = spData.get('drawDropSpots') # Returns python True, False if json boolean. If not present returns None.
    dropSpotColor = spData.get('dropSpotColor') or '#06d6a0' # [#d8315b, #dd2d4a, #ff006e, #f72585, #ef476f, #f15bb5, #ff70a6, #f49cbb]
    
    dropType = spData.get('dropType') or 'point' # (point, line_x, line_y, cross_xy)
    dropParam = spData.get('dropParam') or {}
    dropSpotsOn = dropParam.get('dropSpotsOn') or 1
    dropSpotsOff = dropParam.get('dropSpotsOff') or 0
    dropSpotShift = dropParam.get('dropSpotShift') or 0
    if dropType == 'line_x':
        dropSpots = []
        cycleWidth = dropSpotsOn + dropSpotsOff
        numCycles = int((xMax - dropSpotShift) / cycleWidth)
        lastCycle = (xMax - dropSpotShift) - (numCycles * cycleWidth)
        if lastCycle > dropSpotsOn:
            lastCycle = dropSpotsOn
        numDrops = (dropSpotsOn * numCycles) + lastCycle
        mult = 1.0 / numDrops
        dy = int(yMax/2)
        for x in range(dropSpotShift, xMax, cycleWidth):
            for o in range(dropSpotsOn):
                if (x+o) < xMax:
                    dx = x + o
                    dropSpots.append((dx, dy, mult))
        print(f'Num cycles = {numCycles}, Last Cycle = {lastCycle}, Num drops = {numDrops}, Mult: {mult}.')
        print(f'Drop spots: {dropSpots}.')
    
    # Parse seed attributes
    seedAttr = None
    if seedType == 'checker':
        seedAttr = (int(seedAttrRaw['seed1']), int(seedAttrRaw['seed2']))

    # Calculate the sandpile
    start = time()
    pileTuple = None
    sink = None
    if type == 'square':
        pileTuple = sp.calculate_sandpile_grid_seeded(xMax, yMax, grains, topple, dropSpots, seed=seed, seedType=seedType, seedAttr=seedAttr)
    elif type == 'cylinder' or type == 'squarewrap':
        pileTuple = sp.calculate_sandpile_wraparound_seeded(xMax, yMax, grains, topple, dropSpots, seed=seed, seedType=seedType, seedAttr=seedAttr, shape=type)
    elif type == 'cubesurface':
        pileTuple = sp.calculate_sandpile_cubesurface_seeded(xMax, yMax, zMax, grains, topple, dropSpots, seed=seed, seedType=seedType, seedAttr=seedAttr)
    elif type == 'icosahedronsurface':
        pileTuple = sp.calculate_sandpile_icosahedronsurface_seeded(xMax, yMax, numRows, grains, topple, dropSpots, seed=seed, seedType=seedType, seedAttr=seedAttr)

    end =  time()
    calc_time = end - start

    if type == 'cubesurface':
        toppleCtr = pileTuple[3]
        print(f'Calculation time = {calc_time} sec, Topples: {toppleCtr}.')

        sandbox_front = pileTuple[0][0]
        sandbox_back = pileTuple[0][1]
        sandbox_left = pileTuple[1][0]
        sandbox_right = pileTuple[1][1]
        sandbox_top = pileTuple[2][0]
        sandbox_bottom = pileTuple[2][1]

        np.save(f'saved_piles/{filenom}_front.npy', sandbox_front)
        np.save(f'saved_piles/{filenom}_back.npy', sandbox_back)
        np.save(f'saved_piles/{filenom}_left.npy', sandbox_left)
        np.save(f'saved_piles/{filenom}_right.npy', sandbox_right)
        np.save(f'saved_piles/{filenom}_top.npy', sandbox_top)
        np.save(f'saved_piles/{filenom}_bottom.npy', sandbox_bottom)

        arrW = (2 * xMax) + (2 * zMax)
        arrH = (2 * zMax) + yMax

        flatCube = np.zeros((arrW, arrH), dtype = np.int64)
        flatCubeBound = np.zeros((arrW, arrH), dtype = np.int64)

        numVertices = (2 * xMax * yMax) + (2 * xMax * zMax) + (2 * yMax * zMax)
        numEdges = (2 * (((xMax - 1) * yMax) + ((yMax - 1) * xMax))) + (4 * xMax)
        numEdges += (2 * (((xMax - 1) * zMax) + ((zMax - 1) * xMax))) + (4 * zMax)
        numEdges += (2 * (((yMax - 1) * zMax) + ((zMax - 1) * yMax))) + (4 * yMax)
        stableLimit = (2*numEdges) - numVertices
        print(f'Num vertices: {numVertices}, Num edges: {numEdges}, Stable Limit: {stableLimit}') # TODO: Is the number of edges just 2*numVertices?

        if drawDropSpots == True:
            colors.append(dropSpotColor)
            val = len(colors) - 1
            for drop in dropSpots:
                dx = drop[0]
                dy = drop[1]
                face = drop[3]
                if face == 'left':
                    sandbox_left[dx][dy] = val
                elif face == 'right':
                    sandbox_right[dx][dy] = val
                elif face == 'front':
                    sandbox_front[dx][dy] = val
                elif face == 'back':
                    sandbox_back[dx][dy] = val
                elif face == 'top':
                    sandbox_top[dx][dy] = val
                elif face == 'bottom':
                    sandbox_bottom[dx][dy] = val

        for x in range(zMax):
            for y in range(yMax):
                flatCube[x][y+zMax] = sandbox_left[x][y]
                flatCubeBound[x][y+zMax] = 1
        for x in range(xMax):
            for y in range(yMax):
                flatCube[x+zMax][y+zMax] = sandbox_front[x][y]
                flatCubeBound[x+zMax][y+zMax] = 1
        for x in range(zMax):
            for y in range(yMax):
                flatCube[x+zMax+xMax][y+zMax] = sandbox_right[x][y]
                flatCubeBound[x+zMax+xMax][y+zMax] = 1
        for x in range(xMax):
            for y in range(yMax):
                flatCube[x+(2*zMax)+xMax][y+zMax] = sandbox_back[x][y]
                flatCubeBound[x+(2*zMax)+xMax][y+zMax] = 1
        for x in range(xMax):
            for y in range(zMax):
                flatCube[x+zMax][y] = sandbox_top[x][y]
                flatCubeBound[x+zMax][y] = 1
        for x in range(xMax):
            for y in range(zMax):
                flatCube[x+zMax][y+zMax+yMax] = sandbox_bottom[x][y]
                flatCubeBound[x+zMax][y+zMax+yMax] = 1

        np.save(f'saved_piles/{filenom}_flatcube.npy', flatCube)
        np.save(f'saved_piles/{filenom}_flatcube_bound.npy', flatCubeBound)

        bound_front = None
        bound_back = None
        bound_left = None
        bound_right = None
        bound_top = None
        bound_bottom = None
        if (drawBounded == True or drawBounded == None) and len(pileTuple) == 7:
            bound_front = pileTuple[4][0]
            bound_back = pileTuple[4][1]
            bound_left = pileTuple[5][0]
            bound_right = pileTuple[5][1]
            bound_top = pileTuple[6][0]
            bound_bottom = pileTuple[6][1]

            np.save(f'saved_piles/{filenom}_front_bound.npy', bound_front)
            np.save(f'saved_piles/{filenom}_back_bound.npy', bound_back)
            np.save(f'saved_piles/{filenom}_left_bound.npy', bound_left)
            np.save(f'saved_piles/{filenom}_right_bound.npy', bound_right)
            np.save(f'saved_piles/{filenom}_top_bound.npy', bound_top)
            np.save(f'saved_piles/{filenom}_bottom_bound.npy', bound_bottom)

        sandpileImg = draw_sp.SandpileImg(xMax, yMax, sandbox_front, bound=bound_front, sink=None, filenom=f'images/{filenom}_front', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, bColor=bColor, bWidth=bWidth, bStyle=bStyle)
        sandpileImg.draw_sandbox()
        sandpileImg = draw_sp.SandpileImg(xMax, yMax, sandbox_back, bound=bound_back, sink=None, filenom=f'images/{filenom}_back', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, bColor=bColor, bWidth=bWidth, bStyle=bStyle)
        sandpileImg.draw_sandbox()
        sandpileImg = draw_sp.SandpileImg(zMax, yMax, sandbox_left, bound=bound_left, sink=None, filenom=f'images/{filenom}_left', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, bColor=bColor, bWidth=bWidth, bStyle=bStyle)
        sandpileImg.draw_sandbox()
        sandpileImg = draw_sp.SandpileImg(zMax, yMax, sandbox_right, bound=bound_right, sink=None, filenom=f'images/{filenom}_right', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, bColor=bColor, bWidth=bWidth, bStyle=bStyle)
        sandpileImg.draw_sandbox()
        sandpileImg = draw_sp.SandpileImg(xMax, zMax, sandbox_top, bound=bound_top, sink=None, filenom=f'images/{filenom}_top', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, bColor=bColor, bWidth=bWidth, bStyle=bStyle)
        sandpileImg.draw_sandbox()
        sandpileImg = draw_sp.SandpileImg(xMax, zMax, sandbox_bottom, bound=bound_bottom, sink=None, filenom=f'images/{filenom}_bottom', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, bColor=bColor, bWidth=bWidth, bStyle=bStyle)
        sandpileImg.draw_sandbox()
        sandpileImg = draw_sp.SandpileImg(arrW, arrH, flatCube, bound=flatCubeBound, sink=None, filenom=f'images/{filenom}_flat', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, bColor=bColor, bWidth=bWidth, bStyle=bStyle)
        sandpileImg.draw_sandbox()
    elif type == 'icosahedronsurface':
        # Hex Lattice Draws Triangles'
        pileArr = pileTuple[0]
        toppleCtr = pileTuple[1]
        print(f'Calculation time = {calc_time} sec, Topples: {toppleCtr}.')
        boundArr = pileTuple[2]

        for i in range(20):
            sandpile = pileArr[i]
            bound = boundArr[i]
            orientation = 'normal'
            if i % 2 == 0:
                orientation = 'inverted'
            #np.save(f'saved_piles/{filenom}_{i:02}.npy', sandpile)
            #np.save(f'saved_piles/{filenom}_{i:02}_bound.npy', bound)
            sandpileImg = draw_sp.SandpileImgExtended(xMax, yMax, sandpile, bound=bound, sink=None, filenom=f'images/{filenom}_{i:02}', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, lattice='hex', orientation=orientation)
            sandpileImg.draw_hex_sandbox()

        sandboxRaw = pileTuple[3]
        arrXMax = 2 * (xMax + 1)
        arrW = arrXMax + (5 * yMax) + 1
        arrH = 5 * yMax

        arrWRot = 5 * xMax + int(xMax / 2) + 9
        arrHRot = 3 * yMax

        stripSandpile = np.zeros((arrW, arrH), dtype = np.int64)
        stripSandpileBound = np.zeros((arrW, arrH), dtype = np.int64)
        stripSandpile_rot = np.zeros((arrWRot, arrHRot), dtype = np.int64)
        stripSandpileBound_rot = np.zeros((arrWRot, arrHRot), dtype = np.int64)

        for strip in range(5):
            yOffset = strip * yMax
            for y in range(yMax):
                xOffset = ((yMax-1) - y) + yOffset + 1
                for x in range(arrXMax):
                    x0 = x + xOffset
                    y0 = y + yOffset

                    x1 = (arrWRot - 3) - int(x/2) - y - ((strip * (xMax+1)))
                    y1 = ((yMax-1) - y) + int((x+1)/2)
                    stripSandpile_rot[x1][y1] = sandboxRaw[x][y+yOffset]
                    stripSandpileBound_rot[x1][y1] = 1

                    stripSandpile[x0][y0] = sandboxRaw[x][y+yOffset]
                    stripSandpileBound[x0][y0] = 1

        np.save(f'saved_piles/{filenom}_strips.npy', stripSandpile)
        np.save(f'saved_piles/{filenom}_strips_bound.npy', stripSandpileBound)
        np.save(f'saved_piles/{filenom}_strips_rot.npy', stripSandpile_rot)
        np.save(f'saved_piles/{filenom}_strips_rot_bound.npy', stripSandpileBound_rot)

        sandpileImg = draw_sp.SandpileImgExtended(arrW, arrH, stripSandpile, bound=stripSandpileBound, sink=None, filenom=f'images/{filenom}_flat', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, lattice='hex', orientation=orientation)
        sandpileImg.draw_hex_sandbox()
        sandpileImg = draw_sp.SandpileImgExtended(arrWRot, arrHRot, stripSandpile_rot, bound=stripSandpileBound_rot, sink=None, filenom=f'images/{filenom}_flat_rot', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, lattice='hex', orientation=orientation)
        sandpileImg.draw_hex_sandbox()
    else:
        toppleCtr = pileTuple[1]

        # Save sandpile calculation to numpy array file (binary file format)
        sandpile = pileTuple[0]
        sandpileGrains = np.sum(sandpile) # TODO: This does not take bound into account, also want to calculate average
        print(f'Calculation time = {calc_time} sec, Topples: {toppleCtr}, Sandpile Grains: {sandpileGrains}/{grains}.')

        bound = None
        if (drawBounded == True or drawBounded == None) and len(pileTuple) == 3:
            bound = pileTuple[2]
        np.save(f'saved_piles/{filenom}.npy', sandpile)
        if len(pileTuple) == 3:
            np.save(f'saved_piles/{filenom}_bound.npy', pileTuple[2])

        # Save sandpile image to PNG file
        sandpileImg = draw_sp.SandpileImg(xMax, yMax, sandpile, bound=bound, sink=sink, filenom=f'images/{filenom}', sideLength=imgPxWidth, colors=colors, bgColor=bgColor, bColor=bColor, bWidth=bWidth, bStyle=bStyle)
        sandpileImg.draw_sandbox()

        # Draw sandpile image to the screen using tkinter
        if drawOutput == True:
            sandpileTk = draw_sp.SandpileTk(xMax, yMax, sandpile, bound=bound, sink=sink, title=title, sideLength=tkPxWidth, colors=colors, bgColor=bgColor, bColor=bColor, bWidth=bWidth, bStyle=bStyle)
            sandpileTk.draw_sandbox()
            sandpileTk.main_loop()

# Command: python .\sandpile_surface_toppling.py .\pile_config\Cylinder_LineX_09drops_100k_01.json
if __name__ == '__main__':
	main()