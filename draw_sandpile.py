import tkinter as tk
import numpy as np
import math
import svgwrite
from PIL import Image, ImageColor, ImageDraw

# TODO: Add "light bright" (borders) feature to different lattices.
# TODO: Fix outer-edge border to be same thickness as inner borders on images.
# TODO: Add "corner borders (cross-stitch)" to tkinter render.
# TODO: Draw single line border (top, right, bottom, left borders)
# TODO: Add tkinter support for trihex piles

class SandpileTk():
    def __init__(self, xMax, yMax, sandpile, bound=None, sink=None, title="Abelian Sandpile", sideLength=2, colors=None, bgColor=None, width=None, height=None, bWidth=0, bColor='#000000', bStyle='all'):
        self.xMax = xMax
        self.yMax = yMax
        self.sideLength = sideLength

        if colors == None:
            colors = ['black', 'green', 'purple', 'gold']
        self.colors = colors
        numColors = len(colors)

        self.bgColor = bgColor
        if bgColor == None:
            self.bgColor = colors[0]
        self.borderWidth = int(bWidth) * 2
        self.borderColor = bColor
        self.borderStyle = bStyle

        modArray = np.full((xMax, yMax), numColors, dtype = np.uint32)
        modPile = np.mod(sandpile, modArray)
        self.sandpile = modPile

        if bound is None:
            bound = np.ones((xMax, yMax), dtype = np.uint8)
        self.bound = bound

        if sink is None:
            sink = np.zeros((xMax, yMax), dtype = np.uint8)
        self.sink = sink

        if width == None:
            width = (xMax*sideLength)
        if height == None:
            height = (yMax*sideLength)

        self.title = title
        self.root = tk.Tk()
        self.root.title(title)
        self.root.resizable(0, 0)
        self.canvas = tk.Canvas(self.root, width=width, height=height, bg=self.bgColor)
        self.canvas.pack()

    def main_loop(self):
        self.root.mainloop()

    def draw_pixel(self, x, y, val):
        off = self.sideLength + 1
        self.canvas.create_rectangle(x+1, y+1, x+off, y+off, outline=self.borderColor, fill=self.colors[val], width=self.borderWidth)

    def draw_sandbox(self):
        for x in range(0, self.xMax):
            for y in range(0, self.yMax):
                drawPx = (self.sink[x, y] == 0) and (self.bound[x, y] == 1)
                if drawPx == True:
                    self.draw_pixel(x*self.sideLength, y*self.sideLength, self.sandpile[x, y])
        self.root.update()

class SandpileTkExtended(SandpileTk):
    def __init__(self, xMax, yMax, sandpile, bound=None, title="AbelianSandpile", sideLength=2, colors=None, bgColor=None, lattice='square'):
        width = xMax * sideLength
        height = yMax * sideLength
        if lattice == 'hex':
            width = math.ceil(xMax / 2.0) * sideLength
            height = yMax * (math.sqrt(3) / 2.0) * sideLength
        elif lattice == 'tri':
            width = (1.5 * xMax * sideLength) + sideLength
            height = (math.sqrt(3.0) * sideLength * yMax) + ((math.sqrt(3.0) / 2.0) * sideLength)
        super().__init__(xMax, yMax, sandpile, bound=bound, title=title, sideLength=sideLength, colors=colors, bgColor=bgColor, width=width, height=height)
    
    def draw_down_triangle(self, x, y, val):
        x1 = x + (self.sideLength / 2.0)
        x2 = x + self.sideLength
        y1 = y + ((math.sqrt(3.0) / 2.0) * self.sideLength)
        self.canvas.create_polygon([x, y, x2, y, x1, y1], outline=None, fill=self.colors[val], width=0)

    def draw_up_triangle(self, x, y, val):
        x1 = x + (self.sideLength / 2.0)
        x2 = x + self.sideLength
        y1 = y + ((math.sqrt(3.0) / 2.0) * self.sideLength)
        self.canvas.create_polygon([x, y1, x2, y1, x1, y], outline=None, fill=self.colors[val], width=0)

    def draw_hexagon(self, x1, y1, val):
        x0 = x1 - (0.5 * self.sideLength)
        x2 = x1 + self.sideLength
        x3 = x1 + (1.5 * self.sideLength)
        y2 = y1 + ((math.sqrt(3.0) / 2.0) * self.sideLength)
        y3 = y1 + (math.sqrt(3.0) * self.sideLength)
        self.canvas.create_polygon([x0, y2, x1, y1, x2, y1, x3, y2, x2, y3, x1, y3], outline=None, fill=self.colors[val], width=0)

    def draw_hex_sandbox(self):
        for x in range(0, self.xMax):
            for y in range(0, self.yMax):
                drawPx = (self.sink[x, y] == 0) and (self.bound[x, y] == 1)
                if drawPx == True:
                    x1 = x * (self.sideLength / 2.0)
                    y1 = y * (math.sqrt(3.0) / 2.0) * self.sideLength
                    if (x+y) % 2 == 0:
                        self.draw_down_triangle(x1, y1, self.sandpile[x, y])
                    else:
                        self.draw_up_triangle(x1, y1, self.sandpile[x, y])
        self.root.update()

    def draw_tri_sandbox(self):
        for x in range(0, self.xMax):
            for y in range(0, self.yMax):
                drawPx = (self.sink[x, y] == 0) and (self.bound[x, y] == 1)
                if drawPx == True:
                    x1 = (x * 1.5 * self.sideLength) + (0.5 * self.sideLength)
                    y1 = (y * math.sqrt(3.0) * self.sideLength) + ((x % 2) * ((math.sqrt(3.0) / 2.0) * self.sideLength))
                    self.draw_hexagon(x1, y1, self.sandpile[x, y])
        self.root.update()

class SandpileImg():
    def __init__(self, xMax, yMax, sandpile, bound=None, sink=None, filenom="AbelianSandpile", sideLength=2, colors=None, width=None, height=None, bgColor=None, bWidth=0, bColor='#000000', bStyle='all'):
        self.xMax = xMax
        self.yMax = yMax
        self.sideLength = sideLength

        if colors == None:
            colors = ['black', 'green', 'purple', 'gold']
        self.colors = colors
        self.imgColors = []
        numColors = len(colors)

        for rawColor in colors:
            self.imgColors.append(ImageColor.getcolor(rawColor, 'RGBA'))
        
        self.bgColor = bgColor
        if bgColor == None:
            self.bgColor = colors[0]
        self.borderWidth = int(bWidth)
        self.borderColor = ImageColor.getcolor(bColor, 'RGBA')
        self.borderStyle = bStyle

        modArray = np.full((xMax, yMax), numColors, dtype = np.uint32)
        modPile = np.mod(sandpile, modArray)
        self.sandpile = modPile

        if bound is None:
            bound = np.ones((xMax, yMax), dtype = np.uint8)
        self.bound = bound

        if sink is None:
            sink = np.zeros((xMax, yMax), dtype = np.uint8)
        self.sink = sink

        if width == None:
            width = (xMax*sideLength)
        if height == None:
            height = (yMax*sideLength)

        self.imgFilenom = filenom + '.png'
        self.imgFile = Image.new('RGBA', (math.ceil(width), math.ceil(height)), self.bgColor)

    def draw_pixel(self, x, y, val):
        offX = x * self.sideLength
        offY = y * self.sideLength
        bMin = 0 + self.borderWidth
        bMax = self.sideLength - self.borderWidth
        for i in range(0, self.sideLength):
            for j in range(0, self.sideLength):
                if self.borderStyle == 'all':
                    if i < bMin or i >= bMax or j < bMin or j >= bMax:
                        self.imgFile.putpixel((offX+i, offY+j), self.borderColor)
                    else:
                        self.imgFile.putpixel((offX+i, offY+j), self.imgColors[val])
                elif self.borderStyle == 'corners':
                    if (i >= bMin and i < bMax) or (j >= bMin and j < bMax):
                        self.imgFile.putpixel((offX+i, offY+j), self.imgColors[val])
                    else:
                        self.imgFile.putpixel((offX+i, offY+j), self.borderColor)

    def draw_sandbox(self):
        for x in range(0, self.xMax):
            for y in range(0, self.yMax):
                drawPx = (self.sink[x, y] == 0) and (self.bound[x, y] == 1)
                if drawPx == True:
                    self.draw_pixel(x, y, self.sandpile[x, y])
        self.imgFile.save(self.imgFilenom, dpi=(300, 300))

class SandpileImgExtended(SandpileImg):
    def __init__(self, xMax, yMax, sandpile, bound=None, filenom="AbelianSandpile", sideLength=2, colors=None, sink=None, bgColor=None, lattice='square', orientation='normal'):
        width = xMax * sideLength
        height = yMax * sideLength
        self.orientation = orientation
        if lattice == 'hex':
            width = math.ceil(xMax / 2.0) * sideLength
            height = yMax * (math.sqrt(3) / 2.0) * sideLength
        elif lattice == 'tri':
            width = (1.5 * xMax * sideLength) + sideLength
            height = (math.sqrt(3.0) * sideLength * yMax) + ((math.sqrt(3.0) / 2.0) * sideLength)
        elif lattice == 'trihex':
            width = xMax * (math.sqrt(3) / 2.0) * sideLength
            height = (yMax * 0.75 * sideLength) + sideLength # Approx.
        super().__init__(xMax, yMax, sandpile, bound=bound, sink=sink, filenom=filenom, sideLength=sideLength, colors=colors, width=width, height=height, bgColor=bgColor)
        self.draw = ImageDraw.Draw(self.imgFile)
    
    def draw_down_triangle(self, x, y, val):
        x1 = x + (self.sideLength / 2.0)
        x2 = x + self.sideLength
        y1 = y + ((math.sqrt(3.0) / 2.0) * self.sideLength)
        self.draw.polygon(((x,y),(x2,y),(x1,y1)), fill=self.imgColors[val])

    def draw_up_triangle(self, x, y, val):
        x1 = x + (self.sideLength / 2.0)
        x2 = x + self.sideLength
        y1 = y + ((math.sqrt(3.0) / 2.0) * self.sideLength)
        self.draw.polygon(((x,y1),(x2,y1),(x1,y)), fill=self.imgColors[val])

    def draw_hexagon(self, x1, y1, val):
        x0 = x1 - (0.5 * self.sideLength)
        x2 = x1 + self.sideLength
        x3 = x1 + (1.5 * self.sideLength)
        y2 = y1 + ((math.sqrt(3.0) / 2.0) * self.sideLength)
        y3 = y1 + (math.sqrt(3.0) * self.sideLength)
        self.draw.polygon(((x0,y2),(x1,y1),(x2,y1),(x3,y2),(x2,y3),(x1,y3)), fill=self.imgColors[val])

    def draw_horiz_diamond(self, x, y, val):
        x1 = x + ((math.sqrt(3.0) / 2.0) * self.sideLength)
        x2 = x + (math.sqrt(3.0) * self.sideLength)
        y1 = y - (0.5 * self.sideLength)
        y2 = y + (0.5 * self.sideLength)
        self.draw.polygon(((x,y),(x1,y1),(x2,y),(x1,y2)), fill=self.imgColors[val])
    
    def draw_left_diamond(self, x, y, val):
        x1 = x + ((math.sqrt(3.0) / 2.0) * self.sideLength)
        y1 = y - (0.5 * self.sideLength)
        y2 = y + (0.5 * self.sideLength)
        y3 = y + self.sideLength
        self.draw.polygon(((x,y),(x1,y1),(x1,y2),(x,y3)), fill=self.imgColors[val])

    def draw_right_diamond(self, x, y, val):
        x1 = x + ((math.sqrt(3.0) / 2.0) * self.sideLength)
        y1 = y + (0.5 * self.sideLength)
        y2 = y + (self.sideLength)
        y3 = y + (1.5 * self.sideLength)
        self.draw.polygon(((x,y),(x1,y1),(x1,y3),(x,y2)), fill=self.imgColors[val])

    def draw_hex_sandbox(self):
        downTriangle = 0
        if self.orientation == 'inverted':
            downTriangle = 1
        for x in range(0, self.xMax):
            for y in range(0, self.yMax):
                drawPx = (self.sink[x, y] == 0) and (self.bound[x, y] == 1)
                if drawPx == True:
                    x1 = x * (self.sideLength / 2.0)
                    y1 = y * (math.sqrt(3.0) / 2.0) * self.sideLength
                    if (x+y) % 2 == downTriangle:
                        self.draw_down_triangle(x1, y1, self.sandpile[x, y])
                    else:
                        self.draw_up_triangle(x1, y1, self.sandpile[x, y])
        self.imgFile.save(self.imgFilenom, dpi=(300, 300))

    def draw_tri_sandbox(self):
        for x in range(0, self.xMax):
            for y in range(0, self.yMax):
                drawPx = (self.sink[x, y] == 0) and (self.bound[x, y] == 1)
                if drawPx == True:
                    x1 = (x * 1.5 * self.sideLength) + (0.5 * self.sideLength)
                    y1 = (y * math.sqrt(3.0) * self.sideLength) + ((x % 2) * ((math.sqrt(3.0) / 2.0) * self.sideLength))
                    self.draw_hexagon(x1, y1, self.sandpile[x, y])
        self.imgFile.save(self.imgFilenom, dpi=(300, 300))
    
    def draw_tri_hex_sandbox(self):
        for y in range(self.yMax):
            xStart = 0
            xStep = 1
            if y % 4 == 1:
                xStart = 0
                xStep = 2
            if y % 4 == 3:
                xStart = 1
                xStep = 2
            for x in range(xStart, self.xMax, xStep):
                drawPx = (self.sink[x, y] == 0) and (self.bound[x, y] == 1)
                if drawPx == True:
                    x1 = x * ((math.sqrt(3.0) / 2.0) * self.sideLength)
                    y1 = y * self.sideLength - (int(y/4) * self.sideLength)
                    if x % 2 == 0:
                        y1 = y1 + (0.5 * self.sideLength)
                        if y % 4 == 2:
                            y1 = y1 - self.sideLength

                    if y % 2 == 1:
                        self.draw_horiz_diamond(x1, y1, self.sandpile[x, y])
                    if y % 4 == 0:
                        if x % 2 == 0:
                            self.draw_left_diamond(x1, y1, self.sandpile[x, y])
                        else:
                            self.draw_right_diamond(x1, y1, self.sandpile[x, y])
                    elif y % 4 == 2:
                        if x % 2 == 1:
                            self.draw_left_diamond(x1, y1, self.sandpile[x, y])
                        else:
                            self.draw_right_diamond(x1, y1, self.sandpile[x, y])

        self.imgFile.save(self.imgFilenom, dpi=(300, 300))

class SandpileSvg():
    def __init__(self, xMax, yMax, sandpile, bound=None, sink=None, filenom="AbelianSandpile", sideLength=2, colors=None, width=None, height=None, bgColor=None, bWidth=0, bColor='#000000', bStyle='all'):
        self.xMax = xMax
        self.yMax = yMax
        self.sideLength = sideLength

        if colors == None:
            colors = ['black', 'green', 'purple', 'gold']
        self.colors = colors
        numColors = len(colors)
        
        self.bgColor = bgColor
        if bgColor == None:
            self.bgColor = colors[0]
        self.borderWidth = int(bWidth)
        self.borderColor = bColor
        self.borderStyle = bStyle

        modArray = np.full((xMax, yMax), numColors, dtype = np.uint32)
        modPile = np.mod(sandpile, modArray)
        self.sandpile = modPile

        if bound is None:
            bound = np.ones((xMax, yMax), dtype = np.uint8)
        self.bound = bound

        if sink is None:
            sink = np.zeros((xMax, yMax), dtype = np.uint8)
        self.sink = sink

        if width == None:
            width = (xMax*sideLength)
        if height == None:
            height = (yMax*sideLength)

        self.imgFilenom = filenom + '.svg'
        self.dwg = svgwrite.Drawing(self.imgFilenom, profile='full', size=(math.ceil(width), math.ceil(height)))
        self.dwg.viewbox(minx=0, miny=0, width=math.ceil(width), height= math.ceil(height))
        self.dwg.add(self.dwg.rect(insert=(0,0), size=(math.ceil(width), math.ceil(height)), fill=self.bgColor, stroke_width=0))
        self.body = self.dwg.add(self.dwg.g(id=f'body', shape_rendering='crispEdges', stroke_width=0))

        # Create layers for each color
        #self.body = []
        #for i in range(0, numColors):
            #self.body.append(self.dwg.add(self.dwg.g(id=f'body{i}', fill=self.colors[i], shape_rendering='crispEdges', stroke_width=0)))

    def draw_pixel(self, x, y, val):
        offX = x * self.sideLength
        offY = y * self.sideLength
        bMin = 0 + self.borderWidth
        bMax = self.sideLength - self.borderWidth
        self.body.add(self.dwg.rect(insert=(offX, offY), size=(self.sideLength, self.sideLength), fill=self.colors[val]))

        # Add rect or path to layer for specific color
        #self.body[val].add(self.dwg.rect(insert=(offX, offY), size=(self.sideLength, self.sideLength)))
        #self.body[val].add(self.dwg.path(f'M{offX} {offY}h{self.sideLength}v{self.sideLength}h{-self.sideLength}z')) # Saves SVG file size, but render time seems slower

    def draw_sandbox(self):
        for x in range(0, self.xMax):
            for y in range(0, self.yMax):
                drawPx = (self.sink[x, y] == 0) and (self.bound[x, y] == 1)
                if drawPx == True:
                    self.draw_pixel(x, y, self.sandpile[x, y])
        self.dwg.saveas(self.imgFilenom, pretty=False, indent=4)