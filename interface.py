import pygame as py
from network import Neural_Net
from rich import print

py.init()

side = 784
col = (100, 100, 100)
size = (28, 28)

screen = py.display.set_mode((side, side))
screen.fill((0, 0, 0))

running = True
mouse = py.mouse
y = (0, 0, 0)

pixels = {}
data = [0 for _ in range(784)]

# for i in range(29):
#         py.draw.line(screen, "white", (0, 28*i), (side, 28*i))
#         py.draw.line(screen, "white", (28*i, 0), (28*i, side))

def find_coord(pos: tuple):
    checks = [False, False]
    for i in range(0, 784, 28):   
        if not checks[0] and (pos[0] < i + 28 and pos[0] >= i):
            x = i
            checks[0] = True

        if not checks[1] and (pos[1] < i + 28 and pos[1] >= i):
               y = i
               checks[1] = True
        
        if checks == [True, True]:
             break

    return (x, y)

def color_pixel(pix):
    if pix not in pixels:
        pixels[pix] = col
        return col
    else:
        c = tuple([a+b for a,b in zip(pixels[pix], col)])
        if c[0] > 255:
             c = (255, 255, 255)
        pixels[pix] = c 
        return c

def update_data():
    for key, value in pixels.items():
        # print(key)
        row, col = key
        ind = int(row/28) + col
        data[ind] = value[0]/255


nn = Neural_Net()
nn.load("nn_model.json")


drawing = False
while running:
    for event in py.event.get():
        if event.type == py.QUIT:
            running = False

        if event.type == py.MOUSEBUTTONDOWN:
            drawing = True
        
        if event.type == py.MOUSEBUTTONUP:
            drawing = False

        if event.type == py.MOUSEMOTION:
            if drawing:
                y = tuple([a+b for a,b in zip(y, col)])
                pos = find_coord(mouse.get_pos())
                colors = color_pixel(pos)
                screen.fill(colors, (*pos, *size))
        
        if event.type == py.KEYDOWN:
            if event.key == py.K_r:
                screen.fill((0, 0, 0))
                for pix in pixels:
                    pixels[pix] = (0, 0, 0)

            if event.key == py.K_p:
                update_data()
                prediction = nn.feed_forward(data)
                confidence = max(prediction)
                number = prediction.index(confidence)
                py.display.set_caption(f"Guess: {number}, Confidence: {round(confidence, 3) * 100}")
                # print(data, len(data))

    
    py.display.flip()

py.quit()
