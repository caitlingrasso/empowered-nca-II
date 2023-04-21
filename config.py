import numpy as np

import constants
from visualizations import convert_to_high_res
import pandas as pd

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def targets(grid_size=constants.GRID_SIZE):
    
    targets = {}

    center = grid_size//2

    # Rectangle
    rect = np.zeros((grid_size, grid_size))
    width = grid_size//3 * 2
    height = grid_size//3
    rect[center-height//2:center+height//2,center-width//2:center+width//2+1] = 1
    targets['rectangle'] = rect

    # Square
    square = np.zeros((grid_size, grid_size))
    length = grid_size//2
    square[center-length//2:center+length//2+1, center-length//2:center+length//2+1]=1
    targets['square']=square

    # square_damage_coords = {
    #     'center': (center+1, center+1),
    #     'top':  (center+1-length//2,center+1),
    #     'bottom': (center+1+length//2,center+1),
    #     'right': (center+1, center+1+length//2),
    #     'left': (center+1, center+1-length//2),
    #     'top_right': (center+1-length//2,center+1+length//2),
    #     'top_left': (center+1-length//2, center+1-length//2),
    #     'bottom_right': (center+1+length//2, center+1+length//2),
    #     'bottom_left': (center+1+length//2, center+1-length//2)
    # }

    # Circle
    circle = np.zeros((grid_size, grid_size))
    radius = grid_size//4
    r2 = np.arange(-center, center + 1) ** 2 if grid_size % 2==1 else np.arange(-center, center) ** 2
    dist2 = r2[:, None] + r2
    circle[dist2 < radius**2] = 1
    targets['circle'] = circle

    # Xenobot
    xenobot = np.zeros((25, 25))
    xenobot[9:16, 6:19] = 1
    xenobot[15:21, 6:10] = 1
    xenobot[15:21, 15:19] = 1
    xenobot[9,6:8]=0
    xenobot[9, 17:19]=0
    xenobot[10, 6:19:12] = 0
    xenobot[20,6:8]=0
    xenobot[20, 17:19] = 0
    xenobot[19, 6:19:12] = 0
    xenobot[18, 6:19:12] = 0
    xenobot[20, 9:16:6] = 0
    xenobot[17, 10:15:4]=1
    xenobot[16, 10:12] = 1
    xenobot[16, 13:15] = 1
    targets['xenobot'] = xenobot

    # Biped 
    biped = np.zeros((grid_size,grid_size))
    length = grid_size//2
    biped_center_r = center +3
    biped[biped_center_r-length//2:biped_center_r+length//2+1, center-length//2:center+length//2+1]=1
    biped[biped_center_r:biped_center_r+length//2+1, center-length//4:center+length//4+1] = 0
    targets['biped']=biped

    # Triangle
    triangle = np.zeros((grid_size, grid_size))

    i=0
    # if x is between center-length//2:center+length//2 (for)
    # y goes from center-i:center+i+1
    for x in range(center-length//2,center+length//2):
        triangle[x,center-i:center+i+1]=1
        i+=1
    targets['triangle']=triangle

    # x
    x = np.zeros((grid_size, grid_size))
    try:
        x_arr = np.asarray(pd.read_csv('shapes/x.csv'))
    except:
        x_arr = np.asarray(pd.read_csv('../shapes/x.csv'))
    x[center-10:center+12, center-8:center+9] = x_arr
    targets['x'] = x

    return targets

def init(grid_size=constants.GRID_SIZE):
    center = grid_size//2

    # Initial grid
    init = np.zeros((grid_size, grid_size))
    init[center,center]=1

    return init
