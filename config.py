'''
define some game parameters
'''
# Game configuration parameters
MAP_WIDTH = 1300 #600
MAP_HEIGHT = 500 #500
TILE_SIZE = 20
TITLE = "FCLappyBird"
FPS = 60
IMG_DIR = './img'
SND_DIR = './snd'
FONT_NAME = 'Arial'
FONT_SIZE = 20
WHITE = (255, 255, 255)

JUMP_SPEED = -3 #-3.5     # once the bird flaps, its speed becomes this value
GRAVITY_ACC = 0.35
BIRD_X_SPEED = 3   # the const horizontal speed of the bird
BIRD_MAX_Y_SPEED = 5    # the maximum downward speed

# horizontal space between two adjacent pairs of pipes
MIN_PIPE_SPACE = 250
MAX_PIPE_SPACE = 300
# gap (vertical space) between a pair of pipes
MIN_PIPE_GAP = 150 #120
MAX_PIPE_GAP = 220 #200
# minimum length of a pipe
MIN_PIPE_LENGTH = 100

# FCL SETTINGS
NEURONSPERLAYER_FCL = [4, 8, 3] # number of neurons in each layer for fcl
NEURONSPERLAYER_FCL2 = [4, 8, 4] # number of neurons in each layer for fcl2
NEURONSPERLAYER_FCL3 = [4, 8, 5] # number of neurons in each layer for fcl3

LEARNING_RATE = 0.000001#0.0000007 #0.0000017#0.0000008 #0.0001 # learning rate 
# note that the learning rate is set as 0.0001 for fcl2
# and 0.001 for fcl
MOMENTUM = 0.5

# for reproduction by setting an integer value; otherwise, set `None`
RANDOM_SEED = 14256