# NCA parameters
ITERATIONS = 50 
GRID_SIZE = 25

# Evolution parameters
POP_SIZE = 400
GENERATIONS = 2000
SAVE_ALL = False
objectives = ['error', 'error_phase1', 'error_phase2', 'MI']

# NCA parameters - constant
N_CHANNELS = 2
N_HIDDEN_NODES = 0
N_HIDDEN_LAYERS = 0
NEIGHBORHOOD = 4 # Von Neumann (4) vs. Moore (8)
CONTINUOUS_SIGNALING = False 
MEMORY = False
SIGNAL_NOISE = False
SIGNAL_NOISE_STD = 0.3
SIGNALING_ARTIFACT = False # default = True

# Info metric parameters
HISTORY_LENGTH = 1
SEQUENCE_LENGTH = ITERATIONS-45 # 45 is max history length

# Flags
DIFFUSE = True
DIFFUSION_RATE = 0.5  # 50% of signal diffuses out of the cell each time step

# Visualizations
CMAP_CELLS = 'binary'
CMAP_SIGNAL = 'gray'
CMAP_EMPOWERMENT = 'Blues'
TARGET_OUTLINE_COLOR = 'green'
CELL_STATE_OUTLINE_COLOR = 'm'
color_treatment_dict = {"random" : "k",
                        "error_phase1_error_phase2" : "tab:orange",
                        "error" : "tab:blue",
                        "error_MI" : "tab:green",
                        "MI" : "tab:purple",
                        "grad": "tab:cyan"
                        }