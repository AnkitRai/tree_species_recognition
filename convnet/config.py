import os

base_path = os.getcwd()

# Inout Dimensions

imgRows = 800

imgCols =600

numChannels = 3

# Model parameters

dropout = True

learning_rate = 0.001

momentum = 0.9

figure_name = 'summary_diagnostigs'

# DATA Paths
INPUT_PATH = base_path + '/dataset/images/field/'

OUTPUT_FILE = 'leafsnap_data.csv'