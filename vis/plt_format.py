import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use(['seaborn-paper'])

mpl.rcParams['axes.axisbelow'] = 'True'
mpl.rcParams['axes.edgecolor'] = 'white'
mpl.rcParams['axes.facecolor'] = '#EAEAF2'
mpl.rcParams['axes.grid'] = 'True'
mpl.rcParams['grid.color'] = 'white'
mpl.rcParams['image.cmap'] = 'Greys'
mpl.rcParams['patch.facecolor'] = '#4C72B0'
mpl.rcParams['xtick.major.size'] = '0.0'
mpl.rcParams['ytick.major.size'] = '0.0'
mpl.rcParams['patch.facecolor'] = '#4C72B0'
mpl.rcParams['patch.facecolor'] = '#4C72B0'
mpl.rcParams['patch.facecolor'] = '#4C72B0'

mpl.rcParams['text.usetex'] = True

mpl.rcParams['axes.labelsize'] = '17'  # 8.8
mpl.rcParams['axes.titlesize'] = '18'  # 9.6
mpl.rcParams['xtick.labelsize'] = '14'  # 8.0
mpl.rcParams['ytick.labelsize'] = '14'  # 8.0
mpl.rcParams['legend.fontsize'] = '14'  # 8.0
# -----
# mpl.rcParams['figure.figsize'] = '12.8, 4.4'  # [6.4, 4.4]
#
# mpl.rcParams['axes.labelsize'] = '15'  # 8.8
# mpl.rcParams['axes.titlesize'] = '16'  # 9.6
# mpl.rcParams['xtick.labelsize'] = '14'  # 8.0
# mpl.rcParams['ytick.labelsize'] = '14'  # 8.0
# mpl.rcParams['legend.fontsize'] = '14'  # 8.0

if __name__ == '__main__':
    print(mpl.rcParams.keys())
