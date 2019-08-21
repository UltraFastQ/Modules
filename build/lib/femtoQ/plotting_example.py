# -*- coding: utf-8 -*-
"""
Created on Mon Dec 10 13:48:34 2018

@author: Jan
"""

import sys
import numpy as np
import numpy.matlib as matlib
sys.path.append('../')
import femtoQ.plotting as fqp

figure_number = 1
figure_size = (18, 8) # in cm, for presentation 16:9 the size is 33.8:19
axis_pos = fqp.generate_rectangular_axes_pos((.11, .6), (.21))
axis_size = matlib.repmat((.35, .6), 2, 1)
axis_xlim = ([-10, 700], [1,2])
axis_ylim = ([-1.2, 1.5], [0,3])
axis_xlabel = ('time delay $t_d$ (fs)', 'energy (eV)')
axis_ylabel = (r'$\Delta E(h\nu_{max})$ (meV)', 'test')

# %%
fig, ax = fqp.create_axes(figure_number, figure_size, axis_pos, axis_size, axis_xlabel, axis_ylabel, axis_xlim, axis_ylim)

x = np.linspace(-100, 1000, 1000);
y = np.sin(x/200*2*np.pi)

ax[0].plot(x,y)

x2 = np.linspace(1, 2, 100);
y2 = 2*(x2-1)
ax[1].plot(x2,y2)

tax = fqp.create_top_axis(ax[1], 'ev2nm', np.arange(400, 1300, 250), np.arange(400, 1300, 50))

tax[0].set_xlabel('wavelength (nm)')
tax[0].set_xlim(ax[1].get_xlim())
# %%
fig.savefig('test_py.png', dpi=600)
fig.savefig('test_py.pdf', dpi=600)
fig.savefig('test_py.eps', dpi=600)
