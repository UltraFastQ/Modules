# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 16:52:35 2018

@author: Jan
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rcParams
import numpy as np

def isiterable(p_object):
    try:
        it = iter(p_object)
    except TypeError:
        return False
    return True

def cm2inch(*tupl):
    inch = 2.54
    if isinstance(tupl[0], tuple):
        return tuple(i/inch for i in tupl[0])
    else:
        return tuple(i/inch for i in tupl)


def transform_to_axes_quadruple(axis_pos, axis_size):
    if len(axis_size) < len(axis_pos):
        raise ValueError('The size of not all the axis positions is specified.')

    axis_quad = []
    for i, pos in enumerate(axis_pos):
        axis_quad.append([pos[0], pos[1], axis_size[i][0], axis_size[i][1]])
    return axis_quad


def generate_rectangular_axes_pos(pos_x, pos_y):
    if type(pos_x) is float:
        pos_x = [pos_x]
    else:
        pos_x = list(pos_x)
        pos_x.sort()

    if type(pos_y) is float:
        pos_y = [pos_y]
    else:
        pos_y = list(pos_y)
        pos_y.sort()

    axis_pos = []
    for x in pos_x:
        for y in pos_y:
            axis_pos.append([x, y])
    return axis_pos


def set_default_values_rpl():
    print('Setting default plotting values for RPL')
    plt.rcdefaults()
#   Reopen all you figures once you changed something here!
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 11
    rcParams['axes.linewidth'] = 1
    rcParams['figure.dpi'] = 200
    rcParams['lines.linewidth'] = 1
    rcParams['xtick.top'] = True
    rcParams['ytick.right'] = True
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.minor.visible'] = True
    rcParams['ytick.minor.visible'] = True
    rcParams['xtick.major.size'] = 5
    rcParams['xtick.minor.size'] = 3
    rcParams['xtick.major.width'] = 1
    rcParams['xtick.minor.width'] = 1
    rcParams['ytick.major.size'] = rcParams['xtick.major.size']
    rcParams['ytick.minor.size'] = rcParams['xtick.minor.size']
    rcParams['ytick.major.width'] = rcParams['xtick.major.width']
    rcParams['ytick.minor.width'] = rcParams['xtick.minor.width']
    rcParams['mathtext.default'] = 'regular'
    rcParams["legend.frameon"] = False
    rcParams["legend.handlelength"] = 1
    rcParams["legend.handletextpad"] = .5
    rcParams["legend.columnspacing"] = .7

def set_default_values_cleo():
    print('Setting default plotting values for CLEO')
    plt.rcdefaults()
#   Reopen all you figures once you changed something here!
    rcParams['font.family'] = 'serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 9
    rcParams['axes.linewidth'] = 1
    rcParams['figure.dpi'] = 200
    rcParams['lines.linewidth'] = 1
    rcParams['xtick.top'] = True
    rcParams['ytick.right'] = True
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.minor.visible'] = True
    rcParams['ytick.minor.visible'] = True
    rcParams['xtick.major.size'] = 5
    rcParams['xtick.minor.size'] = 3
    rcParams['xtick.major.width'] = 1
    rcParams['xtick.minor.width'] = 1
    rcParams['ytick.major.size'] = rcParams['xtick.major.size']
    rcParams['ytick.minor.size'] = rcParams['xtick.minor.size']
    rcParams['ytick.major.width'] = rcParams['xtick.major.width']
    rcParams['ytick.minor.width'] = rcParams['xtick.minor.width']
    rcParams['mathtext.rm'] = 'serif'
    rcParams['mathtext.default'] = 'regular'
    rcParams['mathtext.fontset'] = 'dejavuserif'
    rcParams["legend.frameon"] = False
    rcParams["legend.handlelength"] = 1
    rcParams["legend.handletextpad"] = .5
    rcParams["legend.columnspacing"] = .7

def set_default_values_presentation():
    print('Setting default plotting values for presentations')
    plt.rcdefaults()
#   Reopen all you figures once you changed something here!
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Tahoma']
    rcParams['font.serif'] = ['Times New Roman']
    rcParams['font.size'] = 18
    rcParams['axes.linewidth'] = 2
    #rcParams['figure.dpi'] = 100
    rcParams['lines.linewidth'] = 2
    rcParams['xtick.top'] = True
    rcParams['ytick.right'] = True
    rcParams['xtick.direction'] = 'in'
    rcParams['ytick.direction'] = 'in'
    rcParams['xtick.minor.visible'] = True
    rcParams['ytick.minor.visible'] = True
    rcParams['xtick.major.size'] = 7
    rcParams['xtick.minor.size'] = 5
    rcParams['xtick.major.width'] = 2
    rcParams['xtick.minor.width'] = rcParams['xtick.major.width']
    rcParams['ytick.major.size'] = rcParams['xtick.major.size']
    rcParams['ytick.minor.size'] = rcParams['xtick.minor.size']
    rcParams['ytick.major.width'] = rcParams['xtick.major.width']
    rcParams['ytick.minor.width'] = rcParams['xtick.minor.width']
    rcParams['mathtext.default'] = 'regular'
    rcParams['mathtext.fontset'] = 'dejavuserif'
    rcParams["legend.frameon"] = False
    rcParams["legend.handlelength"] = 1
    rcParams["legend.handletextpad"] = .5
    rcParams["legend.columnspacing"] = .7
    rcParams['figure.figsize'] = 6*1.4, 6

def create_axes(figure_number, figure_size, axis_pos, axis_size, axis_xlabel, axis_ylabel, axis_xlim, axis_ylim, label_offset = 0):
    set_default_values_presentation()
    fig = plt.figure(figure_number)
    fig.clear()
    fig.set_size_inches(cm2inch(figure_size), forward=True)

    axis_quad = transform_to_axes_quadruple(axis_pos, axis_size)

    ax = []
    for i, item in enumerate(axis_quad):
        ax.append(fig.add_axes(item))

        if(type(axis_xlabel) is str and i == 0):
            ax[-1].set_xlabel(axis_xlabel)
        elif(type(axis_xlabel) is not str and len(axis_xlabel) > i):
            ax[-1].set_xlabel(axis_xlabel[i])

        if(type(axis_ylabel) is str and i == 0):
            ax[-1].set_ylabel(axis_ylabel)
        elif(type(axis_ylabel) is not str and len(axis_xlabel) > i):
            ax[-1].set_ylabel(axis_ylabel[i])

#       Empty label removes also ticklabels
        if(ax[-1].get_xlabel() == ''):
            ax[-1].tick_params(labelbottom=False)
        if(ax[-1].get_ylabel() == ''):
            ax[-1].tick_params(labelleft=False)

        if(len(axis_xlim) > i):
            if(isiterable(axis_xlim[i]) and len(axis_xlim[i]) == 2):
                ax[-1].set_xlim(axis_xlim[i], auto=False)
            elif i == 0:
                ax[-1].set_xlim(axis_xlim, auto=False)

        if(len(axis_ylim) > i):
            if(isiterable(axis_ylim[i]) and len(axis_ylim[i]) == 2):
                ax[-1].set_ylim(axis_ylim[i], auto=False)
            elif i == 0:
                ax[-1].set_ylim(axis_ylim, auto=False)
            ax[-1].autoscale(False)

        if(len(axis_quad) > 1):
            ax[-1].annotate("("+chr(97+i+label_offset)+")",(.05, .85), xycoords='axes fraction')


    ax = tuple(ax)
    return(fig, ax)


def create_top_axis(axes, conversion, major, minor):
    if type(axes) is mpl.axes._axes.Axes:
        axes = (axes, ) #make it a tuple so indexing works

    tax = [None] * len(axes)
    for i, ax in enumerate(axes):
        tax[i] = ax.twiny()
        for axis in ['top', 'bottom', 'left', 'right']:
            tax[i].spines[axis].set_linewidth(ax.spines[axis].get_linewidth())
        if conversion == 'ev2nm':
            new_major = 1240./major
            new_minor = 1240./minor

        ind = np.argsort(new_major)
        new_major = new_major[ind]
        major = major[ind]
        new_minor.sort()
        tax[i].set_xticks(new_minor, minor=True)
        tax[i].set_xticks(new_major)
        tax[i].set_xticklabels(list(map(str, major)))
#        This needs to be at the end, at least after setting the ticks!
        tax[i].autoscale(False)
        tax[i].set_xlim(ax.get_xlim())
    return tax

def create_side_axis(axes, conversion, major, minor):
    if type(axes) is mpl.axes._axes.Axes:
        axes = (axes, ) #make it a tuple so indexing works

    tax = [None] * len(axes)
    for i, ax in enumerate(axes):
        tax[i] = ax.twinx()
        for axis in ['top', 'bottom', 'left', 'right']:
            tax[i].spines[axis].set_linewidth(ax.spines[axis].get_linewidth())
        if conversion == 'ev2nm':
            new_major = 1240./major
            new_minor = 1240./minor

        ind = np.argsort(new_major)
        new_major = new_major[ind]
        major = major[ind]
        new_minor.sort()
        tax[i].set_yticks(new_minor, minor=True)
        tax[i].set_yticks(new_major)
        tax[i].set_yticklabels(list(map(str, major)))
#        This needs to be at the end, at least after setting the ticks!
        tax[i].autoscale(False)
        tax[i].set_xlim(ax.get_xlim())
    return tax
