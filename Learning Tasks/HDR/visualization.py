import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib as mpl
import cv2
import plotly.graph_objects as go
from scipy.constants import h, c, k
from HDR.temperature.colour_system import cs_hdtv

relative_error_max = 0.5


def visual_divide_x(input, prediction, target, factor, title=None, output=None):
    fig = plt.figure(frameon=False)
    rows = 2
    columns = 3
    if title is not None:
        plt.title(title)
        plt.axis('off')


    fig.add_subplot(rows, columns, 1)
    plt.imshow((input*255).astype(np.uint8))
    plt.axis('off')
    plt.title("input")

    fig.add_subplot(rows, columns, 2)
    plt.imshow((np.clip(target,0,1)*255).astype(np.uint8))
    plt.axis('off')
    plt.title("real_pano")

    fig.add_subplot(rows, columns, 3)
    plt.imshow((np.clip(prediction,0,1)*255).astype(np.uint8))
    plt.axis('off')
    plt.title("output_pano")

    #/factor
    fig.add_subplot(rows, columns, 4)
    plt.imshow((input/factor*255).astype(np.uint8))
    plt.axis('off')
    plt.title("input/" + str(factor))

    fig.add_subplot(rows, columns, 5)
    plt.imshow((np.clip(target/factor,0,1)*255).astype(np.uint8))
    plt.axis('off')
    plt.title("real_pano/" + str(factor))

    fig.add_subplot(rows, columns, 6)
    plt.imshow((np.clip(prediction/factor,0,1)*255).astype(np.uint8))
    plt.axis('off')
    plt.title("output_pano/" + str(factor))

    fig.tight_layout()

    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')

    return fig

def luminanceFalseColors(input, prediction, target, output=None):
    fig = plt.figure(frameon=False)
    rows = 1
    columns = 3

    target_bw = cv2.cvtColor(target, cv2.COLOR_RGB2GRAY)
    prediction_bw = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)

    #vmax = np.max([target_bw,prediction_bw])
    #vmin = np.min([target_bw,prediction_bw])
    #norm=colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    norm=colors.LogNorm(vmin=None, vmax=None, clip=True)
    cmap = 'nipy_spectral'

    fig.add_subplot(rows, columns, 1)
    plt.imshow((input*255).astype(np.uint8))
    plt.axis('off')
    plt.title("input")

    ax = fig.add_subplot(rows, columns, 2)
    plt.pcolormesh(target_bw[::-1,:], cmap=cmap, norm=norm)
    ax.set_aspect(aspect='equal')
    plt.axis('off')
    plt.title("real_pano")

    ax = fig.add_subplot(rows, columns, 3)
    fcm = plt.pcolormesh(prediction_bw[::-1,:], cmap=cmap, norm=norm)
    ax.set_aspect(aspect='equal')
    plt.axis('off')
    plt.title("output_pano")

    cb = plt.colorbar(fcm, shrink=.8, aspect=25,fraction=0.046, pad=0.04)
    fig.tight_layout()

    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')

    return fig


def distributionBoxPlotMatplot(metrics, title=None, output=None):
    plt.figure()
    if title:
        plt.title(title)
    plt.boxplot(metrics, showfliers=False)
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')

def distributionBoxPlot(metrics, title=None, output=None, text=None):
    fig = go.Figure()
    if title:
        fig.update_layout(
            title=title,
        )
    fig = go.Figure(data=[go.Box(y=metrics,
            boxpoints='all', # can also be outliers, or suspectedoutliers, or False
            jitter=0.3, # add some jitter for a better separation between points
            pointpos=-1.8, # relative position of points wrt box
            text=text
              )])
    if output:
        fig.write_html(output)

def distributionBoxPlotMultipleMatplot(metrics, title=None, output=None):
    plt.figure()
    if title:
        plt.title(title)
    datas = []
    names = []
    ticks = []
    count = 1
    for name, data in metrics.items():
        np_metric = np.array(data[1])
        datas.append(np_metric)
        names.append(name)
        ticks.append(count)
        count += 1
    plt.boxplot(datas, showfliers=False)
    plt.xticks(ticks, names)
    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')

def distributionBoxPlotMultiple(metrics, title=None, output=None):
    fig = go.Figure()
    if title:
        fig.update_layout(
            title=title,
        )
    for name, data in metrics.items():
        fig.add_trace(go.Box(y=data[1],
                name=name,
                text=data[0]
                ))

    fig.update_traces(boxpoints='all', pointpos=-1.8,jitter=0.3)
    if output:
        fig.write_html(output)

def scatterPlot(x,y,title=None,axisx=None,axisy=None,output=None,text=None):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=axisx,
        yaxis_title=axisy,
    )
    fig.add_trace(go.Scatter(x=x, y=y,
                    mode='markers',
                    text=text))
    
    if output:
        fig.write_html(output)

def scatterPlotMultiple(x,y,title=None,axisx=None,axisy=None,output=None,text=None,legend=None):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title=axisx,
        yaxis_title=axisy,
    )
    for i in range(len(x)):
        texti = None
        if text is not None:
            texti = text[i]
        legendi = None
        if legend is not None:
            legendi = legend[i]
        fig.add_trace(go.Scatter(x=x[i], y=y[i],
                        mode='markers',
                        text=texti,
                        name=legendi))
    
    if output:
        fig.write_html(output)

def histogram(x, output=None):
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=x))
    if output:
        fig.write_html(output)

def planck(lam, T):
        """ Returns the spectral radiance of a black body at temperature T.

        Returns the spectral radiance, B(lam, T), in W.sr-1.m-2 of a black body
        at temperature T (in K) at a wavelength lam (in nm), using Planck's law.

        """

        lam_m = lam / 1.e9
        fac = h*c/lam_m/k/T
        B = 2*h*c**2/lam_m**5 / (np.exp(fac) - 1)
        return B

def temperatureMap(x, output=None):
    cs = cs_hdtv

    lam = np.arange(380., 781., 5)

    temperature_map = np.zeros((x.shape[0], x.shape[1], 3), dtype=np.float32)
    for i in range(temperature_map.shape[0]):
        for j in range(temperature_map.shape[1]):
            temperature_map[i,j,:] = cs.spec_to_rgb(planck(lam, x[i,j,0]))
    
    if output:
        map_BGR = cv2.cvtColor(temperature_map, cv2.COLOR_RGB2BGR) 
        cv2.imwrite(output, map_BGR*255)

def relativeErrorMap(x, output=None):
    norm = colors.Normalize(vmin=0, vmax=relative_error_max, clip=True)

    fig = plt.figure()
    fcm = plt.pcolormesh(x[:,:,0], cmap='gray', norm=norm)
    fcm.axes.set_aspect(aspect='equal')
    plt.axis('off')

    fig.tight_layout()

    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight',pad_inches = 0)


def temperatureFalseColors(input, prediction, target, rel_err, output=None, title=None):
    cs = cs_hdtv

    lam = np.arange(380., 781., 5)

    temperature_map_pred = np.zeros((prediction.shape[0], prediction.shape[1], 3), dtype=np.float32)
    for i in range(temperature_map_pred.shape[0]):
        for j in range(temperature_map_pred.shape[1]):
            temperature_map_pred[i,j,:] = cs.spec_to_rgb(planck(lam, prediction[i,j,0]))

    temperature_map_target = np.zeros((target.shape[0], target.shape[1], 3), dtype=np.float32)
    for i in range(temperature_map_target.shape[0]):
        for j in range(temperature_map_target.shape[1]):
            temperature_map_target[i,j,:] = cs.spec_to_rgb(planck(lam, target[i,j,0]))

        
    fig = plt.figure(frameon=False)
    rows = 1
    columns = 4
    if title is not None:
        plt.title(title)


    #vmax = np.max([target_bw,prediction_bw])
    #vmin = np.min([target_bw,prediction_bw])
    #norm=colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    norm = colors.Normalize(vmin=0, vmax=relative_error_max, clip=True)

    fig.add_subplot(rows, columns, 1)
    plt.imshow((input*255).astype(np.uint8))
    plt.axis('off')
    plt.title("input")

    ax = fig.add_subplot(rows, columns, 2)
    plt.imshow(temperature_map_target)
    ax.set_aspect(aspect='equal')
    plt.axis('off')
    plt.title("real_temp")

    ax = fig.add_subplot(rows, columns, 3)
    plt.imshow(temperature_map_pred)
    ax.set_aspect(aspect='equal')
    plt.axis('off')
    plt.title("pred_temp")

    ax = fig.add_subplot(rows, columns, 4)
    fcm = plt.pcolormesh(rel_err[:,:,0], cmap='gray', norm=norm)
    ax.set_aspect(aspect='equal')
    plt.axis('off')
    plt.title("rel_err")
    plt.colorbar(fcm)

    fig.tight_layout()

    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight')

    return fig


def luminanceMap(img, output=None):
    fig = plt.figure()

    vmin = 2.385459780693054 
    vmax = 2424.477602539061

    luminance = (0.212671 * img[:,:,0] + 0.715160 * img[:,:,1] + 0.072169 * img[:,:,2])
    luminance[luminance == 0] = np.nan

    cmap = mpl.cm.get_cmap("binary_r").copy()
    cmap.set_bad('k',1.)

    plt.imshow(luminance, norm=mpl.colors.LogNorm(vmin=vmin, vmax=vmax), cmap=cmap)

    plt.axis('off')


    if output:
        plt.savefig(output, dpi=200, bbox_inches='tight',pad_inches = 0)
