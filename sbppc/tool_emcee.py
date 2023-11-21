import numpy as np


def trace_plot(axes, samples, labels=None, alpha=0.3, **kwargs):
    axes = np.atleast_1d(axes)
    if samples.ndim == 2:  # get_chain(flat=True) case
        nsamples, npars = samples.shape
    elif samples.ndim == 3:  # get_chain() case
        nsamples, nchain, npars = samples.shape
    else:
        raise ValueError(f"{samples.ndim = } is not 2 or 3")

    if axes.size != npars:
        raise ValueError(f"{axes.size = } is different from {samples.shape[-1] = }")

    if labels is not None and len(labels) != npars:
        raise ValueError(f"{len(labels) = } is different from {samples.shape[-1] = }")

    for i, ax in enumerate(axes):
        ax.plot(samples[:, i], alpha=alpha, **kwargs)
        if labels is not None:
            ax.set_ylabel(labels[i])
        ax.set_xlim(0, nsamples)
