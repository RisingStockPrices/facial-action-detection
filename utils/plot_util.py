def plot_aus(aus,pth='aus_tmp.png'):
    res = aus.plot()
    res.figure.savefig(pth)