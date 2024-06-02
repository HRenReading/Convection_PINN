import tensorflow as tf
import matplotlib.pyplot as plt


def extract_var(data):
    x = data[:, 0:1]
    t = data[:, 1:2]
    return x, t


def jacobian(tape, pred, x, t):
    ux = tape.gradient(pred, x)
    ut = tape.gradient(pred, t)
    return ux, ut


def prediction(model, input_test):
    steps = input_test.shape[0]
    x, t = extract_var(input_test)
    for i in range(steps):
        if i == 0:
            pred = model(tf.concat([x, tf.ones_like(x) * t[i, :]], axis=1))
        else:
            p = model(tf.concat([x, tf.ones_like(x) * t[i, :]], axis=1))
            pred = tf.concat([pred, p], axis=1)
        del i
    return pred


def plot(pred, beta, n):
    ax = plt.gca()
    im = ax.imshow(pred, cmap='hsv', aspect=0.5)
    plt.colorbar(im, shrink=0.6)
    plt.xlabel('Time: t')
    plt.ylabel('Domain: x')
    plt.ylim(0, n)
    plt.title('Prediction with $\\beta$='+str(beta))
    plt.show()