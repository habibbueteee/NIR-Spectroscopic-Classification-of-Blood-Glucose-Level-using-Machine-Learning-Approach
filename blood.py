from sys import stdout

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split


def prediction(X_calib, Y_calib, X_valid, Y_valid, plot_components=False):
    # Run PLS including a variable number of components, up to 40,  and calculate MSE
    mse = []
    component = np.arange(1,40)
    for i in component:
        pls = PLSRegression(n_components=i)
        # Fit
        pls.fit(X_calib, Y_calib)
        # Prediction
        Y_pred = pls.predict(X_valid)

        mse_p = mean_squared_error(Y_valid, Y_pred)
        mse.append(mse_p)

        comp = 100 * (i + 1) / 40
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the position of minimum in MSE
    msemin = np.argmin(mse)
    print("Suggested number of components: ", msemin + 1)
    stdout.write("\n")

    if plot_components is True:
        with plt.style.context(('ggplot')):
            plt.plot(component, np.array(mse), '-v', color='blue', mfc='blue')
            plt.plot(component[msemin], np.array(mse)[msemin], 'P', ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('MSE')
            plt.title('PLS')
            plt.xlim(xmin=-1)

        plt.show()

    # Run PLS with suggested number of components
    pls = PLSRegression(n_components=msemin + 1)
    pls.fit(X_calib, Y_calib)
    Y_pred = pls.predict(X_valid)

    # Calculate and print scores
    score_p = r2_score(Y_valid, Y_pred)
    mse_p = mean_squared_error(Y_valid, Y_pred)
    sep = np.std(Y_pred[:, 0] - Y_valid)
    rpd = np.std(Y_valid) / sep
    bias = np.mean(Y_pred[:, 0] - Y_valid)

    print('R2: %5.3f' % score_p)
    print('MSE: %5.3f' % mse_p)
    print('SEP: %5.3f' % sep)
    print('RPD: %5.3f' % rpd)
    print('Bias: %5.3f' % bias)

    # Plot regression and figures of merit
    rangey = max(Y_valid) - min(Y_valid)
    rangex = max(Y_pred) - min(Y_pred)

    z = np.polyfit(Y_valid, Y_pred, 1)
    with plt.style.context(('ggplot')):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(Y_pred, Y_valid, c='red', edgecolors='k')
        ax.plot(z[1] + z[0] * Y_valid, Y_valid, c='blue', linewidth=1)
        ax.plot(Y_valid, Y_valid, color='green', linewidth=1)
        plt.xlabel('Predicted')
        plt.ylabel('Measured')
        plt.title('Prediction')

        # Print the scores on the plot
        plt.text(min(Y_pred) + 0.05 * rangex, max(Y_valid) - 0.1 * rangey, 'R$^{2}=$ %5.3f' % score_p)
        plt.text(min(Y_pred) + 0.05 * rangex, max(Y_valid) - 0.15 * rangey, 'MSE: %5.3f' % mse_p)
        plt.text(min(Y_pred) + 0.05 * rangex, max(Y_valid) - 0.2 * rangey, 'SEP: %5.3f' % sep)
        plt.text(min(Y_pred) + 0.05 * rangex, max(Y_valid) - 0.25 * rangey, 'RPD: %5.3f' % rpd)
        plt.text(min(Y_pred) + 0.05 * rangex, max(Y_valid) - 0.3 * rangey, 'Bias: %5.3f' % bias)
        plt.show()


data = pd.read_csv('C:/Courses/bio project/Blood Glucose Dataset.csv')

X = data.drop('ref', axis=1)



Y = data['ref']
X_calib = X[:80]
X_valid = X[81:]
Y_calib=Y[:80]
Y_valid=Y[81:]



X_valid, X_calib, Y_valid, Y_calib = train_test_split(X, Y, test_size = 0.80)

prediction(X_calib, Y_calib, X_valid, Y_valid, plot_components=True)