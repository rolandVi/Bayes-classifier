import numpy as np
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import norm

iris = datasets.load_iris()

# Create a scatter plot of the first two features, and use their labels as colour values.
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target, alpha=0.7)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.show()
# Create a scatter plot of the third and fourth feature.
plt.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target, alpha=0.7)
plt.xlabel(iris.feature_names[2])
plt.ylabel(iris.feature_names[3])
plt.show()

# Separate the flower sets
setosaArr = [i == 0 for i in iris.target]
setosa_flowers = iris.data[setosaArr]

versiArr = [i == 0 for i in iris.target]
versicolor_flowers = iris.data[versiArr]

virgiArr = [i == 0 for i in iris.target]
virginica_flowers = iris.data[virgiArr]

# Split train and test set
X_train, X_test, Y_train, Y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=20)

# Separate the training dataset into the three flower types.
setoFilter = [i == 0 for i in Y_train]
setosa_X_train = X_train[setoFilter]

versFilter = [i == 1 for i in Y_train]
versicolor_X_train = X_train[versFilter]

virgFilter = [i == 2 for i in Y_train]
virginica_X_train = X_train[virgFilter]

# We use the third feature
feature_idx = 2

# Check if the flowers are uniform
fig, axs = plt.subplots(3, 1, figsize=(6, 12), sharex=True)

axs[0].hist(setosa_flowers[:, feature_idx], label=iris.target_names[0], alpha=0.7, color="red")
axs[0].set_title(iris.target_names[0])

axs[1].hist(versicolor_flowers[:, feature_idx], label=iris.target_names[1], alpha=0.7, color="green")
axs[1].set_title(iris.target_names[1])

axs[2].hist(virginica_flowers[:, feature_idx], label=iris.target_names[2], alpha=0.7, color="blue")
axs[2].set_title(iris.target_names[2])

for ax in axs:
    ax.set_ylabel('Number of flowers')
    ax.legend()

axs[-1].set_xlabel(iris.feature_names[feature_idx])
plt.tight_layout()
plt.show()

# Compute the mean for each flower type.
mean_setosa = np.mean(setosa_X_train[:, feature_idx])
mean_versicolor = np.mean(versicolor_X_train[:, feature_idx])
mean_virginica = np.mean(virginica_X_train[:, feature_idx])

# Compute the standard deviation for each flower type.
sd_setosa = np.std(setosa_X_train[:, feature_idx])
sd_versicolor = np.std(versicolor_X_train[:, feature_idx])
sd_virginica = np.std(virginica_X_train[:, feature_idx])

# Histograms of the flower types of the training set
plt.hist(setosa_X_train[:, feature_idx], label=iris.target_names[0], alpha=0.7)
plt.hist(versicolor_X_train[:, feature_idx], label=iris.target_names[1], alpha=0.7)
plt.hist(virginica_X_train[:, feature_idx], label=iris.target_names[2], alpha=0.7)

# Plot your PDFs here
xs = np.linspace(0, 7, 100)

plt.plot(xs, norm.pdf(xs, mean_setosa, sd_setosa))
plt.plot(xs, norm.pdf(xs, mean_versicolor, sd_versicolor))
plt.plot(xs, norm.pdf(xs, mean_virginica, sd_virginica))

plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel('Number of flowers / PDF')
plt.legend()
plt.show()


def posterior(x, means, sds, priors, i):
    """
    Compute the posterior probability P(C_i | x).
    """

    top = norm.pdf(x, means[i], sds[i]) * priors[i]
    bottom = sum([norm.pdf(x, means[i], sds[i]) * priors[i] for i in range(3)])
    return top / bottom


means = [mean_setosa, mean_versicolor, mean_virginica]
sds = [sd_setosa, sd_versicolor, sd_virginica]
priors = [
    setosa_X_train.shape[0] / X_train.shape[0],
    versicolor_X_train.shape[0] / X_train.shape[0],
    virginica_X_train.shape[0] / X_train.shape[0]
]

xs = np.linspace(0, 7, 100)

post_setosa_vals = [posterior(x, means, sds, priors, 0) for x in xs]
post_versicolor_vals = [posterior(x, means, sds, priors, 1) for x in xs]
post_virginica_vals = [posterior(x, means, sds, priors, 2) for x in xs]

plt.plot(xs, post_setosa_vals, label=iris.target_names[0])
plt.plot(xs, post_versicolor_vals, label=iris.target_names[1])
plt.plot(xs, post_virginica_vals, label=iris.target_names[2])

plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel('Posterior probability')
plt.legend()
plt.show()


def classify(x, means, sds, priors):
    a = posterior(x, means, sds, priors, 0)
    b = posterior(x, means, sds, priors, 1)
    c = posterior(x, means, sds, priors, 2)
    if a >= b and a >= c:
        return 0
    elif b >= a and b >= c:
        return 1
    elif c >= a and c >= b:
        return 2


def evaluate(X_test, Y_test, means, sds, priors):
    good = 0
    for i in range(len(X_test)):
        if classify(X_test[i], means, sds, priors) == Y_test[i]:
            good += 1
    return good / len(X_test)


accuracy = evaluate(X_test[:, feature_idx], Y_test, means, sds, priors)

print("Accuracy: ", accuracy)


def decision_boundary(means, sds, priors):
    decision_boundaries = []
    xs = np.linspace(1, 7, 1000)

    l, r = 0, 500
    while l <= r:
        mid = (l + r) // 2
        if (posterior(xs[mid], means, sds, priors, 0) > posterior(xs[mid], means, sds, priors, 1)):
            l = mid + 1
        else:
            r = mid - 1
    decision_boundaries.append(xs[l])

    l, r = 500, 999
    while l <= r:
        mid = (l + r) // 2
        if (posterior(xs[mid], means, sds, priors, 1) > posterior(xs[mid], means, sds, priors, 2)):
            l = mid + 1
        else:
            r = mid - 1
    decision_boundaries.append(xs[l])

    return decision_boundaries


# Create a scatter plot of the third and fourth feature.
feature_idx2 = 3

plt.scatter(iris.data[:, feature_idx], iris.data[:, feature_idx2], c=iris.target, alpha=0.7)
plt.xlabel(iris.feature_names[feature_idx])
plt.ylabel(iris.feature_names[feature_idx2])
decision_boundaries = decision_boundary(means, sds, priors)
for boundary in decision_boundaries:
    plt.axvline(x=boundary)

plt.show()
