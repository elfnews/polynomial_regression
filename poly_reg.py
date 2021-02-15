# TODO: Add import statements

# Assign the data to predictor and outcome variables
# TODO: Load the data
from numpy import reshape, loadtxt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

train_data = loadtxt('poly_reg_data.csv', delimiter=',', skiprows=1)
X = reshape(train_data[:, :-1], (20, 1))
y = train_data[:, -1]

# Create polynomial features
# TODO: Create a PolynomialFeatures object, then fit and transform the
# predictor feature
poly_feat = PolynomialFeatures(degree=1)
X_poly = poly_feat.fit_transform(X)
print(X_poly)

# Make and fit the polynomial regression model
# TODO: Create a LinearRegression object and fit it to the polynomial predictor
# features
poly_model = LinearRegression(fit_intercept=False).fit(X_poly, y)
print(poly_model)

# Once you've completed all of the steps, select Test Run to see your model
# predictions ag