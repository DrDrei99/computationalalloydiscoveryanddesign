dataset <- read.csv("../ML/Results/guessed-magnesium-MCCV-9010.csv")

names = dataset$Image
predictions = dataset$Predicted
expected_values = dataset$Expected

model = lm(predictions~expected_values)

residuals = predictions - expected_values

estimates = lm(predictions~expected_values)$coefficients

plot(expected_values, predictions, main = 'Predicted vs. Expected values for Magnesium Dataset with guessed Neural Net Architecture',
     xlab = 'Expected Tesnile Yield Strength (MPa)', ylab = 'Predicted Tensile Yield Strength (MPa)',
     )

abline(model, col = 'red')
abline(0, 1, col='blue')

