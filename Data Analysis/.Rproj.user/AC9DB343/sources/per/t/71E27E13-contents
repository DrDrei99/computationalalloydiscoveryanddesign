dataset <- read.csv("../ML/Results/Averages/KT-whole-MCCV-9010-200runimageaverages.csv")

dataset_mag = subset(dataset, dataset$Source == "Magnesium Dataset")
dataset_asm = subset(dataset, dataset$Source == "ASM Dataset")
dataset_titan = subset(dataset, dataset$Source == "Ti-6Al-2Sn-4Zr-2Mo-0.1Si Dataset")

source = dataset$Source
predictions = dataset$Average.Prediction
expected_values = dataset$Expected

predictions_asm = dataset_asm$Average.Prediction
expected_values_asm = dataset_asm$Expected

predictions_mag = dataset_mag$Average.Prediction
expected_values_mag = dataset_mag$Expected

predictions_titan = dataset_titan$Average.Prediction
expected_values_titan = dataset_titan$Expected

estimates = lm(predictions~expected_values)$coefficients

plot(expected_values_asm, predictions_asm, main = 'Average Predicted Values vs. Expected values for Whole Dataset',
     xlab = 'Expected Tensile Yield Strength (MPa)', ylab = 'Predicted Tensile Yield Strength (MPa)',
     col = "gold4", pch=20)

points(expected_values_mag,predictions_mag, col="blue", pch = 20)

points(expected_values_titan, predictions_titan, col="violetred", pch = 20)

abline(lm(predictions~expected_values), col = 'red')
abline(0, 1, col='black', lty = 2)

legend("topleft", legend=c("45 Degree Line", "Fitted Line","ASM Micrograph Database \U2122","Titanium Dataset","Magnesium Dataset"),
       col = c("black", "red", "gold4", "violetred", "blue"),pch = c(NA,NA,20,20,20), lty=c(2,1,NA,NA,NA))

text(1450,100, capture.output(cat("Fitted Line Slope = ", estimates[2], ", Intercept = ", estimates[1])))

error = expected_values-predictions
percent_error = (mean(abs(error)/predictions))*100
#hist(percent_error)
#plot(expected_values, error, main = 'Percent Error per Base Image',
#     xlab = 'Expected Tensile Yield Strength (MPa)', ylab = 'Percent Error (%)', pch=20)
