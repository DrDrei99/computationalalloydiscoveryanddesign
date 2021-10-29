dataset <- read.csv("../ML/Results/Averages/KT-magnesium-MCCV-9010imageaverages.csv")
averages <- read.csv("../ML/Results/Averages/KT-magnesium-MCCV-9010averages.csv")

names = dataset$Image
predictions = dataset$Average.Prediction
expected_values = dataset$Expected

estimates = lm(predictions~expected_values)$coefficients

plot(expected_values, predictions, main = 'Average Predicted Values vs. Expected values for Magnesium Dataset',
     xlab = 'Expected Tensile Yield Strength (MPa)', ylab = 'Predicted Tensile Yield Strength (MPa)', pch=20, cex=0.5
    ,xlim = c(107,128))

points(averages$Expected, averages$Average.Prediction, col = "red", pch=20, cex=2)


abline(lm(predictions~expected_values), col = 'red')
abline(0, 1, col='black', lty=2)

legend("topleft", legend=c("Base Image Average", "Alloy Average","45 Degree Line", "Fitted Line"), 
       col = c("black", "red","black", "red"), pch=c(20,20,NA, NA), lty=c(NA,NA,2,1))

text(averages$Expected+0.7, averages$Average.Prediction+1.5,averages[,"Alloy.Name"])
text(122,100, capture.output(cat("Fitted Line Slope = ", estimates[2], ", Intercept = ", estimates[1])))
error = predictions - expected_values

average_percent_error_mag = (mean(abs(error)/predictions))*100