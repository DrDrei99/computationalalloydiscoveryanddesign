dataset <- read.csv("../ML/Results/guessed-whole-MCCV-9010.csv")

source = dataset$Source
names = dataset$Image
predictions = dataset$Predicted
expected_values = dataset$Expected

counter = 1
asms = c()
mags = c()
titans = c()

for (x in predictions){
  if (source[counter] == "ASM Dataset"){
    asms = append(asms, x)
  }
  if (source[counter] == "Ti-6Al-2Sn-4Zr-2Mo-0.1Si Dataset"){
    titans = append(titans, x)
  }
  if (source[counter] == "Magnesium Dataset"){
    mags = append(mags, x)
  }
  counter = counter + 1
}

estimates = lm(predictions~expected_values)$coefficients

plot(expected_values, predictions, main = 'Predicted vs. Expected values for Whole Dataset with guessed Neural Net Architecture',
     xlab = 'Expected Tesnile Yield Strength (MPa)', ylab = 'Predicted Tensile Yield Strength (MPa)', pch=20, cex=0.5
     )

abline(lm(predictions~expected_values), col = 'red')
abline(0, 1, col='blue')
