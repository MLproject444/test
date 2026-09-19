# Install missing dependencies
if (!require("rpart")) install.packages("rpart")
if (!require("randomForest")) install.packages("randomForest")

library(rpart)
library(randomForest)

set.seed(42)
n_samples <- 1000

# 1. Generate Synthetic Health Dataset matching limits
age         <- sample(18:90, n_samples, replace = TRUE)
bmi         <- round(runif(n_samples, 18.5, 48.0), 1)
blood_press <- sample(85:210, n_samples, replace = TRUE)
cholesterol <- sample(120:380, n_samples, replace = TRUE)
glucose     <- sample(65:300, n_samples, replace = TRUE)

# Risk Probability Logit Formula
logit <- -8.5 + (0.04 * age) + (0.08 * bmi) + (0.02 * blood_press) + 
         (0.01 * cholesterol) + (0.025 * glucose)
prob  <- 1 / (1 + exp(-logit))
disease_risk <- ifelse(prob > 0.45, 1, 0)

health_data <- data.frame(
  age = age,
  bmi = bmi,
  blood_pressure = blood_press,
  cholesterol = cholesterol,
  glucose = glucose,
  risk = as.factor(disease_risk)
)

# 2. Train Models
log_model  <- glm(risk ~ ., data = health_data, family = binomial)
tree_model <- rpart(risk ~ ., data = health_data, method = "class", 
                    control = rpart.control(maxdepth = 4))
rf_model   <- randomForest(risk ~ ., data = health_data, ntree = 100, mtry = 2, importance = TRUE)

# 3. Export Models
saveRDS(log_model,  file = "log_model.rds")
saveRDS(tree_model, file = "tree_model.rds")
saveRDS(rf_model,   file = "rf_model.rds")

cat("Models successfully trained and exported to log_model.rds, tree_model.rds, and rf_model.rds\n")