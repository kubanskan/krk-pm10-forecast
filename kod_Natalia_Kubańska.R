setwd("C:/Users/kuban/OneDrive/Pulpit/Semestr7/DataMining/PM10Krk/data")
library(readxl)
library(ggplot2)
library(dplyr)
library(tidyverse)
library(lubridate) 
library(zoo)
library(tseries)
library(neuralnet)
library(Metrics)
library(keras)
library(caret)
# wybranie Kraków Bulwarowa i pominięcie pierwszegp pustego wiersza 
pm10_2013 <- read_excel("Krakow_pm10_2013.xls", sheet = "Pm10_2013", skip = 2)
pm10_2013 <- pm10_2013[-1,c(1,2)]

pm10_2014 <- read_excel("Krakow_pm10_2014.xls", sheet = "PM10_2014", skip = 2)
pm10_2014 <- pm10_2014[-1,c(1,2)]

data <- rbind(pm10_2013, pm10_2014)
colnames(data) <- c("Date", "PM10")


summary(data)
str(data)



data <- data %>%
  mutate(
    Date = dmy(Date),           
    PM10 = as.numeric(PM10)    
  )

summary(data)


ggplot(data, aes(x = Date, y = PM10)) +
  geom_line(color = 'darkblue') + 
  labs(title = "Stężenie PM10 w 2013 i 2014 roku przed imputacją brakujących wartości", x = "Czas", y = "PM10 (µg/m³)") +
  theme_minimal()



na_data <- data[!complete.cases(data), ]


# Uzupełnienie braków danych
rolling_mean <- rollapply(data$PM10, width = 14, FUN = mean, na.rm = TRUE, fill = NA, align = "right")

data$PM10_filled <- ifelse(is.na(data$PM10), rolling_mean, data$PM10)


ggplot(data) +
  geom_line(aes(x = Date, y = PM10_filled), color = 'red') +
  geom_line(aes(x = Date, y = PM10), color = 'darkblue') +
  labs(title = "Stężenie PM10 w 2013 i 2014 roku po imputacji brakujących wartości", 
       x = "Czas", 
       y = "PM10 (µg/m³)") +
  theme_minimal() +
  theme(legend.position = "bottom")

data$PM10 <- data$PM10_filled
data <- data[,c(1,2)]


data <- data %>%
  mutate(year = year(Date)) 

ggplot(data, aes(x = PM10)) +
  geom_histogram(bins = 30, fill = "lightblue", color = "black", alpha = 0.7) +
  facet_wrap(~ year, ncol = 2) + 
  labs(title = "Rozkład wartości PM10 w 2013 i 2014 roku",
       x = "PM10 (µg/m³)",
       y = "Częstość") +
  theme_minimal()




data <- data %>%
  mutate(Month = month(Date))

ggplot(data, aes(group = Month, y = PM10)) +
  geom_boxplot( fill = "lightblue", color = "black", alpha = 0.7) +
  labs(title = "Rozkład wartości PM10 w 2013 i 2014 roku według miesiąca",
       y = "PM10 (µg/m³)",
       x = "Częstość") +
  theme_minimal()



acf(data$PM10, lag.max = 30, main = "Autokorelacja stężenia PM10", 
    xlab = "Opóźnienie (lag)", ylab = "Autokorelacja")


acf(data$PM10, lag.max = 370, main = "Autokorelacja stężenia PM10", 
    xlab = "Opóźnienie (lag)", ylab = "Autokorelacja")

pacf(data$PM10, lag.max = 30, main = "Autokorelacja cząstkowa stężenia PM10", 
     xlab = "Opóźnienie (lag)", ylab = "Autokorelacja")

acf_result <- acf(data$PM10,lag.max = 30, plot = FALSE)
pacf_result <- pacf(data$PM10, ,lag.max = 30, plot = FALSE)




adf_test <- adf.test(data$PM10)
print(adf_test)







lagged_data <- data %>%
  mutate(lag1 = lag(PM10, 1), lag3 = lag(PM10, 3), lag10 = lag(PM10, 10)) %>%  
  drop_na() 


min_pm10 <- min(lagged_data$PM10)
max_pm10 <- max(lagged_data$PM10)

normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

lagged_data <- lagged_data %>%
  mutate(across(c(PM10, lag1, lag3, lag10, Month), normalize))


set.seed(123)
n <- nrow(lagged_data)
train_size <- floor(0.8 * n)

train_data <- lagged_data[1:train_size, ]
test_data <- lagged_data[(train_size + 1):n, ]




train_and_evaluate_nn <- function(train_data, test_data, target_var, predictors, hidden_layers = c(3), 
                                  min_pm10, max_pm10) {
  
  formula <- as.formula(paste(target_var, "~", paste(predictors, collapse = " + ")))
  nn_model <- neuralnet(formula, data = train_data, hidden = hidden_layers, linear.output = TRUE)
  

  
  predictions <- compute(nn_model, test_data[, predictors, drop = FALSE])$net.result
  predicted_pm10 <- predictions * (max_pm10 - min_pm10) + min_pm10
  

  true_pm10 <- test_data[[target_var]] * (max_pm10 - min_pm10) + min_pm10
  
  mae_value <- mape(true_pm10, predicted_pm10)
  
  return(mae_value)
}

#Model 1

target_var <- "PM10"
predictors <- c("lag1")

for (neurons in 1:30) {
  hidden_layers = c(neurons)
  mae_value <- train_and_evaluate_nn(train_data, test_data, target_var, predictors, hidden_layers = hidden_layers, 
                                      min_pm10 = min_pm10, max_pm10 = max_pm10)
  
  print(paste("MAPE,", neurons,":", mae_value))

}



for (layers in 2:4) {
  
  for (neurons in 1:20) {
    hidden_layers <- sample(1:neurons, layers, replace = TRUE) 
    mape_result <- train_and_evaluate_nn(train_data, test_data, target_var, predictors, hidden_layers, min_pm10 = min_pm10, max_pm10 = max_pm10)
    print(paste(
      "Liczba warstw:", layers, 
      "Liczba neuronów w każdej warstwie:", paste(hidden_layers, collapse = "-"), 
      "MAPE:", mape_result
    ))
 }
}



target_var <- "PM10"
predictors <- c("lag1", "lag3")

for (neurons in 1:30) {
  hidden_layers = c(neurons)
  mae_value <- train_and_evaluate_nn(train_data, test_data, target_var, predictors, hidden_layers = hidden_layers, 
                                     min_pm10 = min_pm10, max_pm10 = max_pm10)
  
  print(paste("MAPE,", neurons,":", mae_value))
  
}

for (layers in 2:4) {
  
  for (neurons in 1:20) {
    hidden_layers <- sample(1:neurons, layers, replace = TRUE) 
    mape_result <- train_and_evaluate_nn(train_data, test_data, target_var, predictors, hidden_layers, min_pm10 = min_pm10, max_pm10 = max_pm10)
    print(paste(
      "Liczba warstw:", layers, 
      "Liczba neuronów w każdej warstwie:", paste(hidden_layers, collapse = "-"), 
      "MAPE:", mape_result
    ))
  }
}





predictors <- c("lag1", "lag3", "lag10")

for (neurons in 1:30) {
  hidden_layers = c(neurons)
  mae_value <- train_and_evaluate_nn(train_data, test_data, target_var, predictors, hidden_layers = hidden_layers, 
                                     min_pm10 = min_pm10, max_pm10 = max_pm10)
  
  print(paste("MAPE,", neurons,":", mae_value))
  
}

for (layers in 2:4) {
  
  for (neurons in 1:20) {
    hidden_layers <- sample(1:neurons, layers, replace = TRUE) 
    mape_result <- train_and_evaluate_nn(train_data, test_data, target_var, predictors, hidden_layers, min_pm10 = min_pm10, max_pm10 = max_pm10)
    print(paste(
      "Liczba warstw:", layers, 
      "Liczba neuronów w każdej warstwie:", paste(hidden_layers, collapse = "-"), 
      "MAPE:", mape_result
    ))
  }
}

predictors <- c("lag1", "lag3", "month")


for (neurons in 1:30) {
  hidden_layers = c(neurons)
  mae_value <- train_and_evaluate_nn(train_data, test_data, target_var, predictors, hidden_layers = hidden_layers, 
                                     min_pm10 = min_pm10, max_pm10 = max_pm10)
  
  print(paste("MAPE,", neurons,":", mae_value))
  
}
for (layers in 2:4) {
  
  for (neurons in 1:20) {
    hidden_layers <- sample(1:neurons, layers, replace = TRUE) 
    mape_result <- train_and_evaluate_nn(train_data, test_data, target_var, predictors, hidden_layers, min_pm10 = min_pm10, max_pm10 = max_pm10)
    print(paste(
      "Liczba warstw:", layers, 
      "Liczba neuronów w każdej warstwie:", paste(hidden_layers, collapse = "-"), 
      "MAPE:", mape_result
    ))
  }
}

predictors <- c("lag1",  "Month")


for (neurons in 1:30) {
  hidden_layers = c(neurons)
  mae_value <- train_and_evaluate_nn(train_data, test_data, target_var, predictors, hidden_layers = hidden_layers, 
                                     min_pm10 = min_pm10, max_pm10 = max_pm10)
  
  print(paste("MAPE,", neurons,":", mae_value))
  
}

for (layers in 2:4) {
  
  for (neurons in 1:20) {
    hidden_layers <- sample(1:neurons, layers, replace = TRUE) 
    mape_result <- train_and_evaluate_nn(train_data, test_data, target_var, predictors, hidden_layers, min_pm10 = min_pm10, max_pm10 = max_pm10)
    print(paste(
      "Liczba warstw:", layers, 
      "Liczba neuronów w każdej warstwie:", paste(hidden_layers, collapse = "-"), 
      "MAPE:", mape_result
    ))
  }
}




nn_model <- neuralnet(
  PM10 ~ lag1 +Month,  
  data = train_data,
  hidden = c(4,1), 
  linear.output = TRUE
)

plot(nn_model)

predictions <- compute(nn_model, test_data[, c("lag1", "Month"), drop = FALSE])$net.result

test_data <- test_data %>%
  mutate(predicted_PM10 = predictions)


test_data2 <- test_data %>%
  mutate(predicted_PM10_original = predicted_PM10 * (max_pm10 - min_pm10) + min_pm10)
test_data2 <- test_data2 %>%
  mutate(PM10_original = PM10 * (max_pm10 - min_pm10) + min_pm10)



print(head(test_data2))

mae(test_data2$PM10_original ,test_data2$predicted_PM10_original )
 # "MAE: 14.6399581310587"
mape(test_data2$PM10_original ,test_data2$predicted_PM10_original )
# mape: 0.3234812


residuals <- test_data2$PM10_original- test_data2$predicted_PM10_original




ggplot(test_data2, aes(x = 1:nrow(test_data2))) +
  geom_line(aes(y = PM10_original, color = "Rzeczywiste"), size = 1) +  # Wartości rzeczywiste
  geom_line(aes(y = predicted_PM10_original, color = "Przewidywane"), size = 1) +  # Wartości przewidywane
  labs(title = "Porównanie rzeczywistych i przewidywanych wartości PM10 - zbiór testowy",
       x = "Indeks",
       y = "PM10") +
  scale_color_manual(values = c("Rzeczywiste" = "blue", "Przewidywane" = "red")) +  # Kolory dla każdej serii
  theme_minimal() +
  theme(legend.title = element_blank()) 


hist(residuals)


plot(test_data2$predicted_PM10_original, residuals,
     main = "Wykres punktowy błędów względem wartości przewidywanych",
     xlab = "Wartości przewidywane",
     ylab = "Residua (błędy)",
     pch = 19, col = "blue")

random_indices <- sample(1:nrow(test_data2), 5)
#  25 136  55  85  45
test_pred <- test_data2[random_indices,]
test_pred[c(1,9,10)]
mae(test_pred$predicted_PM10_original, test_pred$PM10_original)
mape(test_pred$predicted_PM10_original, test_pred$PM10_original)


# Model 2
window_size <- 10  
sequence_data <- function(data, window_size) {
  X <- array(dim = c(nrow(data) - window_size, window_size, 1)) 
  Y <- array(dim = c(nrow(data) - window_size, 1))            
  
  for (i in 1:(nrow(data) - window_size)) {
    X[i, , 1] <- data$PM10[i:(i + window_size - 1)]
    Y[i] <- data$PM10[i + window_size]
  }
  list(X = X, Y = Y)
}


data_sequences <- sequence_data(data, window_size)
X <- data_sequences$X
Y <- data_sequences$Y


train_size <- round(0.8 * dim(X)[1])
X_train <- X[1:train_size, , ]
Y_train <- Y[1:train_size]
X_test <- X[(train_size + 1):dim(X)[1], , ]
Y_test <- Y[(train_size + 1):dim(X)[1]]


model <- keras_model_sequential() %>%
  layer_lstm(units = 50, input_shape = c(window_size, 1)) %>%  
  layer_dense(units = 1) 


model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(learning_rate = 0.001)
)


history <- model %>% fit(
  x = X_train,
  y = Y_train,
  epochs = 50,
  batch_size = 16,
  validation_split = 0.2
)


model %>% evaluate(X_test, Y_test)

predictions <- model %>% predict(X_test)
mape(predictions, Y_test)

# > mape(predictions, Y_test)
# 0.3788018






model <- keras_model_sequential() %>%
  layer_lstm(units = 100, input_shape = c(window_size, 1)) %>% 
  layer_dense(units = 1)  

model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(learning_rate = 0.001)
)

history <- model %>% fit(
  x = X_train,
  y = Y_train,
  epochs = 50,
  batch_size = 16,
  validation_split = 0.2
)

model %>% evaluate(X_test, Y_test)
predictions <- model %>% predict(X_test)
mape(predictions, Y_test)

# mape(predictions, Y_test)
#0.3271907





model <- keras_model_sequential() %>%
  layer_lstm(units = 100, input_shape = c(window_size, 1), activation = "relu",) %>%  
  layer_dense(units = 1) 
model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(learning_rate = 0.001)
)

history <- model %>% fit(
  x = X_train,
  y = Y_train,
  epochs = 50,
  batch_size = 16,
  validation_split = 0.2
)

model %>% evaluate(X_test, Y_test)
predictions <- model %>% predict(X_test)
mape(predictions, Y_test)
#> mape(predictions, Y_test)
#  0.3173436

mae(predictions, Y_test)
# "MAE: 17.39894

wyn <- data.frame(date = test_data2$Date, predictions = predictions, actual = Y_test)


residuals <- Y_test - predictions

ggplot(data = wyn, aes(x = 1:nrow(test_data2))) +
  geom_line(aes(y = actual, color = "Rzeczywiste"), size = 1) +  
  geom_line(aes(y = predictions , color = "Przewidywane"), size = 1) + 
  labs(title = "Porównanie rzeczywistych i przewidywanych wartości PM10 - zbiór testowy",
       x = "Indeks",
       y = "PM10") +
  scale_color_manual(values = c("Rzeczywiste" = "blue", "Przewidywane" = "red")) +  # Kolory dla każdej serii
  theme_minimal() +
  theme(legend.title = element_blank()) 


hist(residuals)


plot(predictions, residuals,
     main = "Wykres punktowy błędów względem wartości przewidywanych",
     xlab = "Wartości przewidywane",
     ylab = "Residua (błędy)",
     pch = 19, col = "blue")


test_pred <- wyn[random_indices,]
test_pred
mae(test_pred$predictions, test_pred$actual)
mape(test_pred$predictions, test_pred$actual)


#> mae(test_pred$predictions, test_pred$actual)10.75248
#mape(test_pred$predictions, test_pred$actual) 0.2765949





model <- keras_model_sequential() %>%
  layer_lstm(units = 100,  activation = "relu",return_sequences = TRUE, input_shape = c(window_size, 1)) %>%
  layer_lstm(units = 50) %>%
  layer_dense(units = 1)

model %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_adam(learning_rate = 0.001)
)

history <- model %>% fit(
  x = X_train,
  y = Y_train,
  epochs = 50,
  batch_size = 16,
  validation_split = 0.2
)


model %>% evaluate(X_test, Y_test)
predictions <- model %>% predict(X_test)
mape(predictions, Y_test)

# mape(predictions, Y_test)
#[1] 0.4097577


