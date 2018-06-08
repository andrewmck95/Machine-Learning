# Andrew McKeown

# importing owls15 dataset
library(readr)
owls15 <- read_csv("~/file location")
# randomising data in owls15
set.seed(9850)
g <- runif(nrow(owls15))
owls15 <- owls15[order(g),]
# making owls15 a dataframe
owls15 <- data.frame(owls15)
# setting owl type as factors
owls15$type <- factor(owls15$type, levels = c('LongEaredOwl', 'SnowyOwl', 'BarnOwl'), label = c('LongEaredOwl', 'SnowyOwl', 'BarnOwl'))
# summarise and view owls15 data
summary(owls15)
head(owls15)

#create a matrix for owls15 to be used later
matrix_owls15 <- data.matrix(owls15)
# splitting owls15 data into test and train
split <- c(sample(1:135,90))
# m1 = model
model_prediction <- function(m1, data = matrix_owls15) {
  # new data, transfer to matrix
  matrix_owls15 <- data.matrix(data)
  # sweep used for systematically manipulating a large matrix either column by column, or row by row
  # %*% is used for matrix multiplication
  # feed propagation
  hidden_layers <- sweep(matrix_owls15 %*% m1$weight_1 ,2, m1$bias_1, '+')
  # neurons
  # rectified linear unit activation
  # pmax returns the parallel maxima and minima of the input values.
  hidden_layers <- pmax(hidden_layers, 0)
  outcome <- sweep(hidden_layers %*% m1$weight_2, 2, m1$bias_2, '+')
  # using normalized exponential loss function
  probabilities <-sweep(exp(outcome), 1, rowSums(exp(outcome), na.rm = FALSE), '/') 
  # maximum probability
  classes <- max.col(probabilities)
  return(classes)
}
# train neural network with two layers:
# initialize neurons and the hidden layers
# maximum number of steps
# delta loss
# rate of learning
# rate of reularization
# show results for every step
model_train <- function(x, y, training=data, testing, m1 = NULL, hidden=c(4), max_iterations, delta_rule, learn_rate, regularization, results)
{
  # number of training sets
  train_total <- nrow(training)
  # unname removes names or dimnames attribute of object
  unnamed <- unname(data.matrix(training[,x]))
  # fixing integer categories
  categories <- training[,y]
  if(is.factor(categories)==TRUE) {categories <- as.integer(categories)}
  # indexing rows and columns
  # cbind combines data by columns or rows
  categories.index <- cbind(1:train_total, match(categories, sort(unique(categories))))
  
  # create model
  if(is.null(m1)==FALSE) {
    # categories, input, weights and biases are initialized to the model
    total_cat  <- m1$total_cat
    input  <- m1$input
    hidden  <- m1$hidden
    weight_1 <- m1$weight_1
    weight_2 <- m1$weight_2
    bias_1 <- m1$bias_1
    bias_2 <- m1$bias_2
    
  } else {
    # total categories to be classified
    total_cat <- length(unique(categories))
    # number of input features
    input <- ncol(unnamed)
    # initialize weights (0.008 by matrix by activation function)
    # rnorm generates a random value from the normal distribution
    # after testing with weight 2 as pnorm got a higher m1_accuracy
    weight_1 <- 0.008*matrix(rnorm(input*hidden), nrow=input, ncol=hidden)
    weight_2 <- 0.008*matrix(pnorm(hidden*total_cat), nrow=hidden, ncol=total_cat)
    # initialize biases (to 0)
    bias_1 <- matrix(0, nrow=1, ncol=hidden)
    bias_2 <- matrix(0, nrow=1, ncol=total_cat)
  }
  
  # data_lost calculates compatibility of prediction
  # initalize data_lost
  data_lost <- 1
  # train neural network
  index <- 0
  while(data_lost >= delta_rule && index <= max_iterations) {
    # iterate
    index <- index +1
    
    #-----------------
    # FEED PROPAGATION
    #-----------------
    
    hidden_layers <- sweep(unnamed %*% weight_1 ,2, bias_1, '+')
    # rectified linear unit activation function f(x)=max(x,0), where x is the input to the neuron
    hidden_layers <- pmax(hidden_layers, 0)
    outcome <- sweep(hidden_layers %*% weight_2, 2, bias_2, '+')
    # normalized exponential function used again
    # rowSums is used to sum values of raster objects by row
    probabilities <- exp(outcome)/rowSums(exp(outcome), na.rm = FALSE)
    # finding value for data lost
    # log function applied
    prob_logarithm <- -log(probabilities[categories.index])
    # calculating regularization data lost
    regularization_lost <- (sum(weight_1^2) + sum(weight_2^2))*regularization
    # calculating training data lost
    training_lost  <- sum(prob_logarithm)/train_total
    # getting the total data lost
    data_lost <- regularization_lost+training_lost
    
    #-----------------
    # BACK PROPAGATION
    #-----------------
    
    # minimizing the data lost by modifying weights and biases
    bp_outcomes <- probabilities
    bp_outcomes[categories.index] <- bp_outcomes[categories.index] -1
    bp_outcomes <- bp_outcomes / train_total
    # transposing matrix using t
    bp_weight_2 <- t(hidden_layers) %*% bp_outcomes 
    bp_bias_2 <- colSums(bp_outcomes)
    bp_hidden <- bp_outcomes %*% t(weight_2)
    # hidden layers less or equal to 0 are set to 0
    bp_hidden[hidden_layers <= 0] <- 0
    bp_weight_1 <- t(unnamed) %*% bp_hidden
    bp_bias_1 <- colSums(bp_hidden)
    bp_weight_1 <- bp_weight_1  + regularization*weight_1
    bp_weight_2 <- bp_weight_2 + regularization*weight_2
    # modifying initial weights and biases
    # weight=weight-learn rate*back propagation weight
    weight_1 <- weight_1 - learn_rate * bp_weight_1
    weight_2 <- weight_2 - learn_rate * bp_weight_2
    # bias=bias-learn rate*back propagation bias
    bias_1 <- bias_1 - learn_rate * bp_bias_1
    bias_2 <- bias_2 - learn_rate * bp_bias_2
    
  }
  # final results stored in list
  m1 <- list( total_cat, input, weight_1= weight_1, weight_2= weight_2, bias_1= bias_1, bias_2= bias_2)
  return(m1)
}

#----------------------------------------
# TESTING THE MODEL USING OWLS15 DATASET
#----------------------------------------

# training the data, the number of iterations needs to be a large value
owls15_m1 <- model_train(x=1:4, y=5, training=owls15[split,], testing=owls15[-split,], hidden=4, max_iterations=1000, delta_rule=1e-2, learn_rate = 1e-2,regularization = 1e-3,  results=45)
# make prediction
prediction <- model_prediction(owls15_m1, owls15[-split, -5])
# accuracy of the testing set
accuracy <- mean(as.numeric(owls15[-split, 5]) == prediction)
# check the accuracy
accuracy
# confusion matrix
table(owls15[-split,5], prediction)
