
# we will not train many models in the classroom on this example
# as they will take a long time
inClass = TRUE

# code to load digits and show one digit
# assumption here is that we are in "scripts/" directory
source("mnist.helper.R")


MNIST_DIR = "MNISTDigits"
# load all digits
digit.data = load_mnist(MNIST_DIR)

library(h2o)
# start or connect to h2o server
h2oServer <- h2o.init(ip="localhost", port=54321, max_mem_size="4g", nthreads=2)

# we need to load data into h2o format
train_hex = as.h2o(data.frame(x=digit.data$train$x, y=digit.data$train$y))
test_hex = as.h2o(data.frame(x=digit.data$test$x, y=digit.data$test$y))

predictors = 1:784
response = 785

train_hex[,response] <- as.factor(train_hex[,response])
test_hex[,response] <- as.factor(test_hex[,response])

train_labels = train_hex[,response]
test_labels = test_hex[,response]

# create frames with input features only
# we will need these later for unsupervised training
trainX = train_hex[,-response]
testX = test_hex[,-response]


#########################
## 
## Autoencoder
##
#########################

nfeatures = 50 

# train autoencoder on train_unsupervised
auto_encoder = h2o.deeplearning(x=predictors,
                                training_frame=trainX,
                                activation="Tanh",
                                autoencoder=T,
                                hidden=c(nfeatures),
                                l1=1e-5,
                                ignore_const_cols=F,
                                epochs=1)

train.ae.features = h2o.deepfeatures(auto_encoder, trainX, 1)

test.ae.features = h2o.deepfeatures(auto_encoder, testX, 1)

# we have a 50 dimensional representation of our data
# let's train a random forest
if (inClass == FALSE) {
    deep.ae.rf = h2o.randomForest(x=1:50, y=51, 
                                  training_frame = h2o.cbind(train.ae.features, train_labels),
                                  ntrees = 500,
                                  min_rows = 100,
                                  model_id = "DRF_features.50"
    )
    h2o.saveModel(deep.ae.rf, path="mnist")
} else {
    deep.ae.rf = h2o.loadModel( path = file.path("mnist", "DRF_features.50") )
}

phat = h2o.predict(deep.ae.rf, test.ae.features)
head(phat)

h2o.confusionMatrix(deep.ae.rf, h2o.cbind(test.ae.features, test_labels))

########### we can also use the representation to find outliers

# 2) DETECT OUTLIERS
# h2o.anomaly computes the per-row reconstruction error for the test data set
# (passing it through the autoencoder model and computing mean square error (MSE) for each row)
test_rec_error = as.data.frame(h2o.anomaly(auto_encoder, testX)) 

# 3) VISUALIZE OUTLIERS
# Let's look at the test set points with low/median/high reconstruction errors.
# We will now visualize the original test set points and their reconstructions obtained 
# by propagating them through the narrow neural net.

# Convert the test data into its autoencoded representation (pass through narrow neural net)
test_recon = predict(auto_encoder, testX)

# The good
# Let's plot the 25 digits with lowest reconstruction error.
# First we plot the reconstruction, then the original scanned images.  
plotDigits(test_recon, test_rec_error, c(1:25))
plotDigits(testX,   test_rec_error, c(1:25))

# The bad
# Now the same for the 25 digits with median reconstruction error.
plotDigits(test_recon, test_rec_error, c(4988:5012))
plotDigits(testX,   test_rec_error, c(4988:5012))

# The ugly
# And here are the biggest outliers - The 25 digits with highest reconstruction error!
plotDigits(test_recon, test_rec_error, c(9976:10000))
plotDigits(testX,   test_rec_error, c(9976:10000))


######################
#
# Stacked autoencoder
#
######################

# this function builds a vector of autoencoder models, one per layer
get_stacked_ae_array <- function(training_data,layers,args){  
    vector <- c()
    index = 0
    for(i in 1:length(layers)){    
        index = index + 1
        ae_model <- do.call(h2o.deeplearning, 
                            modifyList(list(x=names(training_data),
                                            training_frame=training_data,
                                            autoencoder=T,
                                            hidden=layers[i]),
                                       args))
        training_data = h2o.deepfeatures(ae_model,training_data,layer=1)
        
        names(training_data) <- gsub("DF", paste0("L",index,sep=""), names(training_data)) 
        vector <- c(vector, ae_model)    
    }
    vector
}

# this function returns final encoded contents
apply_stacked_ae_array <- function(data,ae){
    index = 0
    for(i in 1:length(ae)){
        index = index + 1
        data = h2o.deepfeatures(ae[[i]],data,layer=1)
        names(data) <- gsub("DF", paste0("L",index,sep=""), names(data)) 
    }
    data
}

## Build reference model on full dataset and evaluate it on the test set
## This is a bad model
model_ref = h2o.deeplearning(training_frame=train_hex, 
                             x=predictors, y=resp, 
                             hidden=c(10), 
                             epochs=1)
phat_ref = h2o.performance(model_ref, test_hex)
h2o.logloss(phat_ref)
h2o.confusionMatrix(phat_ref)

## Now build a stacked autoencoder model with three stacked layer AE models
## First AE model will compress the 717 non-const predictors into 200
## Second AE model will compress 200 into 100
## Third AE model will compress 100 into 50
layers <- c(200,100,50)
args <- list(activation="Tanh", epochs=1, l1=1e-5)
stacked_ae <- get_stacked_ae_array(trainX, layers, args)

## Now compress the training/testing data with this 3-stage set of AE models
train_compressed <- apply_stacked_ae_array(trainX, stacked_ae) 
test_compressed <- apply_stacked_ae_array(testX, stacked_ae)

model_on_compressed_data <- h2o.deeplearning(training_frame=h2o.cbind(train_compressed, train_hex[,resp]), 
                                             x=1:50, y=51, 
                                             hidden=c(10), 
                                             epochs=1
)

phat = h2o.performance(model_on_compressed_data, h2o.cbind(test_compressed, test_hex[,resp]))
h2o.logloss(phat)
h2o.confusionMatrix(phat)


#########################
## 
## Autoencoder -- visualization
##
#########################

if (inClass == FALSE) {
    pca.ae = h2o.deeplearning(x=predictors,
                              training_frame=trainX,
                              activation="Tanh",
                              autoencoder=T,
                              hidden=c(1024, 512, 256, 128, 2),
                              l1=1e-5,
                              ignore_const_cols=F,
                              epochs=10,
                              model_id="pca.ae"
    )
    h2o.saveModel(pca.ae, path="mnist")
} else {
    pca.ae = h2o.loadModel( path = file.path("mnist", "pca.ae") )
}

pca.ae.train.features = h2o.deepfeatures(pca.ae, trainX, 5)
dim(pca.ae.train.features)

plot(as.matrix(pca.ae.train.features[1:1000,]), 
     pch=as.character( digit.data$train$y[1:1000] ), 
     col=c(1:10)[digit.data$train$y[1:1000]+1])