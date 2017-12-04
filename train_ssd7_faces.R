library(ssdkeras)
library(keras)
library(stringr)

K <- backend()

# 1: Set some necessary parameters

img_height = 250 # Height of the input images
img_width = 250 # Width of the input images
img_channels = 3 # Number of color channels of the input images
n_classes = 4L # Number of classes including the background class
min_scale = 0.08 # The scaling factor for the smallest anchor boxes
max_scale = 0.96 # The scaling factor for the largest anchor boxes
scales = c(0.08, 0.16, 0.32, 0.64, 0.96) # An explicit list of anchor box scaling factors. If this is passed, it will override `min_scale` and `max_scale`.
aspect_ratios = c(0.5, 1.0, 2.0) # The list of aspect ratios for the anchor boxes
two_boxes_for_ar1 = TRUE # Whether or not you want to generate two anchor boxes for aspect ratio 1
limit_boxes = FALSE # Whether or not you want to limit the anchor boxes to lie entirely within the image boundaries
variances = c(1.0, 1.0, 1.0, 1.0) # The list of variances by which the encoded target coordinates are scaled
coords = 'centroids' # Whether the box coordinates to be used should be in the 'centroids' or 'minmax' format, see documentation
normalize_coords = FALSE # Whether or not the model is supposed to use relative coordinates that are within [0,1]

# 2: Build the Keras model (and possibly load some trained weights)

K$clear_session() # Clear previous models from memory.
# The output `predictor_sizes` is needed below to set up `SSDBoxEncoder`
modelOut = build_model(image_size = c(img_height, img_width, img_channels),
                                     n_classes=n_classes,
                                     min_scale=min_scale,
                                     max_scale=max_scale,
                                     scales=scales,
                                     aspect_ratios_global=aspect_ratios,
                                     aspect_ratios_per_layer=NULL,
                                     two_boxes_for_ar1=two_boxes_for_ar1,
                                     limit_boxes=limit_boxes,
                                     variances=variances,
                                     coords=coords,
                                     normalize_coords=normalize_coords)
model <- modelOut$model
predictor_sizes <- modelOut$predictor_sizes

# model$load_weights("checkpoints.h5")

### Set up training

batch_size = 32L

# 3: Instantiate an Adam optimizer and the SSD loss function and compile the model

adam = optimizer_adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=5e-05)

ssd_loss = SSDLoss$new(neg_pos_ratio=3L, n_neg_min=0L, alpha=1.0, n_classes = n_classes)

model$compile(optimizer=adam, loss=ssd_loss$compute_loss)

# 4: Instantiate an encoder that can encode ground truth labels into the format needed by the SSD loss function

ssd_box_encoder = SSDBoxEncoder$new(img_height=img_height,
                                img_width=img_width,
                                n_classes=n_classes,
                                predictor_sizes=predictor_sizes,
                                min_scale=min_scale,
                                max_scale=max_scale,
                                scales=scales,
                                aspect_ratios_global=aspect_ratios,
                                aspect_ratios_per_layer= NULL,
                                two_boxes_for_ar1=two_boxes_for_ar1,
                                limit_boxes=limit_boxes,
                                variances=variances,
                                pos_iou_threshold=0.5,
                                neg_iou_threshold=0.2,
                                coords=coords,
                                normalize_coords=normalize_coords)

# 5: Create the training set batch generator

train_dataset = BatchGenerator$new(box_output_format = c('class_id', 'xmin', 'xmax', 'ymin', 'ymax')) # This is the format in which the generator is supposed to output the labels. At the moment it **must** be the format set here.

train_dataset$parse_csv(
  images_path = "D:/faces", # make sure to unzip the faces file and put here the correct paths
  labels_path = "train.csv",
  include_classes = 'all',
  input_format = c('image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id')
) # This is the order of the first six columns in the CSV file that contains the labels for your dataset. If your labels are in XML format, maybe the XML parser will be helpful, check the documentation.

# Change the online data augmentation settings as you like
train_generator = train_dataset$generate(batch_size = batch_size,
                                         train = TRUE,
                                         ssd_box_encoder = ssd_box_encoder,
                                         equalize = FALSE,
                                         brightness = c(0.5, 2, 0.5), # Randomly change brightness between 0.5 and 2 with probability 0.5
                                         flip = 0.5, # Randomly flip horizontally with probability 0.5
                                         translate = list(c(5, 50), c(3, 30), 0.5), # Randomly translate by 5-50 pixels horizontally and 3-30 pixels vertically with probability 0.5
                                         scale = c(0.75, 1.3, 0.5), # Randomly scale between 0.75 and 1.3 with probability 0.5
                                         limit_boxes = TRUE,
                                         include_thresh = 0.4,
                                         diagnostics = FALSE)

n_train_samples = train_dataset$get_n_samples()

# 6: Create the validation set batch generator (if you want to use a validation dataset)

val_dataset = BatchGenerator$new(box_output_format = c('class_id', 'xmin', 'xmax', 'ymin', 'ymax'))

val_dataset$parse_csv(images_path = "D:/faces",
                      labels_path = "val.csv",
                      include_classes = 'all',
                      input_format = c('image_name', 'xmin', 'xmax', 'ymin', 'ymax', 'class_id')
                      )

val_generator = val_dataset$generate(batch_size = batch_size,
                                     train = TRUE,
                                     ssd_box_encoder = ssd_box_encoder,
                                     equalize = FALSE,
                                     brightness = FALSE,
                                     flip = FALSE,
                                     translate = FALSE,
                                     scale = FALSE,
                                     limit_boxes = TRUE,
                                     include_thresh = 0.4,
                                     diagnostics = FALSE)

n_val_samples = val_dataset$get_n_samples()

### Run training

# 6: Run training
epochs = 100

history = model$fit_generator(generator = reticulate::py_iterator(train_generator),
                              steps_per_epoch = ceiling(n_train_samples/batch_size),
                              epochs = epochs,
                              validation_data = reticulate::py_iterator(val_generator),
                              validation_steps = ceiling(n_val_samples/batch_size),
                              callbacks = list(
                                callback_model_checkpoint("checkpoints3.h5",
                                                          monitor='val_loss',
                                                          verbose=1,
                                                          save_best_only=TRUE,
                                                          save_weights_only=TRUE,
                                                          mode='auto',
                                                          period=1),
                                callback_early_stopping(monitor='val_loss',
                                                        min_delta=0.001,
                                                        patience=10),
                                callback_reduce_lr_on_plateau(monitor='val_loss',
                                                              factor=0.5,
                                                              patience=0,
                                                              epsilon=0.001,
                                                              cooldown=0)
                              ))

### Make predictions

# 1: Set the generator
predict_generator = val_dataset$generate(batch_size=1L,
                                         train = FALSE,
                                         equalize = FALSE,
                                         brightness=FALSE,
                                         flip=FALSE,
                                         translate=FALSE,
                                         scale=FALSE,
                                         random_crop=FALSE,
                                         crop=FALSE,
                                         resize=FALSE,
                                         gray=FALSE,
                                         limit_boxes=TRUE,
                                         include_thresh=0.4,
                                         diagnostics=FALSE)

# 2: Generate samples
predGen <- predict_generator()

X <- predGen[[1]]
y_true <- predGen[[2]]
filenames <- predGen[[3]]

# 3: Make a prediction
y_pred = model$predict(X)

# 4: Decode the raw prediction `y_pred`
y_pred_decoded = decode_y2(y_pred,
                           confidence_thresh = 0.4,
                           iou_threshold = 0.4,
                           top_k = 'all',
                           input_coords = 'centroids',
                           normalize_coords = FALSE,
                           img_height = NULL,
                           img_width = NULL,
                           n_classes = n_classes,
                           oneOfEach = TRUE)

# 5: Draw the predicted boxes onto the image
classes = c("eyes", "nose", "mouth")

img <- jpeg::readJPEG(stringr::str_c("D:/faces/", filenames))
plot.new()
rasterImage(img, 0, 0, 1, 1)

if (length(y_true) > 0) {
  imgBoxes <- cbind(y_true[[1]][, 2:3] / img_width, 1 -  y_true[[1]][, 4:5] / img_height)
  rect(xleft = imgBoxes[, 1], xright = imgBoxes[, 2], ybottom = imgBoxes[, 3], ytop = imgBoxes[, 4], border = "green")
}

if (dim(y_pred_decoded[[1]])[1] > 0) {
  predBoxes <- cbind(y_pred_decoded[[1]][, 3:4, drop = FALSE] / img_width, 1 - y_pred_decoded[[1]][, 5:6, drop = FALSE] / img_height)
  rect(xleft = predBoxes[, 1], xright = predBoxes[, 2], ybottom = predBoxes[, 3], ytop = predBoxes[, 4], border = "red")
  captions <- str_c(classes[y_pred_decoded[[1]][, 1]], " ", format(round(y_pred_decoded[[1]][, 2], 2), nsmall = 2))
  text(x = predBoxes[, 1], y = predBoxes[, 3], labels = captions, adj = c(-0.1, -0.3), col = "red", cex = 0.8)
}
