#' @export
BatchGenerator <- R6::R6Class("BatchGenerator",
                             public = list(
                               box_output_format = c('class_id', 'xmin', 'xmax', 'ymin', 'ymax'),
                               filenames = NULL,
                               labels = NULL,
                               include_classes = NULL,
                               images_path = NULL,
                               labels_path = NULL,
                               input_format = NULL,

                               # These are the variables that we only need if we want to use parse_xml()
                               images_paths = NULL,
                               annotations_path = NULL,
                               image_set_path = NULL,
                               image_set = NULL,
                               classes = NULL,

                               initialize = function(
                                 box_output_format = c('class_id', 'xmin', 'xmax', 'ymin', 'ymax'),
                                 filenames = NULL,
                                 labels = NULL
                               ) {
                                 library(magick)
                                 # These are the variables we always need
                                 self$include_classes = NULL
                                 self$box_output_format = box_output_format

                                 # These are the variables that we only need if we want to use parse_csv()
                                 self$images_path = NULL
                                 self$labels_path = NULL
                                 self$input_format = NULL

                                 # These are the variables that we only need if we want to use parse_xml()
                                 self$images_paths = NULL
                                 self$annotations_path = NULL
                                 self$image_set_path = NULL
                                 self$image_set = NULL
                                 self$classes = NULL

                                 # The two variables below store the output from the parsers. This is the input for the generate() method.
                                 # `self.filenames` is a list containing all file names of the image samples. Note that it does not contain the actual image files themselves.
                                 # `self.labels` is a list containing one 2D Numpy array per image. For an image with `k` ground truth bounding boxes,
                                 # the respective 2D array has `k` rows, each row containing `(xmin, xmax, ymin, ymax, class_id)` for the respective bounding box.
                                 if (!is.null(filenames)) {
                                   if (is.character(filenames)) {
                                     self$filenames <- readr::read_csv(filenames)
                                   } else if (is.list(filenames) || is.vector(filenames)) {
                                     self$filenames <- filenames
                                   } else {
                                     stop("error")
                                   }
                                 } else {
                                   self$filenames <- list()
                                 }

                                 if (!is.null(labels)) {
                                   if (is.character(labels)) {
                                     self$labels <- readr::read_csv(labels)
                                   } else if (is.list(labels) || is.vector(labels)) {
                                     self$labels <- labels
                                   } else {
                                     stop("error")
                                   }
                                 } else {
                                   self$labels <- list()
                                 }
                               },
                               parse_csv = function(images_path = NULL,
                                                    labels_path = NULL,
                                                    input_format = NULL,
                                                    include_classes = 'all',
                                                    ret = FALSE) {
                                 # If we get arguments in this call, set them
                                 if (!is.null(labels_path)) self$labels_path = labels_path
                                 if (!is.null(input_format)) self$input_format = input_format
                                 if (!is.null(include_classes)) self$include_classes = include_classes
                                 if (!is.null(images_path)) self$images_path = images_path

                                 # Before we begin, make sure that we have a labels_path and an input_format
                                 if (is.null(self$labels_path) || is.null(self$input_format)) {
                                   stop("`labels_path` and/or `input_format` have not been set yet. You need to pass them as arguments.")
                                 }

                                 # Erase data that might have been parsed before
                                 # self$filenames <- list()
                                 # self$labels <- list()

                                 library(tidyverse, quietly = TRUE, warn.conflicts = FALSE)

                                 data <- readr::read_csv(self$labels_path) %>%
                                   mutate(frameID = parse_number(frame)) %>%
                                   arrange(frameID, class_id, xmin, xmax, ymin, ymax) %>% # The data needs to be sorted, otherwise the next step won't give the correct result
                                   select(frame, class_id, xmin, xmax, ymin, ymax)

                                 self$filenames <- unique(data$frame)
                                 self$labels <- data %>%
                                   group_by(frame) %>%
                                   nest() %>%
                                   select(data) %>%
                                   map(identity) %>%
                                   .[[1]] %>%
                                   map(function(x) as.matrix(unname(x)))

                                 if (ret) {# In case we want to return these
                                   return(list(
                                     filenames = self$filenames,
                                     labels = self$labels))
                                 }
                               },
                               generate = function(
                                 batch_size = 32L,
                                 train = TRUE,
                                 ssd_box_encoder = NULL,
                                 equalize = FALSE,
                                 brightness = FALSE,
                                 flip = FALSE,
                                 translate = FALSE,
                                 scale = FALSE,
                                 max_crop_and_resize = FALSE,
                                 full_crop_and_resize = FALSE,
                                 random_crop = FALSE,
                                 crop = FALSE,
                                 resize = FALSE,
                                 gray = FALSE,
                                 limit_boxes = TRUE,
                                 include_thresh = 0.3,
                                 diagnostics = FALSE
                               ) {
                                 samp <- sample(length(self$filenames))
                                 self$filenames <- self$filenames[samp]
                                 self$labels <- self$labels[samp]
                                 # Shuffle the data before we begin
                                 current = 1L
                                 # counter = 0L

                                 # Find out the indices of the box coordinates in the label data
                                 xmin = which(self$box_output_format =='xmin')
                                 xmax = which(self$box_output_format =='xmax')
                                 ymin = which(self$box_output_format =='ymin')
                                 ymax = which(self$box_output_format =='ymax')

                                 function() {
                                   # batch_X = list()
                                   # batch_y = list()

                                   #Shuffle the data after each complete pass
                                   if (current >= length(self$filenames)) {
                                     samp <- sample(length(self$filenames))
                                     self$filenames <- self$filenames[samp]
                                     self$labels <- self$labels[samp]
                                     # Shuffle the data before we begin
                                     current = 1L
                                   }

                                   batchFileNames <- self$filenames[current : min((current + batch_size - 1L), length(self$filenames))]
                                   read_image_matrix <- function(filename) {
                                     # cat(stringr::str_c(self$images_path, "/", filename))
                                     image_to_array(image_load(stringr::str_c(self$images_path, "/", filename)))
                                   }
                                   image_matrices <- map(batchFileNames, read_image_matrix)
                                   batch_X = array (
                                     data = do.call(rbind, map(image_matrices, as.vector)),
                                     dim = c(length(image_matrices), dim(image_matrices[[1]]))
                                   )

                                   batch_y = self$labels[current : min((current + batch_size - 1L), length(self$filenames))]

                                   this_filenames = self$filenames[current : min((current + batch_size - 1L), length(self$filenames))] # The filenames of the files in the current batch

                                   if (diagnostics) {
                                     original_images = batch_X # The original, unaltered images
                                     original_labels = batch_y # The original, unaltered labels
                                   }

                                   current <<- current + batch_size
                                   # counter <<- counter + 1L
                                   # print(counter)

                                   # At this point we're done producing the batch. Now perform some
                                   # optional image transformations:

                                   # batch_items_to_remove = list() # In case we need to remove any images from the batch because of failed random cropping, store their indices in this list
                                   translateBoxes <- function(boxes, xmin, xmax, ymin, ymax, shiftX, shiftY) {

                                     boxes[, xmin] <- boxes[, xmin] + shiftX
                                     boxes[, xmax] <- boxes[, xmax] + shiftX
                                     boxes[, ymin] <- boxes[, ymin] + shiftY
                                     boxes[, ymax] <- boxes[, ymax] + shiftY
                                     boxes
                                   }

                                   flopBoxes <- function(img_width, boxes, shiftX, shiftY) {
                                     newMax <- img_width - boxes[, xmin] + 1
                                     newMin <- img_width - boxes[, xmax] + 1
                                     boxes[, xmin] <-  newMin
                                     boxes[, xmax] <-  newMax
                                     boxes
                                   }

                                   scaleBoxes <- function(boxes, scaleFactor) {
                                     boxes[, 2:5] <- round(boxes[, 2:5] * scaleFactor)
                                     boxes
                                   }

                                   scaleBoxes2 <- function(boxes, scaleFactorX, scaleFactorY) {
                                     boxes[, 2:3] <- round(boxes[, 2:3] * scaleFactorX)
                                     boxes[, 4:5] <- round(boxes[, 4:5] * scaleFactorY)
                                     boxes
                                   }

                                   for (i in seq_len(dim(batch_X)[1])) {
                                     if (equalize) {
                                       tempX <- image_read(batch_X[i, , ,] / 255) %>%
                                         image_equalize() %>%
                                         .[[1]] %>%
                                         as.numeric()
                                       batch_X[i, , ,] <- tempX[, , 1:3] * 255
                                     }

                                     if (brightness[1]) {
                                       if (runif(1) < brightness[3]) {
                                         brightValue <- as.integer(runif(1, min = brightness[1], max = brightness[2]) * 100)
                                         tempX <- image_read(batch_X[i, , ,] / 255) %>%
                                           image_modulate(brightValue) %>%
                                           .[[1]] %>%
                                           as.numeric()
                                         batch_X[i, , ,] <- tempX[, , 1:3] * 255
                                       }
                                     }

                                     if (flip) {
                                       if (runif(1) < flip) {
                                         tempX <- image_read(batch_X[i, , ,] / 255) %>%
                                           image_flop() %>%
                                           .[[1]] %>%
                                           as.numeric()
                                         batch_X[i, , ,] <- tempX[, , 1:3] * 255
                                         batch_y[[i]] <- flopBoxes(dim(batch_X[i,,,])[1], batch_y[[i]], xmin, xmax)
                                       }
                                     }

                                     if (translate[[1]][1]) {
                                       if (runif(1) < translate[[3]]) {
                                         bg_image <- image_read(array(runif(1), c(250, 250, 3)))
                                         shiftX <- sample(c(-1, 1), 1) * as.integer(runif(1, translate[[1]][1], translate[[1]][2]))
                                         shiftY <- sample(c(-1, 1), 1) * as.integer(runif(1, translate[[2]][1], translate[[2]][2]))
                                         shiftXString <- ifelse(shiftX > 0, str_c("+", shiftX), as.character(shiftX))
                                         shiftYString <- ifelse(shiftY > 0, str_c("+", shiftY), as.character(shiftY))
                                         offsetString <- str_c(shiftXString, shiftYString)

                                         temp_image <- image_read(batch_X[i, , ,] / 255)
                                         tempX <- image_composite(bg_image, temp_image, offset = offsetString) %>%
                                           .[[1]] %>%
                                           as.numeric()
                                         batch_X[i, , ,] <- tempX[, , 1:3] * 255
                                         batch_y[[i]] <- translateBoxes(batch_y[[i]], xmin, xmax, ymin, ymax, shiftX, shiftY)
                                       }
                                     }

                                     # if (scale[1]) {
                                     #   if (runif(1) < scale[3]) {
                                     #     bg_image <- image_read(array(runif(1), c(250, 250, 3)))
                                     #     scaleFactorX <- round(runif(1, scale[1], scale[2]), 2)
                                     #     scaleFactorY <- round(runif(1, scale[1], scale[2]), 2)
                                     #     scaleStringX <- str_c(scaleFactorX * 100, "%")
                                     #     scaleStringY <- str_c(scaleFactorY * 100, "%")
                                     #     scaleGeometry <- str_c(scaleStringX, "x", scaleStringY)
                                     #
                                     #     temp_image <- image_read(batch_X[i, , ,] / 255) %>%
                                     #       image_scale(scaleGeometry)
                                     #     tempX <- image_composite(bg_image, temp_image) %>%
                                     #       .[[1]] %>%
                                     #       as.numeric()
                                     #     batch_X[i, , ,] <- tempX[, , 1:3] * 255
                                     #     batch_y[[i]] <- scaleBoxes2(batch_y[[i]], scaleFactorX, scaleFactorY)
                                     #   }
                                     # }
                                     if (scale[1]) {
                                       if (runif(1) < scale[3]) {
                                         bg_image <- image_read(array(runif(1), c(250, 250, 3)))
                                         scaleFactor <- round(runif(1, scale[1], scale[2]), 2)
                                         scaleString <- str_c(scaleFactor * 100, "%")
                                         scaleGeometry <- str_c(scaleString, "x", scaleString)

                                         temp_image <- image_read(batch_X[i, , ,] / 255) %>%
                                           image_scale(scaleGeometry)
                                         tempX <- image_composite(bg_image, temp_image) %>%
                                           .[[1]] %>%
                                           as.numeric()
                                         batch_X[i, , ,] <- tempX[, , 1:3] * 255
                                         batch_y[[i]] <- scaleBoxes(batch_y[[i]], scaleFactor)
                                       }
                                     }
                                   }

                                   if (train) { # During training we need the encoded labels instead of the format that `batch_y` has
                                     if (is.null(ssd_box_encoder)) {
                                       stop("`ssd_box_encoder` cannot be `NULL` in training mode.")
                                     }
                                     y_true = ssd_box_encoder$encode_y(batch_y) # Encode the labels into the `y_true` tensor that the cost function needs
                                   }
                                   if (train) {
                                     return(list(batch_X, y_true))
                                   } else {
                                     return(list(batch_X, batch_y, this_filenames))
                                   }

                                 }
                               },
                              get_n_samples = function() {
                                return(length(self$filenames))
                              }
                             )
)
