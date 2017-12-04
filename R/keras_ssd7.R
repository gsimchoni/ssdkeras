#' Build the SSD7 Model
#'
#' Description
#' @param a a
#' @export
#' @examples
#' build_model()

build_model <- function(image_size,
                        n_classes,
                        min_scale=0.1,
                        max_scale=0.9,
                        scales=None,
                        aspect_ratios_global = c(0.5, 1.0, 2.0),
                        aspect_ratios_per_layer=NULL,
                        two_boxes_for_ar1=TRUE,
                        limit_boxes=TRUE,
                        variances = c(1.0, 1.0, 1.0, 1.0),
                        coords='centroids',
                        normalize_coords=FALSE) {
  
  n_predictor_layers = 4 # The number of predictor conv layers in the network
  
  # Get a few exceptions out of the way first
  if (is.null(aspect_ratios_global) && is.null(aspect_ratios_per_layer)) {
    stop("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
  }
  if (!is.null(aspect_ratios_per_layer)) {
    if (length(aspect_ratios_per_layer) != n_predictor_layers) {
      # stop("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))
      stop("error")
    }
  }
  
  if ((is.null(min_scale) || is.null(max_scale)) && is.null(scales)) {
    stop("Either `min_scale` and `max_scale` or `scales` need to be specified.")
  }
  if (!is.null(scales)) {
    if (length(scales) != n_predictor_layers+1) {
      # stop("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales))))
      stop("error")
    }
  } else {
    scales = seq(min_scale, max_scale,length.out = n_predictor_layers + 1)
  }
  
  if (length(variances) != 4) {
    stop("error")
    # stop("4 variance values must be pased, but {} values were received.".format(len(variances))))
  }
  #variances <- double(4)
  
  if (any(variances <= 0)) {
    stop("error")
    # stop("All variances must be >0, but the variances given are {}".format(variances))
  }
  
  # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
  if (!is.null(aspect_ratios_per_layer)) {
    aspect_ratios_conv4 = aspect_ratios_per_layer[1]
    aspect_ratios_conv5 = aspect_ratios_per_layer[2]
    aspect_ratios_conv6 = aspect_ratios_per_layer[3]
    aspect_ratios_conv7 = aspect_ratios_per_layer[4]
  } else {
    aspect_ratios_conv4 = aspect_ratios_global
    aspect_ratios_conv5 = aspect_ratios_global
    aspect_ratios_conv6 = aspect_ratios_global
    aspect_ratios_conv7 = aspect_ratios_global
  }
  
  # Compute the number of boxes to be predicted per cell for each predictor layer.
  # We need this so that we know how many channels the predictor layers need to have.
  if (!is.null(aspect_ratios_per_layer)) {
    n_boxes = double(0)
    for (aspect_ratios in aspect_ratios_per_layer) {
      if (1 %in% aspect_ratios && two_boxes_for_ar1) {
        n_boxes <- c(n_boxes, length(aspect_ratios) + 1) # +1 for the second box for aspect ratio 1
      } else {
        n_boxes <- c(n_boxes, length(aspect_ratios))
      }
    }
    n_boxes_conv4 = n_boxes[1]
    n_boxes_conv5 = n_boxes[2]
    n_boxes_conv6 = n_boxes[3]
    n_boxes_conv7 = n_boxes[4]
  } else {
    if (1 %in% aspect_ratios_global && two_boxes_for_ar1) {
      n_boxes = length(aspect_ratios_global) + 1
    } else {
      n_boxes = length(aspect_ratios_global)
    }
    n_boxes_conv4 = n_boxes
    n_boxes_conv5 = n_boxes
    n_boxes_conv6 = n_boxes
    n_boxes_conv7 = n_boxes
  }
  
  # Input image format
  img_height <- image_size[1]
  img_width <- image_size[2]
  img_channels <- image_size[3]
  
  # Design the actual network
  x = layer_input(shape = c(img_height, img_width, img_channels))
  normed = layer_lambda(x, function(z) z/127.5 - 1., # Convert input feature range to [-1,1]
                        output_shape = c(img_height, img_width, img_channels),
                        name = 'lambda1')
  
  conv1 = layer_conv_2d(normed, 32, c(5, 5), name ='conv1', strides = c(1, 1), padding= "same")
  conv1 = layer_batch_normalization(conv1, axis=3, momentum =0.99, name = 'bn1')# Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
  conv1 = layer_activation_elu(conv1, name = 'elu1')
  pool1 = layer_max_pooling_2d(conv1, pool_size = c(2, 2), name='pool1')
  
  conv2 = layer_conv_2d(pool1, 48, c(3, 3), name='conv2', strides=c(1, 1), padding="same")
  conv2 = layer_batch_normalization(conv2, axis=3, momentum=0.99, name='bn2')
  conv2 = layer_activation_elu(conv2, name='elu2')
  pool2 = layer_max_pooling_2d(conv2, pool_size=c(2, 2), name='pool2')
  
  conv3 = layer_conv_2d(pool2, 64, c(3, 3), name='conv3', strides=c(1, 1), padding="same")
  conv3 = layer_batch_normalization(conv3, axis=3, momentum=0.99, name='bn3')
  conv3 = layer_activation_elu(conv3, name='elu3')
  pool3 = layer_max_pooling_2d(conv3, pool_size = c(2, 2), name='pool3')
  
  conv4 = layer_conv_2d(pool3, 64, c(3, 3), name='conv4', strides=c(1, 1), padding="same")
  conv4 = layer_batch_normalization(conv4, axis=3, momentum=0.99, name='bn4')
  conv4 = layer_activation_elu(conv4, name='elu4')
  pool4 = layer_max_pooling_2d(conv4, pool_size=c(2, 2), name='pool4')
  
  conv5 = layer_conv_2d(pool4, 48, c(3, 3), name='conv5', strides=c(1, 1), padding="same")
  conv5 = layer_batch_normalization(conv5, axis=3, momentum=0.99, name='bn5')
  conv5 = layer_activation_elu(conv5, name='elu5')
  pool5 = layer_max_pooling_2d(conv5, pool_size=c(2, 2), name='pool5')
  
  conv6 = layer_conv_2d(pool5, 48, c(3, 3), name='conv6', strides=c(1, 1), padding="same")
  conv6 = layer_batch_normalization(conv6, axis=3, momentum=0.99, name='bn6')
  conv6 = layer_activation_elu(conv6, name='elu6')
  pool6 = layer_max_pooling_2d(conv6, pool_size=c(2, 2), name='pool6')
  
  conv7 = layer_conv_2d(pool6, 32, c(3, 3), name='conv7', strides=c(1, 1), padding="same")
  conv7 = layer_batch_normalization(conv7, axis=3, momentum=0.99, name='bn7')
  conv7 = layer_activation_elu(conv7, name='elu7')
  
  # The next part is to add the convolutional predictor layers on top of the base network
  # that we defined above. Note that I use the term "base network" differently than the paper does.
  # To me, the base network is everything that is not convolutional predictor layers or anchor
  # box layers. In this case we'll have four predictor layers, but of course you could
  # easily rewrite this into an arbitrarily deep base network and add an arbitrary number of
  # predictor layers on top of the base network by simply following the pattern shown here.
  
  # Build the convolutional predictor layers on top of conv layers 4, 5, 6, and 7
  # We build two predictor layers on top of each of these layers: One for classes (classification), one for box coordinates (localization)
  # We precidt `n_classes` confidence values for each box, hence the `classes` predictors have depth `n_boxes * n_classes`
  # We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
  # Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
  classes4 = layer_conv_2d(conv4, n_boxes_conv4 * n_classes, c(3, 3), strides=c(1, 1), padding="valid", name='classes4')
  classes5 = layer_conv_2d(conv5, n_boxes_conv5 * n_classes, c(3, 3), strides=c(1, 1), padding="valid", name='classes5')
  classes6 = layer_conv_2d(conv6, n_boxes_conv6 * n_classes, c(3, 3), strides=c(1, 1), padding="valid", name='classes6')
  classes7 = layer_conv_2d(conv7, n_boxes_conv7 * n_classes, c(3, 3), strides=c(1, 1), padding="valid", name='classes7')
  # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
  boxes4 = layer_conv_2d(conv4, n_boxes_conv4 * 4, c(3, 3), strides=c(1, 1), padding="valid", name='boxes4')
  boxes5 = layer_conv_2d(conv5, n_boxes_conv5 * 4, c(3, 3), strides=c(1, 1), padding="valid", name='boxes5')
  boxes6 = layer_conv_2d(conv6, n_boxes_conv6 * 4, c(3, 3), strides=c(1, 1), padding="valid", name='boxes6')
  boxes7 = layer_conv_2d(conv7, n_boxes_conv7 * 4, c(3, 3), strides=c(1, 1), padding="valid", name='boxes7')
  
  # Generate the anchor boxes
  # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
  anchors4 = layer_anchor(boxes4, img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios_conv4,
                          two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors4')
  anchors5 = layer_anchor(boxes5, img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios_conv5,
                          two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors5')
  anchors6 = layer_anchor(boxes6, img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios_conv6,
                          two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors6')
  anchors7 = layer_anchor(boxes7, img_height, img_width, this_scale=scales[4], next_scale=scales[5], aspect_ratios=aspect_ratios_conv7,
                          two_boxes_for_ar1=two_boxes_for_ar1, limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors7')
  
  # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
  # We want the classes isolated in the last axis to perform softmax on them
  classes4_reshaped = layer_reshape(classes4, c(-1, n_classes), name='classes4_reshape')
  classes5_reshaped = layer_reshape(classes5, c(-1, n_classes), name='classes5_reshape')
  classes6_reshaped = layer_reshape(classes6, c(-1, n_classes), name='classes6_reshape')
  classes7_reshaped = layer_reshape(classes7, c(-1, n_classes), name='classes7_reshape')
  # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
  # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
  boxes4_reshaped = layer_reshape(boxes4, c(-1, 4), name='boxes4_reshape')
  boxes5_reshaped = layer_reshape(boxes5, c(-1, 4), name='boxes5_reshape')
  boxes6_reshaped = layer_reshape(boxes6, c(-1, 4), name='boxes6_reshape')
  boxes7_reshaped = layer_reshape(boxes7, c(-1, 4), name='boxes7_reshape')
  # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
  anchors4_reshaped = layer_reshape(anchors4, c(-1, 8), name='anchors4_reshape')
  anchors5_reshaped = layer_reshape(anchors5, c(-1, 8), name='anchors5_reshape')
  anchors6_reshaped = layer_reshape(anchors6, c(-1, 8), name='anchors6_reshape')
  anchors7_reshaped = layer_reshape(anchors7, c(-1, 8), name='anchors7_reshape')

  # Concatenate the predictions from the different layers and the assosciated anchor box tensors
  # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
  # so we want to concatenate along axis 1
  # Output shape of `classes_merged`: (batch, n_boxes_total, n_classes)
  classes_concat = layer_concatenate(list(classes4_reshaped,
                                          classes5_reshaped,
                                          classes6_reshaped,
                                          classes7_reshaped), axis=1, name='classes_concat')
  
  # Output shape of `boxes_final`: (batch, n_boxes_total, 4)
  boxes_concat = layer_concatenate(list(boxes4_reshaped,
                                        boxes5_reshaped,
                                        boxes6_reshaped,
                                        boxes7_reshaped), axis=1, name='boxes_concat')
  
  # Output shape of `anchors_final`: (batch, n_boxes_total, 8)
  anchors_concat = layer_concatenate(list(anchors4_reshaped,
                                          anchors5_reshaped,
                                          anchors6_reshaped,
                                          anchors7_reshaped), axis=1, name='anchors_concat')
  
  # The box coordinate predictions will go into the loss function just the way they are,
  # but for the class predictions, we'll apply a softmax activation layer first
  classes_softmax = layer_activation(classes_concat, 'softmax', name='classes_softmax')
  
  # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
  # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
  predictions = layer_concatenate(list(classes_softmax, boxes_concat, anchors_concat),
                                  axis=2, name='predictions')
  
  model = keras_model(inputs = x, outputs = predictions)
  
  # Get the spatial dimensions (height, width) of the convolutional predictor layers, we need them to generate the default boxes
  # The spatial dimensions are the same for the `classes` and `boxes` predictors
  predictor_sizes = matrix(c(unlist(classes4$shape$as_list())[1:2],
                             unlist(classes5$shape$as_list())[1:2],
                             unlist(classes6$shape$as_list())[1:2],
                             unlist(classes7$shape$as_list())[1:2]), nrow = 4, ncol = 2, byrow = TRUE)
  
  return(list(model = model, predictor_sizes = predictor_sizes))
}

