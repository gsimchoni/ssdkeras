
iou <- function(boxes1, boxes2, coords = 'centroids') {
  np <- reticulate::import("numpy")

  if (length(dim(boxes1)) > 2) stop("boxes1 must have rank either 1 or 2, but has rank {}.")#.format(len(boxes1.shape)))
  if (length(dim(boxes2)) > 2) stop("boxes2 must have rank either 1 or 2, but has rank {}.")#.format(len(boxes2.shape)))

  if (is.null(dim(boxes1))) boxes1 = np$expand_dims(boxes1, axis = 0L)
  if (is.null(dim(boxes2))) boxes2 = np$expand_dims(boxes2, axis = 0L)

  if (!(dim(boxes1)[2] == 4 && dim(boxes2)[2] == 4)) stop("It must be boxes1.shape[1] == boxes2.shape[1] == 4, but it is boxes1.shape[1] == {}, boxes2.shape[1] == {}.")#.format(boxes1.shape[1], boxes2.shape[1]))

  if (coords == 'centroids') {
    # TODO: Implement a version that uses fewer computation steps (that doesn't need conversion)
    boxes1 = convert_coordinates2D(boxes1, start_index = 1L, conversion = 'centroids2minmax')
    boxes2 = convert_coordinates2D(boxes2, start_index = 1L, conversion = 'centroids2minmax')
  } else if (coords != 'minmax') {
    stop("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")
  }

  intersection = np$maximum(0, np$minimum(boxes1[, 2], boxes2[, 2]) - np$maximum(boxes1[, 1], boxes2[, 1])) * np$maximum(0, np$minimum(boxes1[, 4], boxes2[, 4]) - np$maximum(boxes1[, 3], boxes2[, 3]))
  union = (boxes1[, 2] - boxes1[, 1]) * (boxes1[, 4] - boxes1[, 3]) + (boxes2[, 2] - boxes2[, 1]) * (boxes2[, 4] - boxes2[, 3]) - intersection

  return(intersection / union)
}

convert_coordinates <- function(tensor, start_index, conversion='minmax2centroids') {
  ind = start_index
  tensor1 = tensor
  if (conversion == 'minmax2centroids') {
    tensor1[, , , ind] = (tensor[, , , ind] + tensor[, , , ind+1]) / 2.0 # Set cx
    tensor1[, , , ind+1] = (tensor[, , , ind+2] + tensor[, , , ind+3]) / 2.0 # Set cy
    tensor1[, , , ind+2] = tensor[, , , ind+1] - tensor[, , , ind] # Set w
    tensor1[, , , ind+3] = tensor[, , , ind+3] - tensor[, , , ind+2] # Set h
  } else if (conversion == 'centroids2minmax') {
    tensor1[, , , ind] = tensor[, , , ind] - tensor[, , , ind+2] / 2.0 # Set xmin
    tensor1[, , , ind+1] = tensor[, , , ind] + tensor[, , , ind+2] / 2.0 # Set xmax
    tensor1[, , , ind+2] = tensor[, , , ind+1] - tensor[, , , ind+3] / 2.0 # Set ymin
    tensor1[, , , ind+3] = tensor[, , , ind+1] + tensor[, , , ind+3] / 2.0 # Set ymax
  } else {
    stop("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")
  }

  return(tensor1)
}

convert_coordinates1D <- function(tensor, start_index, conversion='minmax2centroids') {
  ind = start_index
  tensor1 = tensor
  if (conversion == 'minmax2centroids') {
    tensor1[ind] = (tensor[ind] + tensor[ind + 1]) / 2.0 # Set cx
    tensor1[ind + 1] = (tensor[ind + 2] + tensor[ind + 3]) / 2.0 # Set cy
    tensor1[ind + 2] = tensor[ind + 1] - tensor[ind] # Set w
    tensor1[ind + 3] = tensor[ind + 3] - tensor[ind + 2] # Set h
  } else if (conversion == 'centroids2minmax') {
    tensor1[ind] = tensor[ind] - tensor[ind + 2] / 2.0 # Set xmin
    tensor1[ind + 1] = tensor[ind] + tensor[ind + 2] / 2.0 # Set xmax
    tensor1[ind + 2] = tensor[ind + 1] - tensor[ind + 3] / 2.0 # Set ymin
    tensor1[ind + 3] = tensor[ind + 1] + tensor[ind + 3] / 2.0 # Set ymax
  } else {
    stop("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")
  }

  return(tensor1)
}

convert_coordinates2D <- function(tensor, start_index, conversion='minmax2centroids') {
  ind = start_index
  tensor1 = tensor
  if (conversion == 'minmax2centroids') {
    tensor1[, ind] = (tensor[, ind] + tensor[, ind+1]) / 2.0 # Set cx
    tensor1[, ind+1] = (tensor[, ind+2] + tensor[, ind+3]) / 2.0 # Set cy
    tensor1[, ind+2] = tensor[, ind+1] - tensor[, ind] # Set w
    tensor1[, ind+3] = tensor[, ind+3] - tensor[, ind+2] # Set h
  } else if (conversion == 'centroids2minmax') {
    tensor1[, ind] = tensor[, ind] - tensor[, ind+2] / 2.0 # Set xmin
    tensor1[, ind+1] = tensor[, ind] + tensor[, ind+2] / 2.0 # Set xmax
    tensor1[, ind+2] = tensor[, ind+1] - tensor[, ind+3] / 2.0 # Set ymin
    tensor1[, ind+3] = tensor[, ind+1] + tensor[, ind+3] / 2.0 # Set ymax
  } else {
    stop("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")
  }

  return(tensor1)
}

convert_coordinates3D <- function(tensor, start_index, conversion='minmax2centroids') {
  ind = start_index
  tensor1 = tensor
  if (conversion == 'minmax2centroids') {
    tensor1[, , ind] = (tensor[, , ind] + tensor[, , ind+1]) / 2.0 # Set cx
    tensor1[, , ind+1] = (tensor[, , ind+2] + tensor[, , ind+3]) / 2.0 # Set cy
    tensor1[, , ind+2] = tensor[, , ind+1] - tensor[, , ind] # Set w
    tensor1[, , ind+3] = tensor[, , ind+3] - tensor[, , ind+2] # Set h
  } else if (conversion == 'centroids2minmax') {
    tensor1[, , ind] = tensor[, , ind] - tensor[, , ind+2] / 2.0 # Set xmin
    tensor1[, , ind+1] = tensor[, , ind] + tensor[, , ind+2] / 2.0 # Set xmax
    tensor1[, , ind+2] = tensor[, , ind+1] - tensor[, , ind+3] / 2.0 # Set ymin
    tensor1[, , ind+3] = tensor[, , ind+1] + tensor[, , ind+3] / 2.0 # Set ymax
  } else {
    stop("Unexpected conversion value. Supported values are 'minmax2centroids' and 'centroids2minmax'.")
  }

  return(tensor1)
}

greedy_nms2 <- function(predictions, iou_threshold=0.45, coords='minmax') {
  np <- reticulate::import("numpy")
  boxes_left = np$copy(predictions)
  maxima = list() # This is where we store the boxes that make it through the non-maximum suppression
  idx = 1
  while (!is.null(dim(boxes_left)) && dim(boxes_left)[1] > 0) { # While there are still boxes left to compare...
    maximum_index = np$argmax(boxes_left[, 2]) + 1L # ...get the index of the next box with the highest confidence...
    maximum_box = np$copy(boxes_left[maximum_index, ]) # ...copy that box and...
    maxima[[idx]] <- maximum_box # ...append it to `maxima` because we'll definitely keep it
    idx <- idx + 1
    boxes_left = np$delete(boxes_left, maximum_index - 1L, axis = 0L) # Now remove the maximum box from `boxes_left`
    if (is.null(dim(boxes_left))) {
      break # If there are no boxes left after this step, break. Otherwise...
    }
    similarities = iou(boxes_left[, 3:6], array(maximum_box[3:6], c(1,4)), coords = coords) # ...compare (IoU) the other left over boxes to the maximum box...
    boxes_left = boxes_left[similarities <= iou_threshold, ] # ...so that we can remove the ones that overlap too much with the maximum box
  }
  return(do.call(rbind, maxima))
}

#' @export
decode_y2 <- function(
  y_pred,
  confidence_thresh = 0.5,
  iou_threshold = 0.45,
  top_k = 'all',
  input_coords ='centroids',
  normalize_coords = FALSE,
  img_height = NULL,
  img_width = NULL,
  n_classes = NULL,
  oneOfEach = FALSE
) {
  np <- reticulate::import("numpy")
  if (normalize_coords && (is.null(img_height) || is.null(img_width))) {
    stop("error")
  }

  # 1: Convert the classes from one-hot encoding to their class ID
  y_pred_converted = np$copy(y_pred[, , (n_classes - 1):(n_classes + 4), drop = FALSE]) # Slice out the four offset predictions plus two elements whereto we'll write the class IDs and confidences in the next step
  y_pred_converted[, , 1] = np$argmax(y_pred[, , 1:n_classes], axis = -1L) # The indices of the highest confidence values in the one-hot class vectors are the class ID
  y_pred_converted[, , 2] = np$amax(y_pred[, , 1:n_classes], axis = -1L) # Store the confidence values themselves, too

  # 2: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
  if (input_coords == 'centroids') {
    y_pred_converted[, , c(5, 6)] = np$exp(y_pred_converted[, , c(5, 6)] * y_pred[, , c(n_classes + 11, n_classes + 12)]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
    y_pred_converted[, , c(5, 6)] = y_pred_converted[, , c(5, 6)] * y_pred[, , c(n_classes + 7, n_classes + 8)] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
    y_pred_converted[, , c(3, 4)] = y_pred_converted[, , c(3, 4)] * (y_pred[, , c(n_classes + 9, n_classes + 10)] * y_pred[, , c(n_classes + 7, n_classes + 8)]) # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
    y_pred_converted[, , c(3, 4)] = y_pred_converted[, , c(3, 4)] + y_pred[, , c(n_classes + 5, n_classes + 6)] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
    y_pred_converted = convert_coordinates3D(y_pred_converted, start_index = 3L, conversion='centroids2minmax')
  } else if (input_coords == 'minmax') {
    y_pred_converted[, , 3:6] = y_pred_converted[, , 3:6] * y_pred[, , (n_classes + 9):(n_classes + 12)] # delta(pred) / size(anchor) / variance * variance == delta(pred) / size(anchor) for all four coordinates, where 'size' refers to w or h, respectively
    y_pred_converted[, , c(3,4)] = y_pred_converted[, , c(3,4)] * np$expand_dims(y_pred[, , n_classes + 6] - y_pred[, , n_classes + 5], axis = -1L) # delta_xmin(pred) / w(anchor) * w(anchor) == delta_xmin(pred), delta_xmax(pred) / w(anchor) * w(anchor) == delta_xmax(pred)
    y_pred_converted[, , c(5,6)] = y_pred_converted[, , c(5,6)] * np$expand_dims(y_pred[, , n_classes + 8] - y_pred[, , n_classes + 7], axis = -1L) # delta_ymin(pred) / h(anchor) * h(anchor) == delta_ymin(pred), delta_ymax(pred) / h(anchor) * h(anchor) == delta_ymax(pred)
    y_pred_converted[, , 3:6] = y_pred_converted[, , 3:6] + y_pred[, , (n_classes + 5):(n_classes + 8)] # delta(pred) + anchor == pred for all four coordinates
  } else {
    stop("error")
  }

  # 3: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that
  if (normalize_coords) {
    y_pred_converted[, , 3:4] = y_pred_converted[, , 3:4] * img_width # Convert xmin, xmax back to absolute coordinates
    y_pred_converted[, , 5:6] = y_pred_converted[, , 5:6] * img_height # Convert ymin, ymax back to absolute coordinates
  }

  # 4: Decode our huge `(batch, #boxes, 6)` tensor into a list of length `batch` where each list entry is an array containing only the positive predictions
  y_pred_decoded = list()
  for (i in seq_len(dim(y_pred_converted)[1])) { # For each image in the batch...
    batch_item <- y_pred_converted[i, , ]
    boxes = batch_item[unlist(np$nonzero(batch_item[, 1])) + 1, , drop = FALSE] # ...get all boxes that don't belong to the background class,...
    boxes = boxes[boxes[, 2] >= confidence_thresh, , drop = FALSE] # ...then filter out those positive boxes for which the prediction confidence is too low and after that...
    if (oneOfEach) {
      boxes <- boxes %>% as_tibble() %>% group_by(V1) %>% top_n(1, V2) %>% as.matrix()
    } else {
      if (!is.null(iou_threshold)) { # ...if an IoU threshold is set...
        boxes = greedy_nms2(boxes, iou_threshold = iou_threshold, coords = 'minmax') # ...perform NMS on the remaining boxes.
      }
      if (top_k != 'all' && !is.null(dim(boxes)) && dim(boxes)[1] > top_k) { # If we have more than `top_k` results left at this point...
        top_k_indices = np$argpartition(boxes[, 2], kth = dim(boxes)[1] - top_k, axis = 0L)[(dim(boxes)[1] - top_k + 1):dim(boxes)[1]] + 1 # ...get the indices of the `top_k` highest-scoring boxes...
        boxes = boxes[top_k_indices, ] # ...and keep only those boxes...
      }
    }
    y_pred_decoded[[i]] <- if (is.null(boxes)) {matrix(, nrow = 0, ncol = 6)} else {boxes} # ...and now that we're done, append the array of final predictions for this batch item to the output list
  }


  return(y_pred_decoded)
}

#' @export
SSDBoxEncoder <- R6::R6Class("SSDBoxEncoder",
                             public = list(
                               img_height = NULL,
                               img_width = NULL,
                               n_classes = NULL,
                               predictor_sizes = NULL,
                               min_scale = 0.1,
                               max_scale = 0.9,
                               scales = NULL,
                               aspect_ratios_global = c(0.5, 1.0, 2.0),
                               aspect_ratios_per_layer = NULL,
                               two_boxes_for_ar1 = TRUE,
                               limit_boxes = TRUE,
                               variances = c(1.0, 1.0, 1.0, 1.0),
                               pos_iou_threshold = 0.5,
                               neg_iou_threshold = 0.3,
                               coords = 'centroids',
                               normalize_coords = FALSE,
                               n_boxes = NULL,

                               initialize = function(
                                 img_height = NULL,
                                 img_width = NULL,
                                 n_classes = NULL,
                                 predictor_sizes = NULL,
                                 min_scale = 0.1,
                                 max_scale = 0.9,
                                 scales = NULL,
                                 aspect_ratios_global = c(0.5, 1.0, 2.0),
                                 aspect_ratios_per_layer = NULL,
                                 two_boxes_for_ar1 = TRUE,
                                 limit_boxes = TRUE,
                                 variances = c(1.0, 1.0, 1.0, 1.0),
                                 pos_iou_threshold = 0.5,
                                 neg_iou_threshold = 0.3,
                                 coords = 'centroids',
                                 normalize_coords = FALSE
                               ) {

                                 if (is.vector(predictor_sizes)) {
                                   predictor_sizes = array(predictor_sizes, c(1, length(predictor_sizes)))
                                 } else {
                                   predictor_sizes = array(predictor_sizes, dim(predictor_sizes))
                                 }

                                 if ((is.null(min_scale) || is.null(max_scale)) && is.null(scales)) {
                                   stop("Either `min_scale` and `max_scale` or `scales` need to be specified.")
                                 }

                                 if (!is.null(scales)) {
                                   if (length(scales) != dim(predictor_sizes)[1] + 1) {
                                     stop("It must be either scales is None or len(scales) == len(predictor_sizes)+1, but len(scales) == {} and len(predictor_sizes)+1 == {}")
                                   }
                                   scales = array(scales, dim = c(length(scales), 1))
                                   if (any(scales <= 0)) {
                                     stop("error")
                                   }
                                 } else {
                                   if (!(0 < min_scale && min_scale <= max_scale)) {
                                     stop("error")
                                   }
                                 }

                                 if (!is.null(aspect_ratios_per_layer)) {
                                   if (length(aspect_ratios_per_layer) != dim(predictor_sizes)[1]) {
                                     stop("error")
                                   }
                                   for (aspect_ratios in aspect_ratios_per_layer) {
                                     aspect_ratios = array(aspect_ratios)
                                     if (any(aspect_ratios <= 0)) {
                                       stop("All aspect ratios must be greater than zero.")
                                     }
                                   }
                                 } else {
                                   if (is.null(aspect_ratios_global)) {
                                     stop("At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` cannot be `None`.")
                                   }
                                   aspect_ratios_global = array(aspect_ratios_global)
                                   if (any(aspect_ratios_global <= 0)) {
                                     stop("All aspect ratios must be greater than zero.")
                                   }
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

                                 if (neg_iou_threshold > pos_iou_threshold) {
                                   stop("It cannot be `neg_iou_threshold > pos_iou_threshold`.")
                                 }

                                 if (!(coords == 'minmax' || coords == 'centroids')) {
                                   stop("Unexpected value for `coords`. Supported values are 'minmax' and 'centroids'.")
                                 }

                                 self$img_height = img_height
                                 self$img_width = img_width
                                 self$n_classes = as.integer(n_classes)
                                 self$predictor_sizes = predictor_sizes
                                 self$min_scale = min_scale
                                 self$max_scale = max_scale
                                 self$scales = scales
                                 self$aspect_ratios_global = aspect_ratios_global
                                 self$aspect_ratios_per_layer = aspect_ratios_per_layer
                                 self$two_boxes_for_ar1 = two_boxes_for_ar1
                                 self$limit_boxes = limit_boxes
                                 self$variances = variances
                                 self$pos_iou_threshold = pos_iou_threshold
                                 self$neg_iou_threshold = neg_iou_threshold
                                 self$coords = coords
                                 self$normalize_coords = normalize_coords

                                 if (!is.null(aspect_ratios_per_layer)) {
                                   self$n_boxes = double(0)
                                   for (aspect_ratios in aspect_ratios_per_layer) {
                                     if (1 %in% aspect_ratios && two_boxes_for_ar1) {
                                       self$n_boxes <- c(self$n_boxes, length(aspect_ratios) + 1) # +1 for the second box for aspect ratio 1
                                     } else {
                                       self$n_boxes <- c(self$n_boxes, length(aspect_ratios))
                                     }
                                   }
                                 } else {
                                   if (1 %in% aspect_ratios_global && two_boxes_for_ar1) {
                                     self$n_boxes = length(aspect_ratios_global) + 1
                                   } else {
                                     self$n_boxes = length(aspect_ratios_global)
                                   }
                                 }
                               },
                                 generate_anchor_boxes = function(
                                   batch_size,
                                   feature_map_size,
                                   aspect_ratios,
                                   this_scale,
                                   next_scale,
                                   diagnostics = FALSE
                                 ) {

                                   np = reticulate::import("numpy")

                                   feature_map_size = as.integer(feature_map_size)
                                   aspect_ratios = sort(aspect_ratios)

                                   size = min(self$img_height, self$img_width)
                                   # Compute the box widths and and heights for all aspect ratios
                                   wh_list = data.frame(w = double(0), h = double(0))
                                   n_boxes = length(aspect_ratios)
                                   for (ar in aspect_ratios) {
                                     if ((ar == 1) && self$two_boxes_for_ar1) {
                                       # Compute the regular default box for aspect ratio 1 and...
                                       w = this_scale * size * sqrt(ar)
                                       h = this_scale * size / sqrt(ar)
                                       wh_list <- rbind.data.frame(wh_list, data.frame(w,h))
                                       # ...also compute one slightly larger version using the geometric mean of this scale value and the next
                                       w = sqrt(this_scale * next_scale) * size * sqrt(ar)
                                       h = sqrt(this_scale * next_scale) * size / sqrt(ar)
                                       wh_list <- rbind.data.frame(wh_list, data.frame(w,h))
                                       n_boxes <- n_boxes + 1L
                                     } else {
                                       w = this_scale * size * sqrt(ar)
                                       h = this_scale * size / sqrt(ar)
                                       wh_list <- rbind.data.frame(wh_list, data.frame(w,h))
                                     }
                                   }

                                   # Compute the grid of box center points. They are identical for all aspect ratios
                                   cell_height = self$img_height / feature_map_size[1]
                                   cell_width = self$img_width / feature_map_size[2]
                                   cx = seq(cell_width/2, self$img_width-cell_width/2, length.out = feature_map_size[2])
                                   cy = seq(cell_height/2, self$img_height-cell_height/2, length.out = feature_map_size[1])
                                   grid = np$meshgrid(cx, cy)
                                   cx_grid = grid[[1]]
                                   cy_grid = grid[[2]]
                                   cx_grid = np$expand_dims(cx_grid, -1L)
                                   cy_grid = np$expand_dims(cy_grid, -1L)

                                   # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 4)`
                                   # where the last dimension will contain `(cx, cy, w, h)`
                                   # boxes_tensor = array(0, dim = c(feature_map_height, feature_map_width, self$n_boxes, 4))
                                   boxes_tensor = np$zeros(reticulate::tuple(feature_map_size[1], feature_map_size[2], n_boxes, 4L))

                                   # boxes_tensor[, , , 1] = array(cx_grid, dim = c(feature_map_height, feature_map_width, self$n_boxes)) #np.tile(cx_grid, (1, 1, self$n_boxes)) # Set cx
                                   boxes_tensor[, , , 1] = np$tile(cx_grid, reticulate::tuple(1L, 1L, n_boxes)) # Set cx
                                   # boxes_tensor[, , , 2] = array(cy_grid, dim = c(feature_map_height, feature_map_width, self$n_boxes)) #np.tile(cy_grid, (1, 1, self$n_boxes)) # Set cy
                                   boxes_tensor[, , , 2] = np$tile(cy_grid, reticulate::tuple(1L, 1L, n_boxes)) # Set cx
                                   boxes_tensor[, , , 3] = array(rep(wh_list[, 1], each = feature_map_size[1] * feature_map_size[2]), dim = c(feature_map_size[1], feature_map_size[2], n_boxes)) # Set w
                                   boxes_tensor[, , , 4] = array(rep(wh_list[, 2], each = feature_map_size[1] * feature_map_size[2]), dim = c(feature_map_size[1], feature_map_size[2], n_boxes)) # Set h

                                   # Convert `(cx, cy, w, h)` to `(xmin, xmax, ymin, ymax)`
                                   boxes_tensor = convert_coordinates(boxes_tensor, start_index=1L, conversion='centroids2minmax')

                                   # If `limit_boxes` is enabled, clip the coordinates to lie within the image boundaries
                                   if (self$limit_boxes) {
                                     x_coords = boxes_tensor[, , , c(1, 2)]
                                     x_coords[x_coords >= self$img_width] = self$img_width - 1L
                                     x_coords[x_coords < 0] = 0L
                                     boxes_tensor[, , , c(1, 2)] = x_coords
                                     y_coords = boxes_tensor[, , , c(3, 4)]
                                     y_coords[y_coords >= self$img_height] = self$img_height - 1L
                                     y_coords[y_coords < 0] = 0L
                                     boxes_tensor[, , , c(3, 4)] = y_coords
                                   }

                                   # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
                                   if (self$normalize_coords) {
                                     boxes_tensor[, , , 1:2] = boxes_tensor[, , , 1:2] / self$img_width
                                     boxes_tensor[, , , 3:4] = boxes_tensor[, , , 3:4] / self$img_height
                                   }

                                   if (self$coords == 'centroids') {
                                     # TODO: Implement box limiting directly for `(cx, cy, w, h)` so that we don't have to unnecessarily convert back and forth
                                     # Convert `(xmin, xmax, ymin, ymax)` back to `(cx, cy, w, h)`
                                     boxes_tensor = convert_coordinates(boxes_tensor, start_index=1L, conversion='minmax2centroids')
                                   }
                                   # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
                                   # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 8)`
                                   boxes_tensor = np$expand_dims(boxes_tensor, axis = 0L)
                                   boxes_tensor = np$tile(boxes_tensor, reticulate::tuple(batch_size, 1L, 1L, 1L, 1L))

                                   # Now reshape the 5D tensor above into a 3D tensor of shape
                                   # `(batch, feature_map_height * feature_map_width * n_boxes, 4)`. The resulting
                                   # order of the tensor content will be identical to the order obtained from the reshaping operation
                                   # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
                                   # use the same default index order, which is C-like index ordering)
                                   boxes_tensor = np$reshape(boxes_tensor, reticulate::tuple(batch_size, -1L, 4L))

                                   if (diagnostics) {
                                     return(list(
                                       boxes_tensor = boxes_tensor,
                                       wh_list = wh_list,
                                       cell_measures = as.integer(c(cell_height, cell_width))))
                                   } else {
                                     return(boxes_tensor)
                                   }
                                },
                               generate_encode_template = function(batch_size, diagnostics=FALSE) {
                                 np = reticulate::import("numpy")
                                 # 1: Get the anchor box scaling factors for each conv layer from which we're going to make predictions
                                 #    If `scales` is given explicitly, we'll use that instead of computing it from `min_scale` and `max_scale`
                                 if (is.null(self$scales)) {
                                   self$scales = seq(self$min_scale, self$max_scale,length.out = dim(self$predictor_sizes)[1] + 1L)
                                 }

                                 # 2: For each conv predictor layer (i.e. for each scale factor) get the tensors for
                                 #    the anchor box coordinates of shape `(batch, n_boxes_total, 4)`
                                 boxes_tensor = list()
                                 if (diagnostics) {
                                   wh_list = list() # List to hold the box widths and heights
                                   cell_sizes = list() # List to hold horizontal and vertical distances between any two boxes

                                   if (!is.null(self$aspect_ratios_per_layer)) { # If individual aspect ratios are given per layer, we need to pass them to `generate_anchor_boxes()` accordingly
                                     for (i in seq_len(dim(self$predictor_sizes)[1])) {
                                       res = self$generate_anchor_boxes(batch_size = batch_size,
                                                                        feature_map_size = self$predictor_sizes[i, ],
                                                                        aspect_ratios = self$aspect_ratios_per_layer[i],
                                                                        this_scale = self$scales[i],
                                                                        next_scale = self$scales[i + 1],
                                                                        diagnostics = TRUE)
                                       boxes = res$boxes_tensor
                                       wh = res$wh_list
                                       cells = res$cell_measures
                                       boxes_tensor[[i]] <- boxes
                                       wh_list[[i]] <- wh
                                       cell_sizes[[i]] <- cells
                                     }
                                   } else { # Use the same global aspect ratio list for all layers
                                     for (i in seq_len(dim(self$predictor_sizes)[1])) {
                                       res = self$generate_anchor_boxes(batch_size = batch_size,
                                                                        feature_map_size = self$predictor_sizes[i, ],
                                                                        aspect_ratios = self$aspect_ratios_global,
                                                                        this_scale = self$scales[i],
                                                                        next_scale = self$scales[i + 1],
                                                                        diagnostics = TRUE)
                                       boxes = res$boxes_tensor
                                       wh = res$wh_list
                                       cells = res$cell_measures
                                       boxes_tensor[[i]] <- boxes
                                       wh_list[[i]] <- wh
                                       cell_sizes[[i]] <- cells
                                     }
                                   }
                                 } else {
                                   if (!is.null(self$aspect_ratios_per_layer)) {
                                     for (i in seq_len(dim(self$predictor_sizes)[1])) {
                                       boxes_tensor[[i]] <- self$generate_anchor_boxes(batch_size = batch_size,
                                                                        feature_map_size = self$predictor_sizes[i, ],
                                                                        aspect_ratios = self$aspect_ratios_per_layer[i],
                                                                        this_scale = self$scales[i],
                                                                        next_scale = self$scales[i + 1],
                                                                        diagnostics = FALSE)
                                     }
                                   } else {
                                     for (i in seq_len(dim(self$predictor_sizes)[1])) {
                                       boxes_tensor[[i]] <- self$generate_anchor_boxes(batch_size = batch_size,
                                                                          feature_map_size = self$predictor_sizes[i, ],
                                                                          aspect_ratios = self$aspect_ratios_global,
                                                                          this_scale = self$scales[i],
                                                                          next_scale = self$scales[i + 1],
                                                                          diagnostics = FALSE)
                                     }
                                   }
                                 }

                                 boxes_tensor = np$concatenate(boxes_tensor, axis = 1L) # Concatenate the anchor tensors from the individual layers to one

                                 # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
                                 #    It will contain all zeros for now, the classes will be set in the matching process that follows
                                 classes_tensor = np$zeros(reticulate::tuple(batch_size, dim(boxes_tensor)[2], self$n_classes))

                                 # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
                                 #    contains the same 4 variance values for every position in the last axis.
                                 variances_tensor = np$zeros_like(boxes_tensor)
                                 variances_tensor = variances_tensor + self$variances # Long live broadcasting

                                 # variances_tensor = array(rep(variances, each = prod(feature_map_size, self$n_boxes)), dim = dim(boxes_tensor)) # Has shape `(feature_map_height, feature_map_width, n_boxes, 4)`

                                 # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
                                 #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encode_template` has the same
                                 #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
                                 #    `boxes_tensor` a second time.
                                 y_encode_template = np$concatenate(reticulate::tuple(classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2L)

                                 if (diagnostics) {
                                   return(list(
                                     y_encode_template = y_encode_template,
                                     wh_list = wh_list,
                                     cell_sizes  = cell_sizes))
                                 } else {
                                   return(y_encode_template)
                                 }
                               },
                               encode_y = function(ground_truth_labels) {
                                 np <- reticulate::import("numpy")
                                 # 1: Generate the template for y_encoded
                                 y_encode_template = self$generate_encode_template(batch_size = length(ground_truth_labels), diagnostics = FALSE)
                                 y_encoded = np$copy(y_encode_template) # We'll write the ground truth box data to this array

                                 # 2: Match the boxes from `ground_truth_labels` to the anchor boxes in `y_encode_template`
                                 #    and for each matched box record the ground truth coordinates in `y_encoded`.
                                 #    Every time there is no match for a anchor box, record `class_id` 0 in `y_encoded` for that anchor box.

                                 class_vector = np$eye(self$n_classes) # An identity matrix that we'll use as one-hot class vectors

                                 for (i in seq_len(dim(y_encode_template)[1])) { # For each batch item...
                                   available_boxes = np$ones(dim(y_encode_template)[2]) # 1 for all anchor boxes that are not yet matched to a ground truth box, 0 otherwise
                                   negative_boxes = np$ones(dim(y_encode_template)[2]) # 1 for all negative boxes, 0 otherwise
                                   for (true_box_idx in seq_len(nrow(ground_truth_labels[[i]]))) { # For each ground truth box belonging to the current batch item...
                                     true_box = as.double(ground_truth_labels[[i]][true_box_idx,])
                                     if (abs(true_box[3] - true_box[2]) < 0.001 || abs(true_box[5] - true_box[4]) < 0.001) next() # Protect ourselves against bad ground truth data: boxes with width or height equal to zero
                                     if (self$normalize_coords) {
                                       true_box[2:3] = true_box[2:3] / self$img_width # Normalize xmin and xmax to be within [0,1]
                                       true_box[4:5] = true_box[4:5] / self$img_height # Normalize ymin and ymax to be within [0,1]
                                     }
                                     if (self$coords == 'centroids') {
                                       true_box = convert_coordinates1D(true_box, start_index = 2L, conversion = 'minmax2centroids')
                                     }
                                     similarities = iou(y_encode_template[i, , (self$n_classes + 1):(self$n_classes + 4)], true_box[-1], coords = self$coords) # The iou similarities for all anchor boxes
                                     negative_boxes[similarities >= self$neg_iou_threshold] = 0 # If a negative box gets an IoU match >= `self.neg_iou_threshold`, it's no longer a valid negative box
                                     similarities = similarities * available_boxes # Filter out anchor boxes which aren't available anymore (i.e. already matched to a different ground truth box)
                                     available_and_thresh_met = np$copy(similarities)
                                     available_and_thresh_met[available_and_thresh_met < self$pos_iou_threshold] = 0 # Filter out anchor boxes which don't meet the iou threshold
                                     assign_indices = np$nonzero(available_and_thresh_met)[[1]] + 1 # Get the indices of the left-over anchor boxes to which we want to assign this ground truth box
                                     if (length(assign_indices) > 0) { # If we have any matches
                                       y_encoded[i,assign_indices,1:(self$n_classes + 4)] = rep(
                                         np$concatenate(reticulate::tuple(class_vector[true_box[1] + 1,], true_box[-1]), axis = 0L),
                                         each = length(assign_indices))# Write the ground truth box coordinates and class to all assigned anchor box positions. Remember that the last four elements of `y_encoded` are just dummy entries.
                                       available_boxes[assign_indices] = 0 # Make the assigned anchor boxes unavailable for the next ground truth box
                                     } else { # If we don't have any matches
                                       best_match_index = np$argmax(similarities) + 1 # Get the index of the best iou match out of all available boxes
                                       y_encoded[i, best_match_index,1:(self$n_classes + 4)] = np$concatenate(reticulate::tuple(class_vector[true_box[1] + 1, ], true_box[-1]), axis=0L) # Write the ground truth box coordinates and class to the best match anchor box position
                                       available_boxes[best_match_index] = 0 # Make the assigned anchor box unavailable for the next ground truth box
                                       negative_boxes[best_match_index] = 0 # The assigned anchor box is no longer a negative box
                                     }
                                   }
                                   # Set the classes of all remaining available anchor boxes to class zero
                                   background_class_indices = np$nonzero(negative_boxes)[[1]] + 1
                                   y_encoded[i, background_class_indices, 1] = 1
                                 }

                                 # 3: Convert absolute box coordinates to offsets from the anchor boxes and normalize them
                                 if (self$coords == 'centroids') {
                                   y_encoded[, , c(self$n_classes + 1, self$n_classes + 2)] =
                                     y_encoded[, , c(self$n_classes + 1, self$n_classes + 2)] - y_encode_template[, , c(self$n_classes + 1, self$n_classes + 2)] # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
                                   y_encoded[, , c(self$n_classes + 1, self$n_classes + 2)] =
                                     y_encoded[, , c(self$n_classes + 1, self$n_classes + 2)] /
                                     (y_encode_template[, , c(self$n_classes + 3, self$n_classes + 4)] *
                                        y_encode_template[, , c(self$n_classes + 9, self$n_classes + 10)]) # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
                                   y_encoded[, , c(self$n_classes + 3, self$n_classes + 4)] =
                                     y_encoded[, , c(self$n_classes + 3, self$n_classes + 4)] / y_encode_template[, , c(self$n_classes + 3, self$n_classes + 4)] # w(gt) / w(anchor), h(gt) / h(anchor)
                                   y_encoded[, , c(self$n_classes + 3, self$n_classes + 4)] = np$log(y_encoded[, , c(self$n_classes + 3, self$n_classes + 4)]) /
                                     y_encode_template[, , c(self$n_classes + 11, self$n_classes + 12)] # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
                                 } else {
                                   y_encoded[, , (self$n_classes + 1):(self$n_classes + 4)] =
                                     y_encoded[, , (self$n_classes + 1):(self$n_classes + 4)] - y_encode_template[, , (self$n_classes + 1):(self$n_classes + 4)] # (gt - anchor) for all four coordinates
                                   y_encoded[, , c(self$n_classes + 1, self$n_classes + 2)] =
                                     y_encoded[, , c(self$n_classes + 1, self$n_classes + 2)] /
                                       np$expand_dims(y_encode_template[, , self$n_classes + 2] -
                                                        y_encode_template[, , self$n_classes + 1], axis = -1L) # (xmin(gt) - xmin(anchor)) / w(anchor), (xmax(gt) - xmax(anchor)) / w(anchor)
                                   y_encoded[, , c(self$n_classes + 3, self$n_classes + 4)] =
                                     y_encoded[, , c(self$n_classes + 3, self$n_classes + 4)] /
                                     np$expand_dims(y_encode_template[, , self$n_classes + 4] -
                                                      y_encode_template[, , self$n_classes + 3], axis = -1L) # (ymin(gt) - ymin(anchor)) / h(anchor), (ymax(gt) - ymax(anchor)) / h(anchor)
                                   y_encoded[, , (self$n_classes + 1):(self$n_classes + 4)] =
                                     y_encoded[, , (self$n_classes + 1):(self$n_classes + 4)] / y_encode_template[, , (self$n_classes + 9):(self$n_classes + 12)] # (gt - anchor) / size(anchor) / variance for all four coordinates, where 'size' refers to w and h respectively
                                 }

                                 return(y_encoded)
                               }
                             ))
